"""P0d(synth_nam_vc §8.4): 描画器 rvoc3hi(ema 330k)を P0a の前段 v3 の条件で追い学習する(学習と推論の一致・写しは識別可能)。
データ = P1 の学習の声(train_snv_p1.split)の E1 の普通の発話 + 音素の無い発声(data/snv_e1・data/snv_e1nv の 48 kHz と data/snv_p_feat の特徴)。
条件 = snv_p0_front.cond(平滑なし = 回路 C の Y の空間と同じ)・励起 = 前段 v3 の f0 のパルス列 + 雑音。区間 0.7 s・先頭 200 ms は損失と判別器から外す。
損失は rvoc3hi と同じ(45·多尺度 logmel L1 + LSGAN(MPD + MRD-log)+ 2·FM + 0.5·変調の線の罰則・両方向)。生成器・判別器・最適化の状態・EMA を 330k から引き継ぐ。
事前登録 = results/<tag>/prereg.yaml。
    cd training && uv run --no-sync python train_snv_p0d.py --tag snv_p0d_rvoc --steps 10000
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import nvoc as N
import rvoc as R
import snv_p0_front as P
import train_snv_p1 as T
from am_loss import am_penalty
from train_nvoc import MRDLog
from train_rvoc import atomic_save, excitation
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1

ROOT = Path(__file__).resolve().parent.parent
CK = ROOT / "results/rvoc3hi/last.pt"
SEG = 33600
WARM = 9600
NF = SEG // N.HOP


def items_for(voices: list[str]) -> list[tuple[Path, Path]]:
    out = []
    for v in voices:
        for w in sorted((ROOT / "data/snv_e1" / v).glob("t*.wav")):
            out.append((w, T.FEAT / f"e1/{v}/{w.stem}.npz"))
        for w in sorted((ROOT / "data/snv_e1nv" / v).glob("*.wav")):
            out.append((w, T.FEAT / f"e1nv/{v}/{w.stem}.npz"))
    return [(w, f) for w, f in out if f.exists()]


class DS(torch.utils.data.IterableDataset):
    def __init__(self, items: list, seed: int) -> None:
        self.items, self.seed = items, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1000)
        while True:
            w, f = self.items[rng.randrange(len(self.items))]
            d = np.load(f)
            x, sr = sf.read(str(w), dtype="float32")
            assert sr == P.SR
            Tn = min(len(d["f0"]), len(x) // N.HOP)
            if Tn < NF + 2:
                continue
            s = rng.randint(1, Tn - NF)
            sl = slice(s, s + NF)
            c = P.cond({"f0": d["f0"][sl], "env": d["env"][:, sl], "per": d["per"][:, sl]})
            tgt = x[s * N.HOP - N.DELAY: s * N.HOP - N.DELAY + SEG]
            if len(tgt) < SEG:
                continue
            yield torch.from_numpy(c), torch.from_numpy(d["f0"][sl].astype(np.float32)), torch.from_numpy(tgt.astype(np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text()) if (ROOT / "results/snv_p1_n/split.json").exists() else None
    voices = split["train"] if split else T.split(sorted(p.name for p in (T.FEAT / "e1").iterdir()))[0]
    items = items_for(voices[:20] if a.smoke else voices)
    print("items", len(items), "voices", len(voices), flush=True)
    st = torch.load(CK, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    gen = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(dev)
    gen.load_state_dict(st["gen"])
    from bigvgan.discriminators import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    mpd.load_state_dict(st["mpd"]), mrd.load_state_dict(st["mrd"])
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    opt_g = torch.optim.AdamW(gen.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(dparams, lr=2e-4, betas=(0.8, 0.99))
    opt_g.load_state_dict(st["opt_g"]), opt_d.load_state_dict(st["opt_d"])
    print("resume rvoc3hi step", st["step"], "lr", opt_g.param_groups[0]["lr"], "env_smooth(元)", st.get("env_smooth"), flush=True)
    ema = {k: v.to(dev) for k, v in st["ema"].items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    loader = iter(torch.utils.data.DataLoader(DS(items, 7), batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4))
    log = open(out / "train.jsonl", "a")
    acc: dict = {}
    t0 = time.time()
    total = 30 if a.smoke else a.steps
    for step in range(1, total + 1):
        cond, f0, tgt = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            exc = excitation(f0, SEG)
        y = gen(cond, exc)
        yl, tl = y[:, WARM:], tgt[:, WARM:]
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        pen, _ = am_penalty(yl, tl, 0.3, 0.0, True)
        loss = 45.0 * lm + 0.5 * pen
        off = random.randrange(0, yl.shape[-1] - 16384)
        r_, f_ = tl[:, off:off + 16384], yl[:, off:off + 16384]
        opt_d.zero_grad(set_to_none=True)
        a1, b1, _, _ = mpd(r_[:, None], f_.detach()[:, None])
        a2, b2, _, _ = mrd(r_, f_.detach())
        dl = sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(a1 + a2, b1 + b2))
        if torch.isfinite(dl):
            dl.backward()
            torch.nn.utils.clip_grad_norm_(dparams, 500.0)
            opt_d.step()
        for p_ in dparams:
            p_.requires_grad_(False)
        _, b1, f1r, f1g = mpd(r_[:, None], f_[:, None])
        _, b2, f2r, f2g = mrd(r_, f_)
        for p_ in dparams:
            p_.requires_grad_(True)
        adv = sum(((q.float() - 1) ** 2).mean() for q in b1 + b2)
        fm = sum(F.l1_loss(u.detach().float(), v.float()) for xr, xg in zip(f1r + f2r, f1g + f2g) for u, v in zip(xr, xg))
        loss = loss + adv + 2 * fm
        opt_g.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(gen.parameters(), 500.0)):
            opt_g.zero_grad(set_to_none=True)
            continue
        opt_g.step()
        with torch.no_grad():
            for k, v in gen.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k_, v_ in {"lm": lm, "am_pen": pen, "adv": adv, "fm": fm, "dl": dl}.items():
            acc.setdefault(k_, []).append(float(v_.detach()))
        if step % 200 == 0 or a.smoke:
            rr = {"step": step, "min": round((time.time() - t0) / 60, 1), **{k_: round(float(np.mean(v_)), 4) for k_, v_ in acc.items()}}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % 2500 == 0 or step == total:
            (out / "snap").mkdir(exist_ok=True)
            atomic_save({"ema": ema, "step": st["step"] + step, "cfg": gen.cfg, "cond": "pae", "env_smooth": 0.0, "front": "snv_p0_front v3"}, out / "snap" / f"ema_{step // 1000}k.pt")
            atomic_save({"gen": gen.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                         "step": st["step"] + step, "cfg": gen.cfg, "cond": "pae", "env_smooth": 0.0, "front": "snv_p0_front v3"}, out / "last.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
