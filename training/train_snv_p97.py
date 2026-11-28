"""段 P97(paper_snv): 描画器 rvoc3hi(330k・生成器・判別器・最適化の状態・EMA)を、製品が渡す条件の分布と、遅れの少ない界面で学び足す。
データ = P1 の学習の声 564 の E1 の普通の発話 + 音素の無い発声(data/snv_p97_wav = data/snv_e1・data/snv_e1nv の可逆の写し・48 kHz)と data/snv_p_feat の前段 v3 の特徴。
条件 = snv_p0_front.cond(--smooth w = 包絡の行に因果の 3 点の平滑 [w, 1 − 2w, w]・0 = 平滑なし)・励起 = 前段 v3 の f0 のパルス列 + 雑音。
--ycond p(step b)= 確率 p で、条件を「模型の出力」(data/snv_p97_y = snv_p97_ycond.py・擬似の操作者の版を K3G で同じ声へ変換した Y と、門つきの前段の高さ ÷ r)に替える。教師は同じ元の音。
時刻の規約 = 出力 y[m] ≈ x[m − delay](--delay 240 = 今の製品・0 = 出力の塊 t が入力の塊 t の中身を描く)。区間 0.7 s・先頭 200 ms は損失と判別器から外す。
損失は rvoc3hi と同じ(45·多尺度 logmel L1 + LSGAN(MPD + MRD-log)+ 2·FM + 0.5·変調の線の罰則・両方向)。
途中の測り(検証の 36 声の t09 の写し・EMA): logmel・PESQ・hf・変調の線・出力が揃う時刻(logmel が最小になるずらし)。
事前登録 = results/<tag>/prereg.yaml。起動時に生成器の未来不変性(CPU・検証の声の条件と励起の書き換え・1 bit)と静的な遅延の台帳を出す。
    cd training && uv run --no-sync python train_snv_p97.py --tag snv_p97_n0 --smooth 0 --delay 0 --steps 30000
"""
from __future__ import annotations

import argparse
import copy
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
WAV = ROOT / "data/snv_p97_wav"
YC = ROOT / "data/snv_p97_y"
SEG = 33600
WARM = 9600
NF = SEG // N.HOP
SHIFTS = tuple(range(0, 601, 60))


def items_for(voices: list[str], nv: bool = True) -> list[tuple[Path, Path]]:
    out = []
    for v in voices:
        for w in sorted((WAV / "e1" / v).glob("t*.flac")):
            out.append((w, T.FEAT / f"e1/{v}/{w.stem}.npz"))
        if nv:
            for w in sorted((WAV / "e1nv" / v).glob("*.flac")):
                out.append((w, T.FEAT / f"e1nv/{v}/{w.stem}.npz"))
    return [(w, f) for w, f in out if f.exists()]


def feat(f: Path, smooth: float) -> dict:
    d = np.load(f)
    return {"f0": d["f0"].astype(np.float32), "c": P.cond({"f0": d["f0"], "env": d["env"], "per": d["per"]}, smooth=smooth)}


def feat_y(f: Path, smooth: float) -> dict:
    d = np.load(YC / f.relative_to(T.FEAT))
    y, f0 = d["y"], d["f0"].astype(np.float32)
    v = (f0 > 0).astype(np.float32)[None]
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
    env = P.causal_smooth(y[:25], smooth) if smooth > 0 else y[:25]
    return {"f0": f0, "c": np.concatenate([env, lf, v, np.clip(y[25:29], 0, 1.5) * v], 0).astype(np.float32)}


REC = ROOT / "data/snv_p_feat_rec"


def rec_scenes(rec: Path) -> dict:
    r = json.loads((rec / "rooms.json").read_text())
    ok = lambda q: q["bg"] == "none" or (q["bg"] == "noise" and q["snr_db"] >= 15.0)
    return {v: [k for k, q in d.items() if ok(q)] for v, d in r.items()}


def feat_rec(f: Path, smooth: float, k: str, rec: Path = REC) -> dict | None:
    q = rec / f.relative_to(T.FEAT).parent / f"{f.stem}_k{k}.npz"
    if not q.exists():
        return None
    d, e = np.load(f), np.load(q)
    n = min(d["env"].shape[1], e["env"].shape[1])
    env, c0 = e["env"][:, :n].astype(np.float32).copy(), d["env"][0, :n]
    m = c0 >= np.percentile(c0, 60)
    env[0] -= float(np.median((env[0] - c0)[m]))
    f0 = d["f0"][:n]
    return {"f0": f0.astype(np.float32), "c": P.cond({"f0": f0, "env": env, "per": e["per"][:, :n]}, smooth=smooth)}


class DS(torch.utils.data.IterableDataset):
    def __init__(self, items: list, seed: int, smooth: float, delay: int, ycond: float = 0.0, rec: float = 0.0, recdir: str = "data/snv_p_feat_rec") -> None:
        self.items, self.seed, self.smooth, self.delay, self.ycond, self.rec = items, seed, smooth, delay, ycond, rec
        self.recdirs = [ROOT / q for q in recdir.split(",")]
        self.scenes = [rec_scenes(q) for q in self.recdirs] if rec > 0 else []

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1000)
        while True:
            w, f = self.items[rng.randrange(len(self.items))]
            d = None
            if self.rec > 0 and rng.random() < self.rec:
                j = rng.randrange(len(self.recdirs))
                ks = self.scenes[j].get(f.parent.name, [])
                if ks:
                    d = feat_rec(f, self.smooth, rng.choice(ks), self.recdirs[j])
            if d is None:
                d = feat_y(f, self.smooth) if rng.random() < self.ycond else feat(f, self.smooth)
            x, sr = sf.read(str(w), dtype="float32")
            assert sr == P.SR
            Tn = min(len(d["f0"]), len(x) // N.HOP)
            if Tn < NF + 2:
                continue
            s = rng.randint(1, Tn - NF)
            tgt = x[s * N.HOP - self.delay: s * N.HOP - self.delay + SEG]
            if len(tgt) < SEG:
                continue
            yield torch.from_numpy(d["c"][:, s:s + NF].copy()), torch.from_numpy(d["f0"][s:s + NF].copy()), torch.from_numpy(tgt.astype(np.float32))


def val_items(voices: list[str], smooth: float, y: bool = False) -> list[dict]:
    out = []
    for w, f in items_for(voices, nv=False):
        if w.stem != "t09":
            continue
        d = feat_y(f, smooth) if y else feat(f, smooth)
        x, _ = sf.read(str(w), dtype="float32")
        n = min(len(d["f0"]), len(x) // N.HOP)
        out.append({"x": x[:n * N.HOP].astype(np.float32), "c": d["c"][:, :n], "f0": d["f0"][:n]})
    return out


def render(gen: R.RVoc, it: dict, dev: str) -> np.ndarray:
    c = torch.from_numpy(it["c"])[None].to(dev)
    f0 = torch.from_numpy(it["f0"])[None].to(dev)
    return gen(c, excitation(f0, c.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()


@torch.no_grad()
def evaluate(gen: R.RVoc, items: list[dict], delay: int, mels: list, dev: str) -> dict:
    import eval_nvoc as E
    rows, sh = [], {d: [] for d in SHIFTS}
    for it in items:
        y = render(gen, it, dev)
        rows.append(E.metrics(y, it["x"], delay, dev))
        yt, xt = torch.from_numpy(y).to(dev), torch.from_numpy(it["x"]).to(dev)
        for d in SHIFTS:
            n = min(len(y) - d, len(it["x"]))
            sh[d].append(float(logmel_l1(yt[None, None, d:d + n], xt[None, None, :n], mels)))
    out = {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in rows[0]}
    m = {d: round(float(np.mean(v)), 4) for d, v in sh.items()}
    out["logmel_by_shift"] = m
    out["align_best"] = min(m, key=m.get)
    return out


def ship_gate(gen: R.RVoc, items: list[dict], delay: int) -> bool:
    import ship_check as SC
    g = copy.deepcopy(gen).cpu().eval()
    worst = 0
    with torch.no_grad():
        ce = []
        for it in items[:2]:
            f = torch.from_numpy(it["f0"])[None]
            c = torch.from_numpy(it["c"])[None]
            ce.append((c, excitation(f, c.shape[-1] * N.HOP, torch.Generator().manual_seed(0))))
        (c1, e1), (c2, e2) = ce
        n = min(c1.shape[-1], c2.shape[-1], 600)
        c1, e1, c2, e2 = c1[..., :n], e1[..., :n * N.HOP], c2[..., :n], e2[..., :n * N.HOP]
        y = g(c1, e1)[0]
        for k in (97, 211, 350, 503):
            ca, ea = c1.clone(), e1.clone()
            ca[..., k:], ea[..., k * N.HOP:] = c2[..., k:], e2[..., k * N.HOP:]
            d = (g(ca, ea)[0] - y).abs()
            nz = (d > 0).nonzero()
            assert len(nz) > 0, "書き換えが出力を動かさない = INCONCLUSIVE"
            worst = max(worst, k * N.HOP - int(nz[0]))
    la = max(0, worst)
    print(f"  生成器の未来不変性(検証の声の条件・励起・1 bit): 先読み {la} sample", flush=True)
    return la == 0 and SC.ledger([("rvoc 生成器(条件・励起の書き換え・検証の声)", la / N.SR * 1000), (f"出力の時刻の規約 delay {delay}", delay / N.SR * 1000), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--delay", type=int, default=N.DELAY)
    ap.add_argument("--ycond", type=float, default=0.0)
    ap.add_argument("--rec", type=float, default=0.0)
    ap.add_argument("--recdir", default="data/snv_p_feat_rec")
    ap.add_argument("--snap_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert 0 <= a.delay <= N.DELAY
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    print("args", json.dumps(vars(a)), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    items = items_for(split["train"][:20] if a.smoke else split["train"])
    vals = val_items(split["val"][:4] if a.smoke else split["val"], a.smooth)
    valy = val_items(split["val"][:4] if a.smoke else split["val"], a.smooth, True) if a.ycond > 0 else []
    if a.ycond > 0:
        miss = [f for _, f in items if not (YC / f.relative_to(T.FEAT)).exists()]
        assert not miss, f"模型の出力の条件が無い発話 {len(miss)} 本(snv_p97_ycond.py)"
    print("items", len(items), "voices", len(split["train"]), "val", len(vals), flush=True)
    resume = a.resume and (out / "last.pt").exists()
    st = torch.load(out / "last.pt" if resume else CK, map_location="cpu", weights_only=False)
    it0 = int(st.get("it", 0)) if resume else 0
    base = int(st["step"]) - it0
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
    ema = {k: v.to(dev) for k, v in st["ema"].items()}
    print("start from", "results/" + a.tag + "/last.pt" if resume else "results/rvoc3hi/last.pt", "base step", base, "it", it0, "lr", opt_g.param_groups[0]["lr"], "env_smooth(元)", st.get("env_smooth"), flush=True)
    if not ship_gate(gen, vals, a.delay):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    evg = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(dev).eval()
    log = open(out / "train.jsonl", "a")

    def run_eval(it: int) -> None:
        evg.load_state_dict(ema)
        r = {"eval_it": it, **evaluate(evg, vals, a.delay, mels, dev)}
        if valy:
            r["y"] = evaluate(evg, valy, a.delay, mels, dev)
        print("eval", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()

    def save(it: int, snap: bool) -> None:
        meta = {"step": base + it, "it": it, "cfg": gen.cfg, "cond": "pae", "env_smooth": a.smooth, "delay": a.delay, "ycond": a.ycond, "rec": a.rec, "recdir": a.recdir, "front": "snv_p0_front v3"}
        if snap:
            (out / "snap").mkdir(exist_ok=True)
            atomic_save({"ema": ema, **meta}, out / "snap" / f"ema_{it // 1000}k.pt")
        atomic_save({"gen": gen.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), **meta}, out / "last.pt")

    if it0 == 0:
        run_eval(0)
    loader = iter(torch.utils.data.DataLoader(DS(items, 7 + it0, a.smooth, a.delay, a.ycond, a.rec, a.recdir), batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4))
    acc: dict = {}
    t0 = time.time()
    total = it0 + 30 if a.smoke else a.steps
    for it in range(it0 + 1, total + 1):
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
        if it % 200 == 0 or a.smoke:
            rr = {"it": it, "min": round((time.time() - t0) / 60, 1), **{k_: round(float(np.mean(v_)), 4) for k_, v_ in acc.items()}}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        snap = it % a.snap_every == 0 or it == total
        if snap:
            run_eval(it)
        if snap or it % 2500 == 0:
            save(it, snap)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
