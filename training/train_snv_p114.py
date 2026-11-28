"""段 P114(paper_snv): 描画器の励起を「複数の周期」に広げる学び足し。rvoc3hi(330k)の励起の入力(高さ f0 のパルス列・雑音)に、2 倍の高さ・倍の周期(f0 / 2)・3 倍の周期(f0 / 3)のパルス列(位相は主のパルス列にそろう)と、
教師の音から測った周期ごとの強さ(data/snv_p114_sal = 周期 T0 の 1/2・1・2・3 倍での周期性・フレームごと)を足す。条件は前段の読み(--ycond 0)。--msub 0 = 足さない対照(同じデータ・同じ歩数)。
時刻の規約・損失・データ・途中の測りは train_snv_p97 と同じ。足した入力の重みは 0 から始める(起動時の関数 = rvoc3hi)。
    cd training && uv run --no-sync python train_snv_p114.py --tag snv_p114_m1 --msub 1 --delay 240 --smooth 0.25 --steps 10000
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
from train_snv_p97 import CK, NF, SEG, SHIFTS, WARM, YC, feat, feat_y, items_for

ROOT = Path(__file__).resolve().parent.parent
SAL = ROOT / "data/snv_p114_sal"
RES = ROOT / "data/snv_p116_res"
RES_GAIN = 0.2
PDROP = [0.0]
N_SAL = 4
SUBS = (2.0, 0.5, 1.0 / 3.0)


def cue_path(f: Path, side: str = "t") -> Path:
    return SAL / f.relative_to(T.FEAT).with_suffix(".npy")


def res_path(f: Path) -> Path:
    return RES / f.relative_to(T.FEAT).with_suffix(".flac")


def fast_rows(u: np.ndarray, T_: int, delay: int = 0) -> np.ndarray:
    return np.pad(u, ((0, 0), (0, max(0, T_ - u.shape[1]))))[:, :T_].astype(np.float32)


def lag(c: np.ndarray, n: int) -> np.ndarray:
    return c if n == 0 else np.concatenate([np.repeat(c[..., :1], n, -1), c[..., :-n]], -1)


def exc_all(f0: torch.Tensor, fast: torch.Tensor, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
    base = excitation(f0, n, gen)
    if fast.shape[1] == 0:
        return base
    if fast.shape[1] == 1:
        if PDROP[0] >= 1.0:
            base = torch.cat([torch.zeros_like(base[:, :1]), base[:, 1:]], 1)
        return torch.cat([base, fast.to(base.device)[..., :n] * RES_GAIN], 1)
    tr = [N.harmonic_source(f0 * m)[:, :n] * 0.1 for m in SUBS]
    sal = fast.to(base.device).repeat_interleave(N.HOP, -1)[..., :n] * 0.1
    return torch.cat([base, torch.stack(tr, 1), sal], 1)


class DS(torch.utils.data.IterableDataset):
    def __init__(self, items: list, seed: int, a: argparse.Namespace) -> None:
        self.items, self.seed, self.a = items, seed, a

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1000)
        a = self.a
        while True:
            w, f = self.items[rng.randrange(len(self.items))]
            ym = rng.random() < a.ycond
            d = feat_y(f, a.smooth) if ym else feat(f, a.smooth)
            l = np.load(cue_path(f)).astype(np.float32) if a.msub == 1 else np.zeros((0, len(d["f0"])), np.float32)
            x, sr = sf.read(str(w), dtype="float32")
            assert sr == P.SR
            Tn = min(len(d["f0"]), len(x) // N.HOP, l.shape[1])
            rr = sf.read(str(res_path(f)), dtype="float32")[0] if a.msub == 2 else None
            if Tn < NF + 2 + a.ylag:
                continue
            s = rng.randint(1 + a.ylag, Tn - NF)
            tgt = x[s * N.HOP - a.delay: s * N.HOP - a.delay + SEG]
            if len(tgt) < SEG:
                continue
            fr = l[:, s - a.ylag: s - a.ylag + NF]
            if rr is not None:
                fr = rr[s * N.HOP - a.delay: s * N.HOP - a.delay + SEG][None]
                if fr.shape[1] < SEG:
                    continue
            c, f0 = d["c"][:, s - a.ylag: s - a.ylag + NF], d["f0"][s - a.ylag: s - a.ylag + NF]
            yield torch.from_numpy(c.copy()), torch.from_numpy(f0.copy()), torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(fr.copy())


def val_items(voices: list[str], a: argparse.Namespace, y: bool = False) -> list[dict]:
    out = []
    for w, f in items_for(voices, nv=False):
        if w.stem != "t09":
            continue
        d = feat_y(f, a.smooth) if y else feat(f, a.smooth)
        x, _ = sf.read(str(w), dtype="float32")
        l = np.load(cue_path(f)).astype(np.float32) if a.msub == 1 else np.zeros((0, len(d["f0"])), np.float32)
        n = min(len(d["f0"]), len(x) // N.HOP, l.shape[1])
        if a.msub == 2:
            rr = sf.read(str(res_path(f)), dtype="float32")[0]
            out.append({"x": x[:n * N.HOP].astype(np.float32), "c": lag(d["c"][:, :n], a.ylag), "f0": lag(d["f0"][:n], a.ylag), "fast": np.concatenate([np.zeros(a.delay, np.float32), rr])[: n * N.HOP][None].astype(np.float32)})
            continue
        out.append({"x": x[:n * N.HOP].astype(np.float32), "c": lag(d["c"][:, :n], a.ylag), "f0": lag(d["f0"][:n], a.ylag), "fast": lag(fast_rows(l, n), a.ylag)})
    return out


def render(gen: R.RVoc, it: dict, dev: str) -> np.ndarray:
    c = torch.from_numpy(it["c"])[None].to(dev)
    f0 = torch.from_numpy(it["f0"])[None].to(dev)
    return gen(c, exc_all(f0, torch.from_numpy(it["fast"])[None].to(dev), c.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()


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
    with torch.no_grad():
        for p in g.src:
            v = p.conv.parametrizations.weight.original1
            v[:, 2:] = torch.randn_like(v[:, 2:]) * v[:, :2].std()
    worst = 0
    with torch.no_grad():
        ce = []
        for it in items[:2]:
            f = torch.from_numpy(it["f0"])[None]
            c = torch.from_numpy(it["c"])[None]
            ce.append((c, exc_all(f, torch.from_numpy(it["fast"])[None], c.shape[-1] * N.HOP, torch.Generator().manual_seed(0))))
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
            if e1.shape[1] > 2:
                eb = e1.clone()
                eb[:, 2:, k * N.HOP:] = e2[:, 2:, k * N.HOP:]
                db = (g(c1, eb)[0] - y).abs()
                nb = (db > 0).nonzero()
                assert len(nb) > 0, "足した励起の行の書き換えが出力を動かさない = INCONCLUSIVE"
                worst = max(worst, k * N.HOP - int(nb[0]))
    la = max(0, worst)
    print(f"  生成器の未来不変性(検証の声の条件・励起・速い道の行・足した重みは乱数で開く・1 bit): 先読み {la} sample", flush=True)
    print("  周期ごとの強さ(snv_p114_sal)= 前段の高さの追跡と同じ左寄せの窓の差分関数: 先読み 0・発話全体の統計なし", flush=True)
    return la == 0 and SC.ledger([("rvoc 生成器(条件・励起・速い道の行の書き換え・検証の声)", la / N.SR * 1000), (f"出力の時刻の規約 delay {delay}", delay / N.SR * 1000), ("ブロック HOP 240", 5.0)])


def widen(sd: dict, d_exc: int) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("src.") and k.endswith("original1") and v.shape[1] < d_exc:
            v = torch.cat([v, torch.zeros(v.shape[0], d_exc - v.shape[1], v.shape[2], dtype=v.dtype)], 1)
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--ycond", type=float, default=0.0)
    ap.add_argument("--msub", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--pdrop", type=float, default=0.0)
    ap.add_argument("--ylag", type=int, default=0)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--snap_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert a.delay in (0, N.DELAY)
    PDROP[0] = a.pdrop
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
    vv = split["val"][:4] if a.smoke else split["val"]
    assert a.ycond == 0
    miss = [f for _, f in items if (a.msub == 1 and not cue_path(f).exists()) or (a.msub == 2 and not res_path(f).exists())]
    assert not miss, f"周期ごとの強さが無い発話 {len(miss)} 本(snv_p114_sal.py)"
    vals = val_items(vv, a)
    valy = val_items(vv, a, True) if a.ycond > 0 else []
    print("items", len(items), "voices", len(split["train"]), "val", len(vals), flush=True)
    resume = a.resume and (out / "last.pt").exists()
    st = torch.load(out / "last.pt" if resume else CK, map_location="cpu", weights_only=False)
    it0 = int(st.get("it", 0)) if resume else 0
    base = int(st["step"]) - it0
    cfg = st["cfg"]
    d_exc = 2 + (len(SUBS) + N_SAL if a.msub == 1 else 1 if a.msub == 2 else 0)
    mk = lambda: R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"], d_exc=d_exc).to(dev)
    gen = mk()
    gen.load_state_dict(widen(st["gen"], d_exc))
    from bigvgan.discriminators import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    mpd.load_state_dict(st["mpd"]), mrd.load_state_dict(st["mrd"])
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    opt_g = torch.optim.AdamW(gen.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(dparams, lr=2e-4, betas=(0.8, 0.99))
    opt_g.load_state_dict(st["opt_g"]), opt_d.load_state_dict(st["opt_d"])
    for p in gen.parameters():
        s_ = opt_g.state.get(p)
        if s_ is not None and s_["exp_avg"].shape != p.shape:
            for k_ in ("exp_avg", "exp_avg_sq"):
                s_[k_] = torch.cat([s_[k_], torch.zeros(p.shape[0], p.shape[1] - s_[k_].shape[1], p.shape[2], device=s_[k_].device, dtype=s_[k_].dtype)], 1)
    ema = {k: v.to(dev) for k, v in widen(st["ema"], d_exc).items()}
    print("start from", "results/" + a.tag + "/last.pt" if resume else "results/rvoc3hi/last.pt", "base step", base, "it", it0, "lr", opt_g.param_groups[0]["lr"], "d_exc", d_exc, flush=True)
    if not ship_gate(gen, vals, a.delay):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    evg = mk().eval()
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
        meta = {"step": base + it, "it": it, "cfg": gen.cfg, "cond": "pae", "env_smooth": a.smooth, "delay": a.delay, "ycond": a.ycond, "msub": a.msub, "pdrop": a.pdrop, "d_exc": d_exc, "ylag": a.ylag, "front": "snv_p0_front v3"}
        if snap:
            (out / "snap").mkdir(exist_ok=True)
            atomic_save({"ema": ema, **meta}, out / "snap" / f"ema_{it // 1000}k.pt")
        atomic_save({"gen": gen.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), **meta}, out / "last.pt")

    if it0 == 0:
        run_eval(0)
    loader = iter(torch.utils.data.DataLoader(DS(items, 7 + it0, a), batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4))
    acc: dict = {}
    t0 = time.time()
    total = it0 + 30 if a.smoke else a.steps
    for it in range(it0 + 1, total + 1):
        cond, f0, tgt, fr = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            exc = exc_all(f0, fr, SEG)
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
