"""D598(推論だけ): 版どうしの OE(奇数 / 偶数の倍音)・F1 の倍読み(入力の高さ r と 9 半音超)・事後のあいまいさのファイルごとの対の差(D594 と同じ定義・r も同じ)。
版の名前の k = OURS<k>… / REFKC<k> / EVEN<k>…(D599)。候補 = r は入力 SRC の harvest × 2^(k/12)(遅れ ±10)・REFKC と EVEN = 描画の輪郭。
    uv run python d598_arm_cmp.py --sets dev2 dev3 --pairs OURS5RY60G-OURS5RY58G OURS5RY60G-REFKC5 --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, SR, boot, harvest

ROOT = Path(__file__).resolve().parent.parent
NFFT = 2048


def prep_(x: np.ndarray) -> np.ndarray:
    x = x[: int(8 * SR)].astype(np.float32)
    return x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05


def load_job(job: tuple) -> tuple:
    from d6_unit_cov import load48
    kind, path = job
    if kind == "out":
        x = prep_(load48(path))
        return x, harvest(x)
    if kind == "outx":
        return prep_(load48(path)), None
    if kind == "src":
        return None, harvest(prep_(load48(path)))
    return None, harvest(load48(path).astype(np.float64)[: 8 * SR])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pl = Pool(NPROC)
    import torch
    import f0est as FE
    import nvoc as N
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())
    HOP = N.HOP

    def f0post(x: np.ndarray) -> tuple:
        n = len(x) // HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * HOP])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        f = np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12))
        return np.where(vo[:m] & (c >= 0.5), f, 0.0), pr[:, :m]

    def shift(r0: np.ndarray, lag: int) -> np.ndarray:
        r = np.roll(r0, lag)
        if lag > 0:
            r[:lag] = 0
        elif lag < 0:
            r[lag:] = 0
        return r

    def align(r0: np.ndarray, h: np.ndarray) -> np.ndarray:
        best = (0, -1.0)
        for lag in range(-10, 11):
            r = shift(r0, lag)
            n = min(len(r), len(h))
            b = (r[:n] > 0) & (h[:n] > 0)
            if b.sum() < 50:
                continue
            ag = float((np.abs(12 * np.log2(h[:n][b] / r[:n][b])) < 1).mean())
            if ag > best[1]:
                best = (lag, ag)
        return shift(r0, best[0])

    st12 = 12 * lhz / np.log(2)

    def stats(x: np.ndarray, r: np.ndarray) -> dict:
        fv, pr = f0post(x)
        _, _, Z = ss.stft(x.astype(np.float64), SR, nperseg=NFFT, noverlap=NFFT - 240)
        A = np.abs(Z)
        n = min(len(fv), len(r), A.shape[1])
        fv, r, pr = fv[:n], r[:n], pr[:, :n]
        bw = SR / NFFT
        vb = (fv > 0) & (r > 0)
        e = np.where(vb, 12 * np.log2(np.where(vb, fv, 1.0) / np.where(r > 0, r, 1.0)), 0.0)
        oe, amb = [], []
        for t in np.nonzero(vb)[0]:
            s0 = 12 * np.log2(r[t])
            p1 = pr[np.abs(st12 - s0) <= 1, t].sum()
            p2 = pr[np.abs(st12 - s0 - 12) <= 1, t].sum()
            amb.append(p2 / max(p1, 1e-9) > 0.3)

            def amp(f: float) -> float:
                i = int(round(f / bw))
                return float(A[max(i - 1, 0): i + 2, t].max()) if i + 1 < A.shape[0] else 0.0

            oe.append(sum(amp(k * r[t]) for k in (3, 5, 7)) / max(sum(amp(k * r[t]) for k in (2, 4, 6, 8)), 1e-12))
        return {"oe": float(np.median(oe)) if oe else float("nan"), "up": float((vb & (e > 9)).sum() / max(vb.sum(), 1)), "amb": float(np.mean(amb)) if amb else float("nan")}

    names = sorted({v for p in a.pairs for v in p.split("-")})
    rows: dict = {nm: {} for nm in names}
    for s in a.sets:
        S = H / s
        files = {nm: [(f, *re.match(r"(p\d+_t(\d+))_", f.name).groups()) for f in sorted((S / "d570" / nm).glob("p*_t*.wav"))] for nm in names}
        keys = sorted({key for v in files.values() for _, key, _ in v})
        tis = sorted({int(ti) for nm, v in files.items() if not nm.startswith("OURS") for _, _, ti in v})
        srcf = dict(zip(keys, (h for _, h in pl.map(load_job, [("src", S / "d56" / f"{key}_SRC.wav") for key in keys]))))
        cenf = dict(zip(tis, (h for _, h in pl.map(load_job, [("cen", S / "cen_reg" / f"cen_{ti:03d}.wav") for ti in tis]))))
        for nm in names:
            k = int(re.match(r"(?:OURS|REFKC|EVEN)(\d+)", nm).group(1))
            ours = nm.startswith("OURS")
            for (f, key, ti), (x, hx) in zip(files[nm], pl.imap(load_job, [("out" if ours else "outx", f) for f, _, _ in files[nm]])):
                ti = int(ti)
                if ours:
                    r = align(srcf[key] * 2 ** (k / 12), hx)
                else:
                    fc = cenf[ti]
                    fs = srcf[key]
                    r = fc * float(np.median(fs[fs > 0])) * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
                rows[nm][(s, key)] = {"t": ti, **stats(x, r)}
            print(s, nm, len(rows[nm]), flush=True)
    pl.close()
    rng = np.random.default_rng(598)
    rep: dict = {"sets": {nm: {st: round(float(np.nanmedian([r[st] for r in rows[nm].values()])), 4) for st in ("oe", "up", "amb")} | {"n": len(rows[nm])} for nm in names}, "pairs": {}}
    for nm in names:
        print(nm, rep["sets"][nm], flush=True)
    for p in a.pairs:
        A_, B_ = p.split("-")
        ks = sorted(set(rows[A_]) & set(rows[B_]))
        tg = np.array([f"{key[0]}|{rows[A_][key]['t']}" for key in ks])
        for st in ("oe", "up", "amb"):
            d = np.array([rows[A_][key][st] - rows[B_][key][st] for key in ks])
            ok = np.isfinite(d)
            rep["pairs"][f"{p}:{st}"] = {"n": int(ok.sum()), "diff": round(float(d[ok].mean()), 4), "ci95": boot(d[ok], tg[ok], rng)}
            print(f"{p}:{st}", rep["pairs"][f"{p}:{st}"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
