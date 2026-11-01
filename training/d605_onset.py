"""D605(推論だけ): 有声の立ち上がりで基音が遅れて立つか(因果の生成器の低い倍音の立ち上がり)。基準の高さ r(d598 と同じ)を使い、
フレーム中心 ±1 周期(2 周期の Hann 窓)で H1〜H3 を最小二乗で当てはめ、H1 − H2 dB を立ち上がりからの位置ごとに。
立ち上がり = r が 4 フレーム以上の無声の後に有声になるフレーム。onset = 位置 0〜3 の平均・steady = 位置 15〜30(その有声区間が続く範囲)の平均・Δ = onset − steady。
F1 の倍読み(F1 > r で 9 半音超)も位置ごとに。
    uv run python d605_onset.py --out ../results/conv_p0/d605/onset.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, SR, boot, harvest
from d598_arm_cmp import prep_

VERS = {"OURS5RY58G": 5, "OURS12RY58G": 12, "REFKC5": 5, "REFKC12": 12}
NPOS = 31


def harm_db(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = np.full((len(r), 3), np.nan)
    for i in np.nonzero(r > 0)[0]:
        T0 = SR / r[i]
        h = int(round(T0))
        c = i * 240
        if c - h < 0 or c + h >= len(x):
            continue
        n = np.arange(-h, h + 1)
        w = 0.5 + 0.5 * np.cos(np.pi * n / (h + 1))
        ph = 2 * np.pi * r[i] * n / SR
        B = np.stack([np.ones_like(ph)] + [f(k * ph) for k in (1, 2, 3) for f in (np.cos, np.sin)], 1)
        co, *_ = np.linalg.lstsq(B * w[:, None], x[c - h:c + h + 1] * w, rcond=None)
        out[i] = 20 * np.log10(np.hypot(co[1::2], co[2::2]) + 1e-9)
    return out


def onsets(r: np.ndarray) -> list:
    v = r > 0
    res = []
    for i in range(4, len(v)):
        if v[i] and not v[i - 4:i].any():
            j = i
            while j < len(v) and v[j]:
                j += 1
            res.append((i, j))
    return res


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


def job(it: tuple) -> tuple:
    from d6_unit_cov import load48
    nm, s, key, ti, k = it
    S = H / s
    x = prep_(load48(S / "d570" / nm / f"{key}_{nm}.wav")).astype(np.float64)
    hs = harvest(prep_(load48(S / "d56" / f"{key}_SRC.wav")))
    if nm.startswith("OURS"):
        r = align(hs * 2 ** (k / 12), harvest(x))
    else:
        fc = harvest(load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR])
        r = fc * float(np.median(hs[hs > 0])) * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
    n = min(len(r), len(x) // 240)
    r = r[:n]
    hd = harm_db(x, r)
    return it, x, r, hd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        for nm, k in VERS.items():
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                if (H / s / "d570" / "REFKC5" / f"{key}_REFKC5.wav").exists():
                    its.append((nm, s, key, int(ti), k))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    print("harmonic fits done", len(res), flush=True)
    import torch
    import f0est as FE
    ROOT = Path(__file__).resolve().parent.parent
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())

    def f1track(x: np.ndarray) -> np.ndarray:
        n = len(x) // 240
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * 240].astype(np.float32))[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        kk = pr[:, :m].argmax(0)
        ii = np.clip(kk[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        return np.where(vo[:m] & (c >= 0.5), np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12)), 0.0)

    rows: dict = {}
    pos_up: dict = {nm: np.zeros((NPOS, 2)) for nm in VERS}
    pos_h12: dict = {nm: [[] for _ in range(NPOS)] for nm in VERS}
    for (nm, s, key, ti, k), x, r, hd in res:
        fv = f1track(x)
        n = min(len(fv), len(r))
        d12 = hd[:n, 0] - hd[:n, 1]
        on_l, st_l = [], []
        for i, j in onsets(r[:n]):
            for p in range(NPOS):
                t = i + p
                if t >= j:
                    break
                if np.isfinite(d12[t]):
                    pos_h12[nm][p].append(d12[t])
                if fv[t] > 0:
                    pos_up[nm][p] += [float(12 * np.log2(fv[t] / r[t]) > 9), 1.0]
            on = d12[i:min(i + 4, j)]
            st = d12[i + 15:min(i + 31, j)]
            on, st = on[np.isfinite(on)], st[np.isfinite(st)]
            if len(on) >= 2 and len(st) >= 5:
                on_l.append(on.mean())
                st_l.append(st.mean())
        if len(on_l) >= 3:
            rows.setdefault(nm, {})[(s, key)] = {"t": ti, "onset": float(np.mean(on_l)), "steady": float(np.mean(st_l)), "delta": float(np.mean(on_l) - np.mean(st_l)), "n_on": len(on_l)}
    rng = np.random.default_rng(605)
    rep: dict = {"sets": {}, "pairs": {}, "pos": {}}
    for nm, R in rows.items():
        rep["sets"][nm] = {m: round(float(np.median([v[m] for v in R.values()])), 3) for m in ("onset", "steady", "delta")} | {"n": len(R), "onsets": int(sum(v["n_on"] for v in R.values()))}
        print(nm, rep["sets"][nm], flush=True)
    for nm in VERS:
        rep["pos"][nm] = {"h1_minus_h2_db": [round(float(np.median(v)), 2) if v else None for v in pos_h12[nm]],
                          "f1_up_rate": [round(float(u / max(c, 1)), 4) for u, c in pos_up[nm]]}
        print(nm, "H1-H2 by pos", rep["pos"][nm]["h1_minus_h2_db"][:10], "up by pos", rep["pos"][nm]["f1_up_rate"][:10], flush=True)
    for A_, B_ in (("OURS5RY58G", "REFKC5"), ("OURS12RY58G", "REFKC12")):
        ks = sorted(set(rows[A_]) & set(rows[B_]))
        tg = np.array([f"{k[0]}|{rows[A_][k]['t']}" for k in ks])
        for m in ("delta", "onset", "steady"):
            d = np.array([rows[A_][k][m] - rows[B_][k][m] for k in ks])
            rep["pairs"][f"{A_}-{B_}:{m}"] = {"n": len(ks), "diff": round(float(d.mean()), 3), "ci95": boot(d, tg, rng)}
            print(f"{A_}-{B_}:{m}", rep["pairs"][f"{A_}-{B_}:{m}"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
