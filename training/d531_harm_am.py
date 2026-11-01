"""D531(耳の所見「フェーザー・コーラス・ピッチの不自然な振幅」の物差し・D520 は窓 85 ms で 10〜40 Hz のうなりを均していた):
出力の WORLD f0(16 kHz・2 ms・stonemask)を標本へ補間した位相 φ で倍音 k = 1..8(k·f0 < 4 kHz)を復調 z_k = LPF(x·e^{−jkφ})(遮断 0.4·区間の f0 の中央値)し、
(1) うなり AM = 20log|z_k| から 50 ms の移動平均を引いた残りの RMS(dB・その瞬間の最大の倍音から 25 dB 以内の標本だけ)、
(2) 倍音の周波数のばらつき INC = 各標本で (z_k の瞬時周波数のずれ)/k の倍音にわたる重みつき標準偏差(Hz・重み = 振幅)= 倍音が一つの周期に揃っているか(コーラスは揃わない)、
を有声の区間(100 ms 以上・端 20 ms を除く)で求め、発話ごとの中央値を出す。群 = <名前>=<dir>:<接尾辞>(<dir>/<pNNNN_tNNN>_<接尾辞>.wav)・REAL_cen(目標の別録音 8 s 区切り)。
    uv run python d531_harm_am.py --set devff5 --groups SRC=<d56>:SRC CP=<dir>:CP ... --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

FS = 16000
KMAX = 8


def measure(x: np.ndarray, sr: int) -> dict:
    import pyworld as pw
    x = resample_poly(x.astype(np.float64), FS, sr)
    f0, t = pw.dio(x, FS, f0_floor=90, f0_ceil=1100, frame_period=2.0)
    f0 = pw.stonemask(x, f0, t, FS)
    ts = np.arange(len(x)) / FS
    v = f0 > 0
    am, inc = [], []
    s = None
    for i, b in enumerate(np.r_[v, False]):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if (i - s) * 2 >= 100:
                a0, a1 = int(t[s] * FS), int(t[i - 1] * FS)
                fs_ = np.interp(ts[a0:a1], t[s:i], f0[s:i])
                fm = float(np.median(fs_))
                ph = 2 * np.pi * np.cumsum(fs_) / FS
                seg = x[a0:a1]
                sos = butter(4, 0.4 * fm, fs=FS, output="sos")
                K = min(KMAX, int(4000 / fm))
                Z = []
                for k in range(1, K + 1):
                    d = seg * np.exp(-1j * k * ph)
                    Z.append(sosfiltfilt(sos, d.real) + 1j * sosfiltfilt(sos, d.imag))
                Z = np.array(Z)
                e = int(0.02 * FS)
                Z = Z[:, e:-e]
                if Z.shape[1] > int(0.05 * FS) and K >= 3:
                    A = 20 * np.log10(np.abs(Z) + 1e-9)
                    w = int(0.05 * FS)
                    ker = np.ones(w) / w
                    As = np.array([np.convolve(np.pad(a, (w // 2, w - 1 - w // 2), mode="edge"), ker, mode="valid") for a in A])
                    ok = A > A.max(0, keepdims=True) - 25
                    am.append(float(np.sqrt(np.mean(((A - As)[ok]) ** 2))))
                    fi = np.diff(np.unwrap(np.angle(Z), axis=1), axis=1) * FS / (2 * np.pi) / np.arange(1, K + 1)[:, None]
                    wt = np.abs(Z[:, 1:]) * ok[:, 1:]
                    mu = (fi * wt).sum(0) / (wt.sum(0) + 1e-12)
                    sd = np.sqrt((wt * (fi - mu) ** 2).sum(0) / (wt.sum(0) + 1e-12))
                    inc.append(float(np.median(sd[wt.sum(0) > 0])))
            s = None
    return {"am_db": float(np.median(am)) if am else np.nan, "inc_hz": float(np.median(inc)) if inc else np.nan}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--cen", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    if a.n:
        pairs = pairs[: a.n]
    rows: dict = {g: {} for g in G}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        for g, (d, sfx) in G.items():
            p = d / f"{k}_{sfx}.wav"
            if p.exists():
                x, sr = sf.read(str(p), dtype="float32", always_2d=True)
                rows[g][k] = measure(x.mean(1)[: 8 * sr], sr)
    if a.cen:
        rows["REAL_cen"] = {}
        for t_ in sorted({r["ti"] for r in pairs}):
            p = S / "d37" / f"cen_{t_:03d}.wav"
            if not p.exists():
                continue
            x, sr = sf.read(str(p), dtype="float32", always_2d=True)
            x = x.mean(1)
            for o in range(0, max(1, len(x) - 8 * sr), 8 * sr):
                rows["REAL_cen"][f"{t_}_{o}"] = measure(x[o:o + 8 * sr], sr)
    rep = {}
    for g, d in rows.items():
        rep[g] = {m: [round(float(np.nanmedian([v[m] for v in d.values()])), 3), round(float(np.nanpercentile([v[m] for v in d.values()], 15)), 3),
                      round(float(np.nanpercentile([v[m] for v in d.values()], 85)), 3)] for m in ("am_db", "inc_hz")}
        rep[g]["n"] = len(d)
    ref = list(G)[0]
    for g in G:
        if g == ref:
            continue
        ks = sorted(set(rows[g]) & set(rows[ref]))
        rep[f"{g}-{ref}"] = {m: [round(float(np.nanmedian([rows[g][k][m] - rows[ref][k][m] for k in ks])), 3), round(float(np.nanmean([rows[g][k][m] > rows[ref][k][m] for k in ks])), 3)] for m in ("am_db", "inc_hz")}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps({"summary_median_p15_p85": rep, "per_utt": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
