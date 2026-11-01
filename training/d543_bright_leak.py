"""D543(新しい欠陥の物差し: 操作者の明るさの漏れ): 発話ごとに hf = 4〜10 kHz / 0.1〜4 kHz のエネルギー比(dB・発声中 = 最大から −40 dB)と tilt(1/3 oct の長時間スペクトル 300 Hz〜8 kHz の傾き dB/oct)を
目標の参照(<set>/d37/_work/ref_TTT.wav 30 s)との差 dhf・dtilt で出し、各群の dhf を写し合成 CP の dhf(= 操作者 − 目標の明るさの差)に回帰した傾き(漏れの割合・操作者でまとめないブートストラップ 95%)。
理想の変換 = 0(出力は目標の明るさ)・写し合成 = 1。d109 と同じ定義(RMS をそろえ・F1 なし)。
    uv run python d543_bright_leak.py --set devff5 --groups CP=<dir>:CP CAND=<dir>:OP15 OFF=<dir>:OFF --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d6_unit_cov import load48

CF = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
CF = CF[(CF >= 100) & (CF <= 10000)]
SEL = (CF >= 300) & (CF <= 8000)


def spec(x: np.ndarray) -> tuple:
    x = x.astype(np.float64)
    x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
    n = len(x) // N.HOP
    e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
    act = e > e.max() - 40
    fq, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
    Pw = np.abs(Z[:, : len(act)]) ** 2
    Pa = Pw[:, act[: Pw.shape[1]]]
    hf = float(10 * np.log10(Pa[(fq >= 4000) & (fq < 10000)].sum() / Pa[(fq >= 100) & (fq < 4000)].sum()))
    Pm = Pa.mean(1)
    L = np.array([Pm[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in CF])
    L = 10 * np.log10(L + 1e-20)
    tilt = float(np.polyfit(np.log2(CF[SEL]), L[SEL], 1)[0])
    return hf, tilt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--ref", default="CP")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = P.H / a.set
    G = {g.split("=")[0]: (Path(g.split("=")[1].split(":")[0]), g.split(":")[-1]) for g in a.groups}
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    tref = {}
    rows: dict = {g: {} for g in G}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        t = r["ti"]
        if t not in tref:
            tref[t] = spec(load48(S / "d37/_work" / f"ref_{t:03d}.wav")[: 30 * N.SR])
        for g, (d, sfx) in G.items():
            p = d / f"{k}_{sfx}.wav"
            if p.exists():
                hf, tl = spec(load48(p)[: 8 * N.SR])
                rows[g][k] = (hf - tref[t][0], tl - tref[t][1])
    rep = {}
    ks0 = set(rows[a.ref])
    for g in G:
        ks = sorted(ks0 & set(rows[g]))
        for j, m in ((0, "dhf"), (1, "dtilt")):
            x = np.array([rows[a.ref][k][j] for k in ks])
            y = np.array([rows[g][k][j] for k in ks])
            A = np.c_[x, np.ones_like(x)]
            b = float(np.linalg.lstsq(A, y, rcond=None)[0][0])
            rng = np.random.default_rng(543)
            bs = [np.linalg.lstsq(A[i], y[i], rcond=None)[0][0] for i in (rng.integers(0, len(x), len(x)) for _ in range(1000))]
            rep.setdefault(g, {})[m] = {"slope": round(b, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                                         "median": round(float(np.median(y)), 2), "abs_median": round(float(np.median(np.abs(y))), 2), "n": len(ks)}
        print(g, rep[g], flush=True)
    Path(a.out).write_text(json.dumps({"summary": rep, "per_utt": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
