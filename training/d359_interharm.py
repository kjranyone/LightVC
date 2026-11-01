"""D359: 倍音の山と倍音の間の谷の比(dB)を帯域ごとに測る(倍音間の雑音・jirijiri-interharmonic-aa)。D356 の推定「出力の高さが低いほど生成器の傷が目立つ」の直接の検査。
保留の SR 30 本: 高さ = 書き出しの log f0(+ 群ごとの移動 s 半音)・有声で前後も有声のフレーム・2048 点の Hann 窓・倍音 h の山 = h f0 ± f0/8 の最大・谷 = (h + 0.5) f0 ± f0/8 の最小。
比が小さい = 倍音の間に雑音がある。同じ高さの群どうし(本物・写し合成・SR の元の高さ)で比べるのが主(窓の漏れは高さに依存するので、高さを動かした群は本物と直接比べない)。診断。
    uv run python d359_interharm.py --groups REAL=<dir>:REAL:0 CP=<dir>:CP:0 S0=<dir>:SR:0 Sm4=<dir>:SR:-4 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

BANDS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 10000)]


def ratio(y: np.ndarray, f0: np.ndarray) -> np.ndarray:
    nf, win = 2048, np.hanning(2048)
    fr = np.fft.rfftfreq(nf, 1 / N.SR)
    acc = [[] for _ in BANDS]
    for k in range(2, len(f0) - 2):
        if not (f0[k - 2:k + 3] > 0).all():
            continue
        c = k * N.HOP + N.HOP // 2
        if c - nf // 2 < 0 or c + nf // 2 > len(y):
            continue
        X = np.abs(np.fft.rfft(y[c - nf // 2:c + nf // 2] * win)) + 1e-9
        f = f0[k]
        h = 1
        while (h + 0.5) * f < BANDS[-1][1]:
            pk = X[(fr > h * f - f / 8) & (fr < h * f + f / 8)]
            vl = X[(fr > (h + 0.5) * f - f / 8) & (fr < (h + 0.5) * f + f / 8)]
            if len(pk) and len(vl):
                r = 20 * np.log10(pk.max() / vl.min())
                for j, (lo, hi) in enumerate(BANDS):
                    if lo <= h * f < hi:
                        acc[j].append(r)
            h += 1
    return np.array([np.mean(a) if a else np.nan for a in acc])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True, help="NAME=dir:suffix:shift_semitones")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    held = P.e2e_held()
    rep = {}
    for g in a.groups:
        name, rest = g.split("=", 1)
        d, suf, sh = rest.split(":")
        R = []
        for i, (_, ka, _) in enumerate(held):
            f = Path(d) / f"h{i:04d}_t{i:03d}_{suf}.wav"
            if not f.exists():
                continue
            y, sr = sf.read(f, dtype="float32")
            assert sr == N.SR
            c, _ = P.load_dump(P.H / "e2e" / "fem", ka)
            f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25] + float(sh) * np.log(2) / 12), 0.0)
            R.append(ratio(y, f0))
        R = np.array(R)
        rep[name] = {"n": len(R), "mean_db": np.nanmean(R, 0).round(2).tolist(), "se_db": (np.nanstd(R, 0) / np.sqrt(len(R))).round(2).tolist()}
        print(f"{name:8s} n {len(R)}", " ".join(f"{lo}-{hi}Hz {m:5.1f}±{s:.1f}" for (lo, hi), m, s in zip(BANDS, rep[name]["mean_db"], rep[name]["se_db"])), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
