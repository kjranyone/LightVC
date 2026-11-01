"""D141(学習なし): 出力の長時間スペクトルの等化(登録の定数・因果な最小位相 FIR・先読み 0)。D140: 同一性が kNN-VC 並みの G03S10B(γ0.3・β1・較正)は
高域比・傾きだけで落ちる(出力が目標より暗い)。条件の段の較正(D114)では出力の暗さが残る → 出力の波形に固定の等化を掛ける。
等化 = 目標の参照の 1/3 oct の長時間スペクトル(発声中・水準をそろえる)− 同じ構成で操作者の参照を変換した出力(d114 の _cal_<cfg>/op*.wav)の同じ量 → 平滑・±12dB に制限 →
firwin2(511 タップ)→ minimum_phase。評価の変換 <cfg>B に掛けて <cfg>Q を書く(D109 の関門・CER・LDA で測る)。群遅延(1kHz 以下の中央値)を印字。
    uv run python d141_out_eq.py --s <scratchpad> --cfgs G03S10 G00S10 G07S03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--strength", type=float, default=1.0, help="等化の強さ(dB を倍・1 = 目標の長時間スペクトルへ全部)")
    ap.add_argument("--sfx", default="Q")
    a = ap.parse_args()
    import scipy.signal as ss
    from d6_unit_cov import load48
    S = Path(a.s)
    sr = 48000
    cf = 1000.0 * 2.0 ** (np.arange(-13, 15) / 3.0)
    cf = cf[(cf >= 60) & (cf <= 20000)]

    def ltas(x: np.ndarray) -> np.ndarray:
        n = len(x) // 240
        e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        fq, _, Z = ss.stft(x, sr, nperseg=2048, noverlap=2048 - 240)
        P = np.abs(Z[:, :n]) ** 2
        P = P[:, act[: P.shape[1]]].mean(1)
        L = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() + 1e-20 for c in cf]))
        return L - L.mean()

    tl = {}
    gd = []
    for cfg in a.cfgs:
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            src = S / "d114" / f"{k}_{cfg}B.wav"
            o = S / "d114" / f"{k}_{cfg}{a.sfx}.wav"
            if o.exists() or not src.exists():
                continue
            if ti not in tl:
                tl[ti] = ltas(load48(S / "d37/_work" / f"ref_{ti:03d}.wav").astype(np.float64))
            cal = S / "d114" / f"_cal_{cfg}" / f"op{pi:04d}_t{ti:03d}.wav"
            g = tl[ti] - ltas(load48(cal).astype(np.float64))
            g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
            g = np.clip(a.strength * (g - np.median(g)), -12, 12)
            fr = np.r_[0.0, cf / (sr / 2), 1.0]
            gl = 10 ** (np.r_[g[0], g, g[-1]] / 20)
            h = ss.minimum_phase(ss.firwin2(1023, np.clip(fr, 0, 1), gl ** 2, fs=2.0), method="homomorphic")
            w, gdel = ss.group_delay((h, 1), w=np.linspace(50, 1000, 50), fs=sr)
            gd.append(float(np.median(gdel)) / sr * 1000)
            y = load48(src).astype(np.float64)
            z = ss.lfilter(h, 1.0, y)
            z = z / (np.abs(z).max() + 1e-9) * 0.9
            sf.write(o, z.astype(np.float32), sr)
        print(cfg, "done", flush=True)
    if gd:
        print("group delay ms (50–1000Hz median): median", round(float(np.median(gd)), 2), "p95", round(float(np.percentile(gd, 95)), 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
