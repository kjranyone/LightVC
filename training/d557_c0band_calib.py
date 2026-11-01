"""D557(腕 56 の準備・学習なし): 音量の条件を C1 と同じ因果な log-mel(c1_content.MelFront・128 帯・(log|mel| − log 1e-5)/10)の 100 Hz〜4 kHz の帯の平均 L にする(c0band)。
旧来の c0n(線形周波数の対数包絡の平均 = 高域が 2/3・D400 / D431)は入力の高域の水準を運び、c0low(腕 47・DCT の行の近似)は声道の形の動きが混ざって音量の揺れを増やした(D555: 音量の揺れ 6.14 → 6.80)。
較正 = 実女声の入力(devff2 / devff3 = devff23x の 126 組・前置き込み 20 s)の Rust の書き出しの c0n(列 0)と同じフレームの L を、発声中(c0n > 0.5)で平均と標準偏差をそろえる一次変換 c0b = max(0, a·L + b)。
出力 = results/diag_ppgvoc56/c0band.json(帯の添字 lo / hi・a・b・相関)。
    uv run python d557_c0band_calib.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import nvoc as N
import train_ppgvoc as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    cf = librosa.mel_frequencies(n_mels=N.N_MEL + 2, fmin=0.0, fmax=N.FMAX)[1:-1]
    idx = np.flatnonzero((cf >= 100) & (cf <= 4000))
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    fr = CC.MelFront().eval()
    S = P.H / "devff23x"
    ed = P.H / "e2e" / "eval_devff23x_pre"
    C, L = [], []
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, _ = P.load_dump(ed, k)
        x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
        n = len(x) // N.HOP
        xx = torch.from_numpy(np.concatenate([np.zeros(N.WIN - N.HOP, np.float32), x[: n * N.HOP]]))[None]
        with torch.no_grad():
            m = fr(xx)[0].numpy()
        T = min(len(c), m.shape[1])
        C.append(c[:T, 0]); L.append(m[lo:hi, :T].mean(0))
    C, L = np.concatenate(C), np.concatenate(L)
    act = C > 0.5
    a = float(C[act].std() / L[act].std())
    b = float(C[act].mean() - a * L[act].mean())
    r_ = float(np.corrcoef(C[act], L[act])[0, 1])
    cb = np.maximum(0.0, a * L + b)
    out = {"lo": lo, "hi": hi, "f_lo": float(cf[lo]), "f_hi": float(cf[hi - 1]), "a": a, "b": b, "corr_active": round(r_, 4),
           "c0n_active_mean_std": [float(C[act].mean()), float(C[act].std())], "c0b_active_mean_std": [float(cb[act].mean()), float(cb[act].std())],
           "silence_c0n_c0b": [float(np.median(C[~act])), float(np.median(cb[~act]))], "frames": int(len(C)), "set": "devff23x(実女声の入力・Rust の書き出しの c0n)"}
    print(json.dumps(out, ensure_ascii=False), flush=True)
    od = ROOT / "results/diag_ppgvoc56"
    od.mkdir(parents=True, exist_ok=True)
    (od / "c0band.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
