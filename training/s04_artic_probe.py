"""Artic-A2 S0-4(中心仮説の耳検証): 構音界面(LAR)の誤差はコーラスにならないか。事前固定: current/artic_a2.md §8。

同じ発話(chorus_probe と同じ 0005e65d3f11f99d_00002260・6s)で、励起は実残差のまま、LAR に帯域別・大きさ別の
ガウス誤差(次元ごとに自然変動の std に比例)を足して再合成する。錨=元音声、陽性対照=chorus_probe で「有」だった
codec 潜在+合成雑音(synth_all)。試行内で元音声に RMS 整合→共通減衰し Y1..Y8 に盲検化。鍵は _key_聴取後に開く.json。
判定(事前固定): 0–15Hz の誤差(4本)がコーラス「無」なら中心仮説は生存、「有」なら反証=打ち切り。

    CUDA_VISIBLE_DEVICES= uv run python s04_artic_probe.py --order 24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
from train_dec2 import load48
from render_d1_ab import norm_trial

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/earbattery/artic_s04"
UTT = "0005e65d3f11f99d_00002260"


def band_noise(K: int, p: int, lo: float, hi: float, rng: np.random.Generator) -> np.ndarray:
    fs = D.SR / D.H
    n = rng.standard_normal((K, p))
    F = np.fft.rfft(n, axis=0)
    f = np.fft.rfftfreq(K, 1 / fs)
    F[~((f >= lo) & (f < hi))] = 0
    n = np.fft.irfft(F, n=K, axis=0)
    return n / np.maximum(n.std(0, keepdims=True), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", type=int, default=24)
    a = ap.parse_args()
    import torch
    from train_d1 import build_index
    pairs, _, _ = build_index(0)
    f = next(p for p in pairs if p.stem == UTT)
    d = torch.load(f, map_location="cpu", weights_only=False)
    x = load48(d["path"]).astype(np.float64)[: 6 * D.SR]
    lar, a_sub, e = D.analyze(x, a.order)
    K, p = lar.shape
    sd = lar.std(0, keepdims=True)
    rng = np.random.default_rng(20260926)
    clips = {"source": x}
    meta = {}
    for lo, hi in ((0, 4), (4, 15), (15, 50)):
        for lv in (0.25, 0.5):
            nm = f"lar_{lo}-{hi}Hz_x{lv}"
            lar_n = lar + lv * sd * band_noise(K, p, lo, hi, rng)
            y = D.synthesize(e, D.coef_schedule(lar_n, len(x)))
            clips[nm] = y
            meta[nm] = {"band_hz": [lo, hi], "level_of_natural_std": lv,
                        "lar_err_rms": round(float((lar_n - lar).std()), 4)}
    kp = json.loads((ROOT / "results/earbattery/chorus_probe/_key_聴取後に開く.json").read_text())
    xs = next(k for k, v in kp["map"].items() if v == "synth_all")
    pc = load48(str(ROOT / f"results/earbattery/chorus_probe/{xs}.wav")).astype(np.float64)
    n = min(len(pc), len(x))
    clips = {k: v[:n] for k, v in clips.items()}
    clips["positive_control_codec_synth_all"] = pc[:n]
    normed = norm_trial(clips, clips["source"])
    names = list(normed)
    random.Random("artic_s04_20260926").shuffle(names)
    OUT.mkdir(parents=True, exist_ok=True)
    key = {"utt": UTT, "order": a.order, "map": {}, "meta": meta}
    for i, nm in enumerate(names):
        soundfile.write(OUT / f"X{i + 1}.wav", normed[nm], D.SR)
        key["map"][f"X{i + 1}"] = nm
    (OUT / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False), "->", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
