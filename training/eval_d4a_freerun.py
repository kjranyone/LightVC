"""D4a free-run評価: 自分の生成波形を食いながらブロックAR生成し真の品質を測る。

teacher-forced evalは周期予測で見かけ上良くなるため、ゲートはこちらを使う。
ブロック長=1フレーム(480サンプル)=厳密ARの分割並列。

    CUDA_VISIBLE_DEVICES=0 uv run python eval_d4a_freerun.py [--ckpt ..._best.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import D4Voc, HOP
from train_d4a import get_utt_cond, held_paths, ROOT

OUT = ROOT / "results/d4a_namvoc"


def freerun(m, cond: torch.Tensor, n_samples: int, dev,
            block: int = HOP) -> torch.Tensor:
    """cond [1,274,T] から波形をブロックAR生成(厳密AR)。"""
    gen = torch.zeros(1, n_samples, device=dev)
    rf = 3060 + 480
    pos = 0
    hist = torch.zeros(1, rf, device=dev)
    while pos < n_samples:
        nb = min(block, n_samples - pos)
        ctx = torch.cat([hist, torch.zeros(1, nb, device=dev)], -1)
        with torch.no_grad():
            y = m(ctx, cond[:, :, : (pos + nb) // HOP + 1])
        gen[:, pos:pos + nb] = y[:, -nb:]
        new = gen[:, max(0, pos + nb - rf):pos + nb]
        if new.shape[-1] < rf:
            hist = torch.cat([torch.zeros(1, rf - new.shape[-1], device=dev), new], -1)
        else:
            hist = new
        pos += nb
    return gen


def metrics(y: np.ndarray) -> dict:
    w44 = librosa.resample(y.astype(np.float64), orig_sr=48000, target_sr=44100)
    f0, t_ = pyworld.harvest(w44, 44100, f0_floor=65, f0_ceil=1000,
                             frame_period=512 / 44100 * 1000)
    ap = pyworld.d4c(w44, f0, t_, 44100, fft_size=2048)
    n = min(len(f0), ap.shape[1]); v = f0[:n] > 60
    S = np.abs(librosa.stft(w44, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=44100, n_fft=2048)
    tot = S.sum() + 1e-12
    return {"aperiod": round(float(ap[:, :n][:, v].mean(0).mean()), 3) if v.sum() > 10 else 1.0,
            "voiced": round(float((f0 > 60).mean()), 3),
            "hi_mid": round(float(S[(f > 2000) & (f < 6000)].sum() / tot), 4),
            "f0_median": round(float(np.median(f0[f0 > 60])) if (f0 > 60).any() else 0.0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "d4a_namvoc_best.pt"))
    ap.add_argument("--out", default="freerun.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = D4Voc().to(dev).eval()
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m.load_state_dict(ck["net"])
    print(f"  ckpt step {ck.get('step')}", flush=True)

    res = {}
    for i, p in enumerate(held_paths()[:4]):
        wv, cond = get_utt_cond(p)
        n = min(wv.shape[0], 192000)
        ct = cond[None, :, : n // HOP].to(dev)
        y = freerun(m, ct, n, dev)
        y = y[0].cpu().numpy()
        soundfile.write(OUT / f"freerun_held{i}.wav", np.clip(y, -1, 1), 48000)
        res[f"held{i}"] = metrics(y)
        res[f"held{i}_src_f0"] = metrics(wv[:n].numpy())["f0_median"]
        print(f"  held{i}: {res[f'held{i}']} (src f0 {res[f'held{i}_src_f0']})",
              flush=True)
    (OUT / a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
