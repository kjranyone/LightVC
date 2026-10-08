"""全話者mel重心計算(定理E処方=話者正規化melの前提資産)。

data/spk_mel_centroids.pt = {spk_id: [80]} — causal_mel48の発話平均。
2775話者×最大4発話(先頭4s)を10並列で処理。

    CUDA_VISIBLE_DEVICES= uv run python spk_mel_centroids.py
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import causal_mel48, HOP

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "female-dataset"
OUT = ROOT / "data/spk_mel_centroids.pt"


def one_spk(spk_dir: str):
    try:
        wavs = sorted((Path(spk_dir)).glob("*.wav"))[:4]
        ms = []
        for w in wavs:
            y, sr = sf.read(str(w), dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if sr != 48000:
                y = librosa.resample(y, orig_sr=sr, target_sr=48000)
            n = min(len(y), 192000)
            if n < HOP:
                continue
            wv = torch.from_numpy(y[: n // HOP * HOP])[None]
            ms.append(causal_mel48(wv)[0].mean(-1))
        if not ms:
            return (Path(spk_dir).name, None)
        return (Path(spk_dir).name, torch.stack(ms).mean(0))
    except Exception:  # noqa: BLE001
        return (Path(spk_dir).name, None)


def main() -> int:
    spks = sorted([str(d) for d in FD.iterdir() if d.is_dir()])
    print(f"  {len(spks)} speakers", flush=True)
    t0 = time.time()
    out = {}
    miss = 0
    with ProcessPoolExecutor(max_workers=10) as ex:
        for i, (name, m) in enumerate(ex.map(one_spk, spks, chunksize=4)):
            if m is None:
                miss += 1
                continue
            out[name] = m
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(spks)} miss {miss} ({time.time()-t0:.0f}s)",
                      flush=True)
    torch.save(out, OUT)
    print(json.dumps({"speakers": len(out), "miss": miss,
                      "elapsed_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
