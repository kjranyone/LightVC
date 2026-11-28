from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf
import torch
from scipy.signal import resample_poly

os.environ["SNV_WIN_CAP_MS"] = "7.5"
os.environ["SNV_W_YIN"] = "600"
os.environ["SNV_TAU_UV"] = "500"

import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent
WAV = ROOT / "data/snv_p97_wav"
OUT = ROOT / "data/snv_p122_feat"
HI_PREFIX = ("scream", "cry", "mix")


def _init_worker() -> None:
    torch.set_num_threads(1)
    P.WIN_CAP_MS = 7.5
    P.W_YIN = 600
    P.TAU_UV_HZ = 500.0


def extract_f0(x: np.ndarray, high_mode: bool) -> np.ndarray:
    n = P.n_frames(len(x))
    xd = x[: n * P.H].astype(np.float64)
    x16 = resample_poly(xd, 1, 3)
    f1, t1 = pyworld.harvest(x16, 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    f1 = pyworld.stonemask(x16, f1, t1, 16000)[:n]
    if not high_mode:
        return np.pad(f1, (0, max(0, n - len(f1)))).astype(np.float32)
    fh, th = pyworld.harvest(xd, P.SR, f0_floor=60.0, f0_ceil=1600.0, frame_period=5.0)
    fh = pyworld.stonemask(xd, fh, th, P.SR)[:n]
    fh = np.pad(fh, (0, max(0, n - len(fh))))
    f1 = np.pad(f1, (0, max(0, n - len(f1))))
    if not (fh > 950.0).any():
        return f1.astype(np.float32)
    py, vf, _ = librosa.pyin(x16, fmin=60.0, fmax=1600.0, sr=16000, frame_length=1024, hop_length=80, center=True)
    py = np.where(vf, np.nan_to_num(py), 0.0)[:n]
    py = np.pad(py, (0, max(0, n - len(py))))
    agree = (py > 0) & (np.abs(np.log2(np.maximum(fh, 1.0) / np.maximum(py, 1.0))) < 0.2)
    out = np.where((fh > 950.0) & agree, fh, f1)
    return out.astype(np.float32)


def process_voice(voice: str) -> tuple[str, int]:
    cnt = 0
    for kind in ("e1", "e1nv"):
        src_dir = WAV / kind / voice
        if not src_dir.exists():
            continue
        dst_dir = OUT / kind / voice
        dst_dir.mkdir(parents=True, exist_ok=True)
        for w in sorted(src_dir.glob("*.flac")):
            dst = dst_dir / f"{w.stem}.npz"
            if dst.exists():
                cnt += 1
                continue
            x, sr = sf.read(str(w), dtype="float32")
            assert sr == P.SR
            n = P.n_frames(len(x))
            xc = x[: n * P.H]
            hi = kind == "e1nv" and w.stem.startswith(HI_PREFIX)
            f0 = extract_f0(xc, hi)
            env = P.envelope(xc, f0.astype(np.float64))
            per = P.periodicity(torch.from_numpy(xc), torch.from_numpy(f0)).numpy()
            tmp = dst.with_suffix(".tmp.npz")
            np.savez(tmp, f0=f0, env=env.astype(np.float32), per=per.astype(np.float32))
            tmp.replace(dst)
            cnt += 1
    return voice, cnt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    voices = sorted(p.name for p in (WAV / "e1").iterdir() if p.is_dir())
    t0 = time.time()
    total = 0
    workers = min(18, os.cpu_count() or 8)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        for i, (v, c) in enumerate(ex.map(process_voice, voices), 1):
            total += c
            if i % 50 == 0 or i == len(voices):
                print(f"{i}/{len(voices)} voices ({total} files) in {time.time() - t0:.1f}s", flush=True)
    meta = {
        "source_wav": "data/snv_p97_wav",
        "f0": "harvest+stonemask (16k ceil 1000; e1nv scream/cry/mix 48k ceil 1600 pyin-verified >950Hz)",
        "front_const": {"WIN_CAP_MS": 7.5, "W_YIN": 600, "TAU_UV_HZ": 500.0},
        "voices": len(voices),
        "files": total,
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("done", json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
