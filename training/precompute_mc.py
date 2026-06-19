"""
Pre-compute WORLD mel-cepstrum features for all VCTK utterances.

Cache: data/mc_cache/{spk}/{utt_stem}.npz
  - mc: mel-cepstrum [T, 25]
  - f0: fundamental frequency [T]
  - vuv: voiced/unvoiced [T]
  - codeap: coded aperiodicity [T, 1]
  - energy: frame energy [T]
"""
import sys, os, time
from pathlib import Path

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import librosa

SR = 16000
FRAME_PERIOD = 5.0
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24

VCTK_WAV = Path("../data/vctk_200")
CACHE_DIR = Path("data/mc_cache")


def analyze_wav(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)

    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)

    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA).astype(np.float32)
    codeap = world.code_aperiodicity(ap, SR).astype(np.float32)
    vuv = (f0 > 0).astype(np.float32)
    energy = 10 * np.log10(np.sqrt(np.sum(sp ** 2, axis=1)) + 1e-10).astype(np.float32)

    return {
        "mc": mc,
        "f0": f0.astype(np.float32),
        "vuv": vuv,
        "codeap": codeap,
        "energy": energy,
    }


def main():
    print(f"Pre-computing mel-cepstrum features at {SR}Hz, {FRAME_PERIOD}ms hop")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = []
    for spk_dir in sorted(VCTK_WAV.iterdir()):
        if not spk_dir.is_dir():
            continue
        for wav_path in spk_dir.glob("*.wav"):
            cache_path = CACHE_DIR / spk_dir.name / (wav_path.stem + ".npz")
            if cache_path.exists():
                continue
            wav_files.append(wav_path)

    print(f"Total to process: {len(wav_files)}")
    t0 = time.time()
    for i, wav_path in enumerate(wav_files):
        try:
            feat = analyze_wav(wav_path)
            cache_path = CACHE_DIR / wav_path.parent.name / (wav_path.stem + ".npz")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, **feat)
        except Exception as e:
            print(f"  ERROR {wav_path.name}: {e}")
            continue

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(wav_files) - i - 1) / rate
            print(f"  {i+1}/{len(wav_files)} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"Done: {len(wav_files)} files in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
