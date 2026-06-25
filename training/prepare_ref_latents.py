"""
Pre-compute reference latent pool: per-speaker DAC latents from random utterances.

For each VCTK speaker, picks N_REF random utterances (excluding eval text_ids),
DAC-encodes each, crops/pads to REF_FRAMES, saves as a single tensor.

Output: data/ref_latents/{speaker}.pt  →  [N_REF, 1024, REF_FRAMES]

Usage:
  cd training
  uv run python prepare_ref_latents.py
"""
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR, load_dac

VCTK_WAV = Path("../data/vctk_200")
OUT_DIR = Path("../data/ref_latents")
N_REF = 5
REF_FRAMES = 128  # ~1.5s at 86fps
HOP = 512
REF_SAMPLES = REF_FRAMES * HOP  # 65536 samples ≈ 1.49s


def load_wav_44k(p):
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def main():
    print("=== Reference Latent Pool Preparation ===\n")
    dac = load_dac()
    dac.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir():
            continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[d.name].append(w)

    print(f"Speakers: {len(groups)}")
    print(f"Refs per speaker: {N_REF}")
    print(f"Ref frames: {REF_FRAMES} ({REF_SAMPLES} samples ≈ {REF_SAMPLES/DAC_SR:.1f}s)\n")

    rng = random.Random(42)
    total = 0

    for spk in sorted(groups.keys()):
        wavs = groups[spk]
        if len(wavs) < N_REF:
            print(f"  SKIP {spk}: only {len(wavs)} utterances")
            continue

        refs = rng.sample(wavs, min(N_REF, len(wavs)))
        latents = []

        for wpath in refs:
            wav = load_wav_44k(wpath)
            if len(wav) < REF_SAMPLES:
                wav = np.pad(wav, (0, REF_SAMPLES - len(wav)))
            else:
                start = (len(wav) - REF_SAMPLES) // 2
                wav = wav[start:start + REF_SAMPLES]

            x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                z = dac.encoder(x).squeeze(0).cpu()  # [1024, T]

            T = z.shape[1]
            if T > REF_FRAMES:
                z = z[:, :REF_FRAMES]
            elif T < REF_FRAMES:
                z = torch.nn.functional.pad(z, (0, REF_FRAMES - T))
            latents.append(z)

        latents = torch.stack(latents)  # [N_REF, 1024, REF_FRAMES]
        torch.save(latents, OUT_DIR / f"{spk}.pt")
        total += 1

        if total % 20 == 0:
            print(f"  [{total}/{len(groups)}] saved", flush=True)

    print(f"\nDone: {total} speakers → {OUT_DIR}/")


if __name__ == "__main__":
    main()
