"""
Phase A: Encode a multi-speaker speech corpus into DAC continuous latents.

Output structure:
  latents/
    {speaker_id}/
      {utterance_id}.npy    # [latent_dim=1024, T_frames]
    index.tsv               # speaker_id \t utterance_id \t n_frames \t path

No teacher. No pairing. No synthetic generation.
Just encode real speech with DAC.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def load_dac(model_id: str = "descript/dac_44khz"):
    """Load DAC model on the best available device."""
    from transformers import AutoModel

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    dac = AutoModel.from_pretrained(model_id).to(device).eval()
    print(f"DAC loaded on device: {device}")
    return dac, device


def find_audio_files(root: str, extensions=(".wav", ".flac", ".mp3")):
    """Recursively find audio files, grouped by speaker.

    Assumes directory structure: root/{speaker_id}/{anything}.{ext}
    Falls back to treating each filename's first underscore-separated token
    as speaker_id if no subdirectory structure exists.
    """
    root_path = Path(root)
    files = []
    for ext in extensions:
        files.extend(root_path.rglob(f"*{ext}"))
        files.extend(root_path.rglob(f"*{ext.upper()}"))
    files = sorted(set(files))

    # Detect structure
    sample = files[0] if files else None
    if sample and sample.parent != root_path:
        # Subdirectory structure: use parent folder name as speaker_id
        def speaker_of(p: Path) -> str:
            return p.parent.relative_to(root_path).parts[0]
    else:
        # Flat: use filename prefix (e.g., "spk_001_utt000.wav" → "spk_001")
        def speaker_of(p: Path) -> str:
            stem = p.stem
            parts = stem.split("_")
            return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]

    return [(f, speaker_of(f)) for f in files], speaker_of


@torch.no_grad()
def encode_audio(dac, wav_44k: np.ndarray, device: str) -> np.ndarray:
    """44.1kHz mono → continuous latent [1024, T_frames]."""
    x = torch.from_numpy(wav_44k).float()
    x = x.unsqueeze(0).unsqueeze(0).to(device)
    latent = dac.encoder(x)
    return latent.squeeze(0).cpu().numpy()


def load_and_resample(path: str, target_sr: int = 44100):
    """Load audio and resample to target_sr. Returns mono numpy array."""
    import soundfile as sf
    import librosa

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32), target_sr


def main():
    parser = argparse.ArgumentParser(
        description="Encode multi-speaker corpus into DAC latents"
    )
    parser.add_argument("--source", required=True, help="Audio corpus root")
    parser.add_argument("--output", required=True, help="Latent output directory")
    parser.add_argument("--dac-model", default="descript/dac_44khz")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Skip utterances shorter than this (seconds)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Truncate utterances longer than this (seconds)",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dac, device = load_dac(args.dac_model)
    files, speaker_of = find_audio_files(args.source)
    print(f"Found {len(files)} audio files")

    # Group by speaker
    speakers = {}
    for f, spk in files:
        speakers.setdefault(spk, []).append(f)
    print(f"Speakers: {len(speakers)}")
    for spk in sorted(speakers.keys())[:10]:
        print(f"  {spk}: {len(speakers[spk])} files")
    if len(speakers) > 10:
        print(f"  ... ({len(speakers)} total)")

    min_samples = int(args.min_duration * 44100)
    max_samples = int(args.max_duration * 44100)

    index_path = os.path.join(args.output, "index.tsv")
    with open(index_path, "w", newline="") as idx_file:
        writer = csv.writer(idx_file, delimiter="\t")
        writer.writerow(["speaker_id", "utterance_id", "n_frames", "path"])

        for spk in tqdm(sorted(speakers.keys()), desc="Speakers"):
            spk_dir = os.path.join(args.output, spk)
            os.makedirs(spk_dir, exist_ok=True)

            for f in tqdm(speakers[spk], desc=f"  {spk}", leave=False):
                try:
                    wav, sr = load_and_resample(str(f), 44100)
                except Exception as e:
                    print(f"    Skip {f.name}: {e}")
                    continue

                if len(wav) < min_samples:
                    continue
                if len(wav) > max_samples:
                    wav = wav[:max_samples]

                # Pad to hop length
                rem = len(wav) % 512
                if rem > 0:
                    wav = np.pad(wav, (0, 512 - rem))

                latent = encode_audio(dac, wav, device)

                utt_id = f.stem
                out_path = os.path.join(spk_dir, f"{utt_id}.npy")
                np.save(out_path, latent.astype(np.float32))

                writer.writerow([spk, utt_id, latent.shape[1], out_path])

    print(f"\nDone. Index: {index_path}")


if __name__ == "__main__":
    main()
