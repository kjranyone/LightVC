"""
Encode the original female-dataset (real voice acting recordings) into DAC latents.

This creates the "real female manifold" layer — natural female speech with
emotional variation, laughter, whispers, and live speech patterns that
TTS-generated corpora cannot cover.

Output: data/female_real_latents/{speaker}/{clip_name}.pt
Each file: {"z": latent [1024, T], "text": transcript, "clip": wav_path}

Usage:
  cd training
  uv run python encode_female_dataset.py --max-utts 20000
"""
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR, load_dac

FEMALE_DIR = Path("../female-dataset")
OUT_DIR = Path("../data/female_real_latents")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--female-dir", default=str(FEMALE_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--max-utts", type=int, default=20000,
                        help="max utterances to encode (-1 for all)")
    parser.add_argument("--max-per-speaker", type=int, default=10,
                        help="max clips per speaker")
    args = parser.parse_args()

    print("=== Female Dataset DAC Encoding ===\n")
    dac = load_dac()

    female_dir = Path(args.female_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_dirs = sorted([d for d in female_dir.iterdir() if d.is_dir()])
    print(f"Speakers: {len(speaker_dirs)}")

    total = 0
    skipped = 0

    for si, sd in enumerate(speaker_dirs):
        wavs = sorted(sd.glob("*.wav"))
        labs = {f.stem: f.read_text().strip() for f in sd.glob("*.lab")}

        if len(wavs) > args.max_per_speaker:
            rng = np.random.default_rng(42 + si)
            indices = rng.choice(len(wavs), size=args.max_per_speaker, replace=False)
            wavs = [wavs[i] for i in indices]

        spk_out = out_dir / sd.name
        spk_out.mkdir(exist_ok=True)

        for wav_path in wavs:
            if args.max_utts > 0 and total >= args.max_utts:
                break

            out_path = spk_out / (wav_path.stem + ".pt")
            if out_path.exists():
                skipped += 1
                total += 1
                continue

            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio[:, 0]
                if sr != DAC_SR:
                    audio = librosa.resample(
                        audio.astype(np.float64), orig_sr=sr, target_sr=DAC_SR
                    ).astype(np.float32)
                if len(audio) < DAC_SR:
                    continue

                x = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    z = dac.encoder(x).squeeze(0).cpu().half()

                text = labs.get(wav_path.stem, "")
                torch.save({"z": z, "text": text, "clip": str(wav_path)}, out_path)
                total += 1

            except Exception as e:
                print(f"  SKIP {wav_path.name}: {e}")
                skipped += 1

        if (si + 1) % 100 == 0:
            print(f"  [{si+1}/{len(speaker_dirs)}] speakers, {total} utterances", flush=True)

        if args.max_utts > 0 and total >= args.max_utts:
            print(f"  Reached max_utts={args.max_utts}")
            break

    print(f"\nDone: {total} utterances ({skipped} skipped) -> {out_dir}")


if __name__ == "__main__":
    main()
