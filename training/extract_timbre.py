"""Extract ECAPA timbre embedding from a reference WAV file."""
import sys
from pathlib import Path

import torch
import numpy as np
from safetensors.torch import save_file
import librosa


def main():
    if len(sys.argv) != 3:
        print("Usage: uv run python extract_timbre.py <ref.wav> <timbre.safetensors>")
        sys.exit(1)

    ref_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    wav, sr = librosa.load(str(ref_path), sr=16000)
    if len(wav) < 8000:
        print(f"Reference too short: {len(wav)} samples ({len(wav)/16000:.1f}s)")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": device},
    )

    with torch.no_grad():
        emb = ecapa.encode_batch(torch.from_numpy(wav).unsqueeze(0).to(device))

    emb = emb.squeeze(0).cpu()
    print(f"Timbre embedding: {emb.shape}  from {ref_path.name} ({len(wav)/16000:.1f}s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"timbre": emb}, str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
