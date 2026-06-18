"""
Precompute WavLM L6 features for all VCTK utterances.
Used for kNN-VC target generation during FlowConverter distillation training.

Output: data/wavlm_l6/{speaker}/{utt_id}.npy  [T_wlm, 1024]
Frame rate: 50 Hz (20ms hop)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

VCTK_WAV = Path("../data/vctk_200")
OUT_DIR = Path("data/wavlm_l6")
DEVICE = torch.device("cuda")


def main():
    print("=== WavLM L6 Feature Precompute ===\n")

    model = AutoModel.from_pretrained("microsoft/wavlm-large").to(DEVICE).eval()
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")

    utts = []
    for spk_dir in sorted(VCTK_WAV.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        out_spk = OUT_DIR / spk
        out_spk.mkdir(parents=True, exist_ok=True)
        for wav_path in spk_dir.glob("*.wav"):
            out_path = out_spk / (wav_path.stem + ".npy")
            if out_path.exists():
                continue
            utts.append((wav_path, out_path, spk))

    print(f"Total utterances: {len(utts)}")

    with torch.no_grad():
        for wav_path, out_path, spk in tqdm(utts, desc="WavLM L6"):
            wav, sr = sf.read(str(wav_path), dtype="float32")
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            if len(wav) < 320:
                continue

            inputs = extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
            outputs = model(inputs, output_hidden_states=True)
            feat = outputs.hidden_states[6].squeeze(0).cpu().numpy()

            np.save(out_path, feat.astype(np.float32))

    print(f"\nDone. Features saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
