import sys, csv, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

VCTK_WAV = Path("../data/vctk_200")
LATENTS_DIR = Path("data/vctk_latents_200")
DEVICE = torch.device("cuda")
CACHE_PATH = Path("data/wavlm_sv_embeddings.pkl")

def main():
    print("=== 09-B1: WavLM-SV embedding precompute ===\n")

    model = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

    speakers = {}
    with open(LATENTS_DIR / "index.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            spk = row["speaker_id"]
            utt_id = row["utterance_id"]
            speakers.setdefault(spk, []).append(utt_id)

    total = sum(len(v) for v in speakers.values())
    print(f"Speakers: {len(speakers)}, Utterances: {total}")

    cache = {}
    pbar = tqdm(total=total, desc="Embedding")
    for spk, utts in sorted(speakers.items()):
        for utt_id in utts:
            wav_path = VCTK_WAV / spk / f"{utt_id}.wav"
            if not wav_path.exists():
                pbar.update(1)
                continue

            wav, sr = sf.read(str(wav_path), dtype="float32")
            if sr != 16000:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            if len(wav) > 15 * 16000:
                wav = wav[:15 * 16000]

            with torch.no_grad():
                inputs = extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
                outputs = model(input_values=inputs)
                embed = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                embed = F.normalize(embed, dim=-1).cpu().numpy()

            cache[f"{spk}/{utt_id}"] = embed
            pbar.update(1)
    pbar.close()

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    print(f"\nCached {len(cache)} embeddings to {CACHE_PATH}")
    sample_key = list(cache.keys())[0]
    print(f"Sample: {sample_key} → {cache[sample_key].shape}")

if __name__ == "__main__":
    main()
