"""
Oracle check: evaluate SECS with GROUND TRUTH DTW-aligned target mel-cepstrum.

This verifies the evaluation pipeline. Should match O1d ≈ 0.66.
"""
import sys, os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import pyworld as world
import pysptk as sptk
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from eval_sf_vc import load_speaker_embeddings, compute_speaker_f0_stats, shift_f0, synth_wav

DEVICE = torch.device("cuda")
SR = 16000
PAIRS_DIR = Path("data/sf_pairs")
VCTK_WAV = Path("../data/vctk_200")


def main():
    print("=== Oracle Check: Ground Truth Target MC ===\n")

    f0_stats = compute_speaker_f0_stats()

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pair_files = sorted(PAIRS_DIR.glob("pair_*.npz"))[:30]

    tgt_scores_f0, src_scores_f0 = [], []
    tgt_scores_nof0, src_scores_nof0 = [], []

    for i, npz_path in enumerate(pair_files):
        data = np.load(npz_path, allow_pickle=True)
        mc_tgt = data["mc_tgt"]  # Ground truth aligned target
        f0_src = data["f0_src"]
        codeap_tgt = data["codeap_tgt"]
        spk_src = str(data["spk_src"])
        spk_tgt = str(data["spk_tgt"])

        T = min(len(mc_tgt), 800)

        tgt_mean_f0 = f0_stats.get(spk_tgt, 200.0)
        f0_original = f0_src[:T].astype(np.float64)
        f0_shifted = shift_f0(f0_src[:T], tgt_mean_f0)

        wav_syn_f0 = synth_wav(mc_tgt[:T], f0_shifted, codeap_tgt[:T])
        wav_syn_nof0 = synth_wav(mc_tgt[:T], f0_original, codeap_tgt[:T])

        src_wavs = list((VCTK_WAV / spk_src).glob("*.wav"))
        tgt_wavs = list((VCTK_WAV / spk_tgt).glob("*.wav"))
        if not src_wavs or not tgt_wavs:
            continue

        wav_src, sr = sf.read(str(src_wavs[0]), dtype="float32")
        wav_tgt, sr = sf.read(str(tgt_wavs[0]), dtype="float32")
        if sr != SR:
            wav_src = librosa.resample(wav_src, orig_sr=sr, target_sr=SR)
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            e_src = secs_model.encode_batch(torch.from_numpy(wav_src.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = secs_model.encode_batch(torch.from_numpy(wav_tgt.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_syn_f0 = secs_model.encode_batch(torch.from_numpy(wav_syn_f0.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_syn_nof0 = secs_model.encode_batch(torch.from_numpy(wav_syn_nof0.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)

        tgt_f0 = F.cosine_similarity(e_tgt, e_syn_f0, dim=-1).item()
        src_f0 = F.cosine_similarity(e_src, e_syn_f0, dim=-1).item()
        tgt_nof0 = F.cosine_similarity(e_tgt, e_syn_nof0, dim=-1).item()
        src_nof0 = F.cosine_similarity(e_src, e_syn_nof0, dim=-1).item()

        tgt_scores_f0.append(tgt_f0)
        src_scores_f0.append(src_f0)
        tgt_scores_nof0.append(tgt_nof0)
        src_scores_nof0.append(src_nof0)

        print(f"  [{i+1}/30] {spk_src}→{spk_tgt}: F0={tgt_f0:.3f}/{src_f0:.3f} noF0={tgt_nof0:.3f}/{src_nof0:.3f}", flush=True)

    print(f"\n=== Oracle Check Results ===")
    print(f"With F0 shift:")
    print(f"  SECS(tgt): {np.mean(tgt_scores_f0):.4f} ± {np.std(tgt_scores_f0):.4f}")
    print(f"  SECS(src): {np.mean(src_scores_f0):.4f} ± {np.std(src_scores_f0):.4f}")
    print(f"Without F0 shift:")
    print(f"  SECS(tgt): {np.mean(tgt_scores_nof0):.4f} ± {np.std(tgt_scores_nof0):.4f}")
    print(f"  SECS(src): {np.mean(src_scores_nof0):.4f} ± {np.std(src_scores_nof0):.4f}")


if __name__ == "__main__":
    main()
