"""
kNN-VC baseline in DAC latent space.

For each source frame, find k nearest reference frames (from target speaker)
in DAC latent space, replace with their average. No training required.

Pipeline:
  source WAV → DAC encode → z_src [1024, T_src]
  ref WAV    → DAC encode → z_ref [1024, T_ref]
  For each t in T_src:
    z_out[:, t] = mean(top-k nearest z_ref frames to z_src[:, t])
  z_out → DAC decode → output WAV

CONCEPT.md alignment: "codec tokenをリアルタイムに翻訳するVC"
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from infer_flow import encode, decode, load_dac

DEVICE = torch.device("cuda")
VCTK_WAV = Path("../data/vctk_200")


def knn_convert(z_src, z_ref, k=4):
    """kNN-VC: replace each source frame with mean of k nearest ref frames.

    Args:
        z_src: [D, T_src] — source DAC latent
        z_ref: [D, T_ref] — reference DAC latent (target speaker)
        k: number of nearest neighbors

    Returns:
        z_out: [D, T_src] — converted DAC latent
    """
    z_src_norm = F.normalize(z_src, dim=0)
    z_ref_norm = F.normalize(z_ref, dim=0)
    sim = z_src_norm.t() @ z_ref_norm
    topk_sim, topk_idx = sim.topk(k, dim=-1)
    weights = F.softmax(topk_sim * 5.0, dim=-1)
    z_out = torch.zeros_like(z_src)
    for t in range(z_src.shape[1]):
        neighbors = z_ref[:, topk_idx[t]]
        z_out[:, t] = (neighbors * weights[t].unsqueeze(0)).sum(dim=-1)
    return z_out


def evaluate_secs(converter_fn, pairs, dac, device, secs_model=None):
    """Evaluate SECS for a conversion function."""
    import librosa
    from speechbrain.inference.speaker import EncoderClassifier

    if secs_model is None:
        secs_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="hf_models/spkrec-ecapa",
            run_opts={"device": str(device)},
        )

    def get_embed(wav_44k):
        wav16k = librosa.resample(wav_44k.astype(np.float32), orig_sr=44100, target_sr=16000)
        t = torch.from_numpy(wav16k).float().unsqueeze(0).to(device)
        return secs_model.encode_batch(t).squeeze().cpu()

    scores = []
    for src_path, ref_path in pairs:
        src_w, sr = sf.read(str(src_path), dtype="float32")
        ref_w, _ = sf.read(str(ref_path), dtype="float32")
        if sr != 44100:
            src_w = librosa.resample(src_w, orig_sr=sr, target_sr=44100)
            ref_w = librosa.resample(ref_w, orig_sr=sr, target_sr=44100)
        rem = len(src_w) % 512
        if rem:
            src_w = np.pad(src_w, (0, 512 - rem))
        rem = len(ref_w) % 512
        if rem:
            ref_w = np.pad(ref_w, (0, 512 - rem))

        z_src = encode(dac, src_w, device)
        z_ref = encode(dac, ref_w, device)
        z_out = converter_fn(z_src, z_ref)
        conv_wav = decode(dac, z_out, device)

        ref_embed = get_embed(ref_w)
        conv_embed = get_embed(conv_wav)
        sim = F.cosine_similarity(conv_embed.unsqueeze(0), ref_embed.unsqueeze(0), dim=-1).item()
        scores.append(sim)

    return np.mean(scores), scores


def main():
    print("=== kNN-VC Baseline in DAC Latent Space ===\n")

    dac, device = load_dac()

    import csv
    index_path = Path("data/vctk_latents_200/index.tsv")
    speakers_set = set()
    with open(index_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            speakers_set.add(row["speaker_id"])
    spk_list = sorted(speakers_set)
    np.random.seed(42)
    heldout = set(np.random.choice(spk_list, 19, replace=False))

    pairs = []
    heldout_list = sorted(heldout)
    for spk in heldout_list[:10]:
        spk_dir = VCTK_WAV / spk
        if not spk_dir.exists():
            continue
        wavs = sorted(spk_dir.glob("*.wav"))
        if len(wavs) >= 2:
            src_wav = wavs[0]
            other_spks = [s for s in heldout_list if s != spk and (VCTK_WAV / s).exists()]
            if other_spks:
                tgt_spk = np.random.choice(other_spks)
                ref_wavs = sorted((VCTK_WAV / tgt_spk).glob("*.wav"))
                if ref_wavs:
                    pairs.append((str(src_wav), str(ref_wavs[0])))
        if len(pairs) >= 8:
            break

    print(f"Eval pairs: {len(pairs)} (held-out speakers)")

    print("\n--- Identity baseline ---")
    mean_secs, _ = evaluate_secs(lambda z_s, z_r: z_s, pairs, dac, device)
    print(f"Identity SECS: {mean_secs:.4f}")

    for k in [1, 2, 4, 8]:
        print(f"\n--- kNN-VC k={k} ---")
        mean_secs, scores = evaluate_secs(
            lambda z_s, z_r, _k=k: knn_convert(z_s, z_r, k=_k),
            pairs, dac, device
        )
        print(f"k={k}: SECS = {mean_secs:.4f}  (per-pair: {['%.3f' % s for s in scores[:4]]})")

    print(f"\n=== Summary ===")
    print(f"Target: SECS > 0.50")
    print(f"Judgment: see above")


if __name__ == "__main__":
    main()
