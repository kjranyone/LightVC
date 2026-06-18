"""
kNN-VC with bottleneck-content matching + DAC-latent replacement.

Match on bottleneck output (speaker-invariant content features, 256-dim) to find
the phonetically similar reference frame, then replace in full DAC latent space.
Since all reference frames are from the target speaker, speaker is auto-converted.

This is "codec latentから軽量tokenizerで分ける" (CONCEPT.md:170) in practice:
  bottleneck = lightweight content tokenizer from codec latent
  kNN = frame translation using content similarity

No external model at inference (no WavLM/HuBERT). Pure codec-space.
"""
import sys, csv, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from converter import Converter, ConverterConfig
from infer_flow import encode, decode, load_dac

DEVICE = torch.device("cuda")
VCTK_WAV = Path("../data/vctk_200")


def load_bottleneck(ckpt_path):
    """Load Phase B bottleneck encoder for content feature extraction."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ConverterConfig(**ckpt["config"]["model"])
    model = Converter(config).to(DEVICE)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


def knn_convert_bottleneck(z_src, z_ref, model, k=4, temperature=0.1):
    """kNN-VC using bottleneck content features for matching.

    1. Extract content features: c_src = bottleneck(z_src), c_ref = bottleneck(z_ref)
    2. Match: for each c_src frame, find k nearest c_ref frames
    3. Replace: z_out = weighted mean of matched z_ref frames
    """
    with torch.no_grad():
        c_src = model.content_code(z_src.unsqueeze(0)).squeeze(0)
        c_ref = model.content_code(z_ref.unsqueeze(0)).squeeze(0)

        c_src_norm = F.normalize(c_src, dim=0)
        c_ref_norm = F.normalize(c_ref, dim=0)
        sim = c_src_norm.t() @ c_ref_norm

        topk_sim, topk_idx = sim.topk(min(k, c_ref.shape[1]), dim=-1)
        weights = F.softmax(topk_sim / temperature, dim=-1)

        z_out = torch.zeros_like(z_src)
        for t in range(z_src.shape[1]):
            neighbors = z_ref[:, topk_idx[t]]
            z_out[:, t] = (neighbors * weights[t].unsqueeze(0)).sum(dim=-1)

    return z_out


def knn_convert_hybrid(z_src, z_ref, model, k=4, temperature=0.1,
                       content_weight=0.7, speaker_weight=0.3):
    """kNN-VC with weighted content+speaker matching.

    Match distance = content_weight * content_sim + speaker_weight * latent_sim
    This balances phonetic matching with acoustic matching.
    """
    with torch.no_grad():
        c_src = model.content_code(z_src.unsqueeze(0)).squeeze(0)
        c_ref = model.content_code(z_ref.unsqueeze(0)).squeeze(0)

        c_sim = F.normalize(c_src, dim=0).t() @ F.normalize(c_ref, dim=0)
        z_sim = F.normalize(z_src, dim=0).t() @ F.normalize(z_ref, dim=0)
        combined = content_weight * c_sim + speaker_weight * z_sim

        topk_sim, topk_idx = combined.topk(min(k, z_ref.shape[1]), dim=-1)
        weights = F.softmax(topk_sim / temperature, dim=-1)

        z_out = torch.zeros_like(z_src)
        for t in range(z_src.shape[1]):
            neighbors = z_ref[:, topk_idx[t]]
            z_out[:, t] = (neighbors * weights[t].unsqueeze(0)).sum(dim=-1)

    return z_out


def main():
    print("=== kNN-VC: Bottleneck-Content Matching ===\n")

    dac, device = load_dac()
    model = load_bottleneck("checkpoints/phase_b_distill/best.pt")
    print("Loaded Phase B bottleneck (content encoder)")

    import librosa
    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(device)},
    )

    def get_embed(wav_44k):
        wav16k = librosa.resample(wav_44k.astype(np.float32), orig_sr=44100, target_sr=16000)
        t = torch.from_numpy(wav16k).float().unsqueeze(0).to(device)
        return secs_model.encode_batch(t).squeeze().cpu()

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
        if len(wavs) >= 3:
            src_wav = wavs[0]
            ref_wav = wavs[1]
            pairs.append((str(src_wav), str(ref_wav), str(src_wav)))
        if len(pairs) >= 8:
            break

    cross_pairs = []
    for i, spk in enumerate(heldout_list[:10]):
        spk_dir = VCTK_WAV / spk
        if not spk_dir.exists():
            continue
        wavs = sorted(spk_dir.glob("*.wav"))
        if len(wavs) < 2:
            continue
        other_spks = [s for s in heldout_list if s != spk and (VCTK_WAV / s).exists()]
        if not other_spks:
            continue
        tgt_spk = other_spks[i % len(other_spks)]
        ref_wavs = sorted((VCTK_WAV / tgt_spk).glob("*.wav"))
        if ref_wavs:
            cross_pairs.append((str(wavs[0]), str(ref_wavs[0]), str(wavs[1])))
        if len(cross_pairs) >= 8:
            break

    print(f"Self-recon pairs: {len(pairs)}, Cross-speaker pairs: {len(cross_pairs)}")

    print("\n--- Self-reconstruction (identity check) ---")
    for label, fn in [
        ("identity", lambda z_s, z_r: z_s),
        ("kNN bottleneck k=4", lambda z_s, z_r: knn_convert_bottleneck(z_s, z_r, model, k=4)),
    ]:
        scores = []
        for src_path, ref_path, tgt_path in pairs:
            src_w, _ = sf.read(src_path, dtype="float32")
            ref_w, _ = sf.read(ref_path, dtype="float32")
            rem = len(src_w) % 512
            if rem: src_w = np.pad(src_w, (0, 512-rem))
            rem = len(ref_w) % 512
            if rem: ref_w = np.pad(ref_w, (0, 512-rem))
            z_src = encode(dac, src_w, device)
            z_ref = encode(dac, ref_w, device)
            z_out = fn(z_src, z_ref)
            conv_wav = decode(dac, z_out, device)
            tgt_w, _ = sf.read(tgt_path, dtype="float32")
            tgt_embed = get_embed(tgt_w)
            conv_embed = get_embed(conv_wav)
            scores.append(F.cosine_similarity(conv_embed.unsqueeze(0), tgt_embed.unsqueeze(0), dim=-1).item())
        print(f"  {label}: SECS = {np.mean(scores):.4f}")

    print("\n--- Cross-speaker conversion ---")
    for label, fn in [
        ("identity", lambda z_s, z_r: z_s),
        ("raw DAC k=4", lambda z_s, z_r: knn_convert_raw(z_s, z_r, k=4)),
        ("bottleneck k=4 T=0.1", lambda z_s, z_r: knn_convert_bottleneck(z_s, z_r, model, k=4, temperature=0.1)),
        ("bottleneck k=4 T=0.01", lambda z_s, z_r: knn_convert_bottleneck(z_s, z_r, model, k=4, temperature=0.01)),
        ("bottleneck k=1 T=0.01", lambda z_s, z_r: knn_convert_bottleneck(z_s, z_r, model, k=1, temperature=0.01)),
        ("hybrid 0.7/0.3 k=4", lambda z_s, z_r: knn_convert_hybrid(z_s, z_r, model, k=4, content_weight=0.7, speaker_weight=0.3)),
        ("hybrid 0.5/0.5 k=4", lambda z_s, z_r: knn_convert_hybrid(z_s, z_r, model, k=4, content_weight=0.5, speaker_weight=0.5)),
    ]:
        scores = []
        for src_path, ref_path, _ in cross_pairs:
            src_w, _ = sf.read(src_path, dtype="float32")
            ref_w, _ = sf.read(ref_path, dtype="float32")
            rem = len(src_w) % 512
            if rem: src_w = np.pad(src_w, (0, 512-rem))
            rem = len(ref_w) % 512
            if rem: ref_w = np.pad(ref_w, (0, 512-rem))
            z_src = encode(dac, src_w, device)
            z_ref = encode(dac, ref_w, device)
            z_out = fn(z_src, z_ref)
            conv_wav = decode(dac, z_out, device)
            ref_embed = get_embed(ref_w)
            conv_embed = get_embed(conv_wav)
            scores.append(F.cosine_similarity(conv_embed.unsqueeze(0), ref_embed.unsqueeze(0), dim=-1).item())
        print(f"  {label}: SECS = {np.mean(scores):.4f}  (scores: {['%.3f' % s for s in scores[:4]]})")

    print(f"\nTarget: SECS > 0.50")


def knn_convert_raw(z_src, z_ref, k=4):
    z_src_norm = F.normalize(z_src, dim=0)
    z_ref_norm = F.normalize(z_ref, dim=0)
    sim = z_src_norm.t() @ z_ref_norm
    topk_sim, topk_idx = sim.topk(min(k, z_ref.shape[1]), dim=-1)
    weights = F.softmax(topk_sim * 5.0, dim=-1)
    z_out = torch.zeros_like(z_src)
    for t in range(z_src.shape[1]):
        neighbors = z_ref[:, topk_idx[t]]
        z_out[:, t] = (neighbors * weights[t].unsqueeze(0)).sum(dim=-1)
    return z_out


if __name__ == "__main__":
    main()
