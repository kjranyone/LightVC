import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import csv
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

VCTK_WAV = Path("../data/vctk_200")
LATENTS_DIR = Path("data/vctk_latents_200")
DEVICE = torch.device("cuda")

def load_index():
    speakers = defaultdict(dict)
    with open(LATENTS_DIR / "index.tsv") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            speakers[row["speaker_id"]][row["utterance_id"]] = row["path"]
    utt_groups = defaultdict(list)
    for spk, utts in speakers.items():
        for utt_id in utts:
            num = utt_id.split("_")[-1]
            utt_groups[num].append((spk, utt_id))
    return speakers, utt_groups

def load_wavlm_features(wav_path, model, layer=14, max_sec=10):
    import soundfile as sf
    import librosa
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if len(wav) > max_sec * 16000:
        wav = wav[:max_sec * 16000]
    with torch.no_grad():
        t = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)
        outputs = model(t, output_hidden_states=True)
        feat = outputs.hidden_states[layer].squeeze(0).cpu().numpy()
    return feat

def dtw_align(feat_src, feat_tgt):
    from dtw import dtw
    from scipy.spatial.distance import cdist
    cost = cdist(feat_src, feat_tgt, metric="cosine")
    alignment = dtw(cost, step_pattern="symmetric2")
    src_idx = alignment.index1
    tgt_idx = alignment.index2
    return src_idx, tgt_idx

def dac_to_wavlm_frame(dac_frames, ratio=86/50):
    return (np.arange(dac_frames) * ratio).astype(int)

def measure_variance(z_src, z_tgt, spk_mean_src, spk_mean_tgt):
    v_full = z_tgt - z_src
    v_shift = (spk_mean_tgt - spk_mean_src)
    v_shift_expanded = np.broadcast_to(v_shift[:, None], z_src.shape).copy()
    v_residual = v_full - v_shift_expanded
    var_full = np.var(v_full)
    var_shift = np.var(v_shift_expanded)
    var_residual = np.var(v_residual)
    return var_full, var_shift, var_residual

def main():
    speakers, utt_groups = load_index()

    spk_means = {}
    for spk in sorted(speakers.keys()):
        latents = []
        for utt_id, npy_path in speakers[spk].items():
            full_path = Path(npy_path)
            if full_path.exists():
                z = np.load(full_path).astype(np.float32)
                latents.append(z.mean(axis=1))
        if latents:
            spk_means[spk] = np.mean(latents, axis=0)

    print("Loading WavLM...")
    from transformers import AutoModel
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-large").to(DEVICE).eval()

    texts_with_pairs = sorted([t for t, spks in utt_groups.items() if len(spks) >= 10])
    print(f"Texts with >=10 speakers: {len(texts_with_pairs)}")

    np.random.seed(42)
    sample_texts = np.random.choice(texts_with_pairs, min(30, len(texts_with_pairs)), replace=False)

    results_raw = {"var_full": [], "var_shift": [], "var_residual": []}
    results_dtw = {"var_full": [], "var_shift": [], "var_residual": []}

    pbar = tqdm(sample_texts, desc="Processing")
    for text_num in pbar:
        spk_list = utt_groups[text_num]
        if len(spk_list) < 2:
            continue
        n_pairs = min(5, len(spk_list) // 2)
        np.random.shuffle(spk_list)

        pairs = []
        for i in range(0, len(spk_list)-1, 2):
            if len(pairs) >= n_pairs:
                break
            spk_a, utt_a = spk_list[i]
            spk_b, utt_b = spk_list[i+1]
            pairs.append((spk_a, utt_a, spk_b, utt_b))

        for spk_a, utt_a, spk_b, utt_b in pairs:
            npy_a = speakers[spk_a][utt_a]
            npy_b = speakers[spk_b][utt_b]
            path_a = Path(npy_a)
            path_b = Path(npy_b)
            if not path_a.exists() or not path_b.exists():
                continue
            wav_a = VCTK_WAV / spk_a / f"{utt_a}.wav"
            wav_b = VCTK_WAV / spk_b / f"{utt_b}.wav"
            if not wav_a.exists() or not wav_b.exists():
                continue

            z_a = np.load(path_a).astype(np.float32)
            z_b = np.load(path_b).astype(np.float32)
            if z_a.shape[1] < 30 or z_b.shape[1] < 30:
                continue

            min_t = min(z_a.shape[1], z_b.shape[1])
            z_a_raw = z_a[:, :min_t]
            z_b_raw = z_b[:, :min_t]
            vf, vs, vr = measure_variance(z_a_raw, z_b_raw, spk_means[spk_a], spk_means[spk_b])
            results_raw["var_full"].append(vf)
            results_raw["var_shift"].append(vs)
            results_raw["var_residual"].append(vr)

            try:
                feat_a = load_wavlm_features(str(wav_a), wavlm, layer=14)
                feat_b = load_wavlm_features(str(wav_b), wavlm, layer=14)

                dac_a_to_wlm = dac_to_wavlm_frame(z_a.shape[1])
                dac_b_to_wlm = dac_to_wavlm_frame(z_b.shape[1])
                feat_a_dac = feat_a[np.clip(dac_a_to_wlm, 0, feat_a.shape[0]-1)]
                feat_b_dac = feat_b[np.clip(dac_b_to_wlm, 0, feat_b.shape[0]-1)]

                src_idx, tgt_idx = dtw_align(feat_a_dac, feat_b_dac)

                z_a_aligned = z_a[:, src_idx]
                z_b_aligned = z_b[:, tgt_idx]

                vf, vs, vr = measure_variance(z_a_aligned, z_b_aligned, spk_means[spk_a], spk_means[spk_b])
                results_dtw["var_full"].append(vf)
                results_dtw["var_shift"].append(vs)
                results_dtw["var_residual"].append(vr)
            except Exception as e:
                pbar.write(f"DTW failed for {utt_a} vs {utt_b}: {e}")

    print(f"\n=== Results ({len(results_raw['var_full'])} raw, {len(results_dtw['var_full'])} DTW pairs) ===\n")

    for label, results in [("RAW (crop only)", results_raw), ("WavLM L14 DTW", results_dtw)]:
        vf = np.mean(results["var_full"])
        vs = np.mean(results["var_shift"])
        vr = np.mean(results["var_residual"])
        ratio = vs / vf * 100 if vf > 0 else 0
        print(f"{label}:")
        print(f"  var_full = {vf:.4f}")
        print(f"  var_speaker_shift = {vs:.4f} ({ratio:.1f}% of full)")
        print(f"  var_content_residual = {vr:.4f} ({vr/vf*100:.1f}% of full)")
        print()

    raw_ratio = np.mean(results_raw["var_shift"]) / np.mean(results_raw["var_full"]) * 100
    dtw_ratio = np.mean(results_dtw["var_shift"]) / np.mean(results_dtw["var_full"]) * 100 if results_dtw["var_full"] else 0
    print(f"=== Summary ===")
    print(f"Speaker shift ratio: RAW={raw_ratio:.1f}% → DTW={dtw_ratio:.1f}%")
    print(f"Improvement: {dtw_ratio/max(raw_ratio, 0.01):.1f}x")
    print(f"\nJudgment: {'PASS' if dtw_ratio > 10 else 'MARGINAL' if dtw_ratio > 5 else 'FAIL'}")
    print(f"  ratio > 10% → C-4 (same-text pair with DTW) viable")
    print(f"  ratio < 5% → C-1 (timbre shift) as primary")

if __name__ == "__main__":
    main()
