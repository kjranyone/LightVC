"""
E1: 時間平滑化 — Viterbi path retrieval

フレーム独立kNN → 大域最適path

cost(t, j) = d(src_t, bank_j) + λ * |bank_idx_j - bank_idx_{prev}|

追加メトリクス:
  - index jump rate: mean |i_t - i_{t-1}|
  - MCD: mel-cepstral distortion vs DTW-aligned target
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa
from scipy.spatial import cKDTree
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000; FRAME_PERIOD = 5.0; FFTL = 2048; ALPHA = 0.410
MC_ORDER = 24; MC_DIM = 25
VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")
N_PAIRS = 20
CTX = 8
K_CANDIDATES = 20


def analyze_wav(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)
    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)
    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
    return {"f0": f0.astype(np.float32), "mc": mc.astype(np.float32), "ap": ap}


def synth(f0, mc, ap):
    mc64 = np.ascontiguousarray(mc, dtype=np.float64)
    sp = sptk.mc2sp(mc64, ALPHA, FFTL)
    ap64 = np.ascontiguousarray(ap, dtype=np.float64)
    f064 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f064, sp, ap64, SR, frame_period=FRAME_PERIOD).astype(np.float32)


def shift_f0(f0, tgt_mean):
    voiced = f0[f0 > 0]
    if len(voiced) == 0: return f0.astype(np.float64)
    src_mean = float(np.exp(np.mean(np.log(voiced))))
    return np.where(f0 > 0, f0 * tgt_mean / src_mean, 0).astype(np.float64)


def find_pairs(n=20):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    used = set()
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb or sa in used or sb in used: continue
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                used.add(sa); used.add(sb)
                if len(pairs) >= n: return pairs
    return pairs


def compute_speaker_mean(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    all_mc = [np.load(f)["mc"] for f in files]
    return np.concatenate(all_mc, axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    all_mc = [np.load(f)["mc"] for f in files]
    all_f0 = [np.load(f)["f0"] for f in files]
    bank_mc = np.concatenate(all_mc, axis=0).astype(np.float32)
    bank_f0 = np.concatenate(all_f0, axis=0).astype(np.float32)
    utt_boundaries = []
    pos = 0
    for f in files:
        d = np.load(f)
        n = len(d["mc"])
        utt_boundaries.append((pos, pos + n))
        pos += n
    return bank_mc, bank_f0, utt_boundaries


def build_context_key(mc, spk_mean, ctx=8, weights=None):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


def knn_candidates(query_keys, bank_keys, k=20):
    tree = cKDTree(bank_keys)
    dist, idx = tree.query(query_keys, k=k)
    return dist.astype(np.float32), idx.astype(np.int32)


def viterbi_path(dist_matrix, cand_idx, lam_smooth):
    """
    Viterbi to find optimal path through bank candidates.
    
    dist_matrix: (T, K) distances for K candidates per frame
    cand_idx: (T, K) bank indices for K candidates
    lam_smooth: penalty for index jumps between consecutive frames
    
    Returns: bank_indices (T,) — selected bank index per source frame
    """
    T, K = dist_matrix.shape
    
    V = np.zeros((T, K), dtype=np.float64)
    back = np.zeros((T, K), dtype=np.int32)
    
    V[0] = dist_matrix[0]
    
    for t in range(1, T):
        curr_bank = cand_idx[t]  # (K,)
        prev_bank = cand_idx[t-1]  # (K,)
        
        for ki in range(K):
            jumps = np.abs(curr_bank[ki] - prev_bank.astype(np.float64))
            total = V[t-1] + lam_smooth * jumps
            best_prev = np.argmin(total)
            V[t, ki] = dist_matrix[t, ki] + total[best_prev]
            back[t, ki] = best_prev
    
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmin(V[-1])
    for t in range(T-2, -1, -1):
        path[t] = back[t+1, path[t+1]]
    
    return cand_idx[np.arange(T), path]


def soft_blend_from_path(query_keys, bank_keys, bank_values, path_indices, k=3, temp=1.0):
    """Pathの各位置で、path indexの周囲k個でsoft blend"""
    T = len(query_keys)
    result = np.zeros((T, bank_values.shape[1]), dtype=np.float32)
    
    tree = cKDTree(bank_keys)
    for t in range(T):
        center = path_indices[t]
        lo = max(0, center - k + 1)
        hi = min(len(bank_keys), center + k)
        local_keys = bank_keys[lo:hi]
        local_vals = bank_values[lo:hi]
        
        d = np.sqrt(((local_keys - query_keys[t])**2).sum(axis=1))
        w = np.exp(-d / (temp + 1e-10))
        w = w / (w.sum() + 1e-10)
        result[t] = (local_vals * w[:, None]).sum(axis=0)
    
    return result


def compute_mcd(mc_pred, mc_tgt_aligned):
    diff = mc_pred - mc_tgt_aligned
    return float(np.mean(np.sqrt((diff ** 2).sum(axis=1))) * 10 * np.sqrt(2.0 / np.log(10)))


def main():
    DEVICE = torch.device("cuda")
    print("=== E1: Temporal Smoothing (Viterbi) ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    fratio = np.load("data/mc_fratio_weights.npy")
    inv_fratio = (1.0 / (fratio + 1e-6)).astype(np.float32)
    inv_fratio = inv_fratio / inv_fratio.mean()

    pairs = find_pairs(N_PAIRS)
    print(f"Pairs: {len(pairs)}\n")

    lambdas = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = defaultdict(lambda: defaultdict(list))
    speaker_means = {}

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])
        mc_s = feat_s["mc"]; f0_s = feat_s["f0"]; ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]; f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        for spk in [p["src"], p["tgt"]]:
            if spk not in speaker_means:
                speaker_means[spk] = compute_speaker_mean(spk)
        src_mean = speaker_means[p["src"]]
        tgt_mean = speaker_means[p["tgt"]]

        bank_mc, bank_f0, utt_bounds = build_bank(p["tgt"], p["text"], n_utts=10)

        src_keys = build_context_key(mc_s, src_mean, CTX, inv_fratio)
        bank_keys = build_context_key(bank_mc, tgt_mean, CTX, inv_fratio)

        dist_knn, idx_knn = knn_candidates(src_keys, bank_keys, k=K_CANDIDATES)

        dist_dtw, path_dtw = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path_dtw:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for lam in lambdas:
                if lam == 0.0:
                    w = np.exp(-dist_knn[:, :3] / 1.0)
                    w = w / w.sum(axis=1, keepdims=True)
                    mc_pred = np.einsum('nk,nkd->nd', w, bank_mc[idx_knn[:, :3]])
                    name = "baseline_k3"
                else:
                    path = viterbi_path(dist_knn, idx_knn, lam)
                    mc_pred = soft_blend_from_path(
                        src_keys, bank_keys, bank_mc, path, k=3, temp=1.0)
                    name = f"viterbi_l{lam}"

                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])

                sim = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
                mcd = compute_mcd(mc_pred[:T], mc_t_aligned[:T])

                if lam == 0.0:
                    jumps = np.mean(np.abs(np.diff(idx_knn[np.arange(T), np.argmin(dist_knn, axis=1)])))
                else:
                    jumps = np.mean(np.abs(np.diff(path)))

                results[name]["secs"].append(sim)
                results[name]["mcd"].append(mcd)
                results[name]["jumps"].append(jumps)

        if (idx+1) % 5 == 0:
            base_secs = np.mean(results["baseline_k3"]["secs"][-5:])
            best_lam = max(lambdas[1:], key=lambda l: np.mean(results[f"viterbi_l{l}"]["secs"][-5:]))
            best_secs = np.mean(results[f"viterbi_l{best_lam}"]["secs"][-5:])
            print(f"  [{idx+1}/{len(pairs)}] base={base_secs:.3f} "
                  f"best_vit(l={best_lam})={best_secs:.3f}", flush=True)

    print(f"\n{'='*75}")
    print(f"{'config':<20} {'SECS':>8} {'MCD':>8} {'jumps':>8}")
    print(f"{'-'*48}")
    for name in ["baseline_k3"] + [f"viterbi_l{l}" for l in lambdas[1:]]:
        s = np.mean(results[name]["secs"])
        m = np.mean(results[name]["mcd"])
        j = np.mean(results[name]["jumps"])
        print(f"{name:<20} {s:>8.4f} {m:>8.3f} {j:>8.1f}")

    best_name = max(results.keys(), key=lambda k: np.mean(results[k]["secs"]))
    best_score = np.mean(results[best_name]["secs"])
    print(f"\n最高: {best_name} = {best_score:.4f}")
    if best_score >= 0.45:
        print("→ E1 Go条件 (>= 0.45) クリア!")
    elif best_score >= 0.43:
        print("→ 0.45にほぼ到達")
    else:
        print("→ 0.45に届かず — key識別力不足の可能性")

    out = {name: {k: [float(x) for x in v] for k, v in d.items()}
           for name, d in results.items()}
    with open("results/e1_smoothing.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
