"""
Heuristic reranker for top-20 kNN candidates

score(i, t) =
  - α d_content(q_t, k_i)
  + β s_speaker(v_i, profile_tgt_register)
  - γ d_f0reg(t, i)
  - δ d_energy(t, i)
  - η jump_cost(i, i_prev)

speaker profile: per-register (μ, σ) with diagonal variance clipping
NOT full z-score (that was catastrophic at 0.18)
"""
import sys, json, time, pickle
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
N_CAND = 20


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


def compute_register(f0, n_bins=4):
    T = len(f0)
    reg = np.zeros(T, dtype=np.int32)
    voiced = f0 > 0
    if voiced.any():
        voiced_f0 = f0[voiced]
        quartiles = np.percentile(np.log(voiced_f0), [25, 50, 75])
        reg[voiced] = np.digitize(np.log(f0[voiced]), quartiles) + 1
    return reg


_speaker_profile_cache = {}

def get_speaker_profile(spk_id):
    if spk_id in _speaker_profile_cache:
        return _speaker_profile_cache[spk_id]
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:30]
    all_mc = []; all_reg = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
        all_reg.append(compute_register(d["f0"]))
    all_mc = np.concatenate(all_mc).astype(np.float32)
    all_reg = np.concatenate(all_reg)

    profiles = {}
    global_mean = all_mc.mean(axis=0)
    global_std = all_mc.std(axis=0) + 1e-3
    for r in range(5):
        mask = all_reg == r
        if mask.sum() > 20:
            profiles[r] = (all_mc[mask].mean(axis=0), all_mc[mask].std(axis=0) + 1e-3)
        else:
            profiles[r] = (global_mean, global_std)
    _speaker_profile_cache[spk_id] = profiles
    return profiles


def compute_speaker_mean(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    return np.concatenate([np.load(f)["mc"] for f in files], axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    bank_mc = np.concatenate([np.load(f)["mc"] for f in files], axis=0).astype(np.float32)
    bank_f0 = np.concatenate([np.load(f)["f0"] for f in files], axis=0).astype(np.float32)
    bank_reg = compute_register(bank_f0)
    return bank_mc, bank_f0, bank_reg


def build_context_key(mc, spk_mean, ctx=8, weights=None):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


def speaker_score(cand_mc, profile_mean, profile_std, clip_val=2.0):
    z = (cand_mc - profile_mean) / profile_std
    z = np.clip(z, -clip_val, clip_val)
    return -np.sum(z ** 2)


def rerank_select(d_content, cands_mc, cands_bank_idx, cands_f0, cands_reg,
                  src_f0, src_energy, src_reg,
                  tgt_profile, prev_bank_idx,
                  alpha, beta, gamma, delta, eta, clip_val,
                  blend_k=3):
    N = len(d_content)
    scores = np.zeros(N)

    reg_key = min(src_reg, 4)
    prof_mean, prof_std = tgt_profile.get(reg_key, tgt_profile[0])

    for i in range(N):
        scores[i] = -alpha * d_content[i]

        if beta > 0:
            s = speaker_score(cands_mc[i], prof_mean, prof_std, clip_val)
            scores[i] += beta * s

        if gamma > 0:
            if src_f0 > 0 and cands_f0[i] > 0:
                d_f0 = abs(np.log(src_f0) - np.log(cands_f0[i]))
            elif src_f0 <= 0 and cands_f0[i] <= 0:
                d_f0 = 0.0
            else:
                d_f0 = 5.0
            scores[i] -= gamma * d_f0

        if delta > 0:
            scores[i] -= delta * abs(src_energy - cands_mc[i, 0])

        if eta > 0 and prev_bank_idx is not None:
            scores[i] -= eta * abs(cands_bank_idx[i] - prev_bank_idx)

    if blend_k > 1:
        top_k = min(blend_k, N)
        best_idx = np.argsort(-scores)[:top_k]
        s_top = scores[best_idx]
        s_top = s_top - s_top.max()
        w = np.exp(s_top)
        w = w / (w.sum() + 1e-10)
        result_mc = (cands_mc[best_idx] * w[:, None]).sum(axis=0)
        result_bank_idx = cands_bank_idx[best_idx[np.argmax(w)]]
    else:
        best = np.argmax(scores)
        result_mc = cands_mc[best]
        result_bank_idx = cands_bank_idx[best]

    return result_mc, result_bank_idx


def main():
    DEVICE = torch.device("cuda")
    print("=== Heuristic Reranker ===\n")

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

    configs = [
        ("baseline",          1.0, 0.0,  0.0, 0.0, 0.0,   2.0, 3),
        ("spk_b0.01",         1.0, 0.01, 0.0, 0.0, 0.0,   2.0, 3),
        ("spk_b0.05",         1.0, 0.05, 0.0, 0.0, 0.0,   2.0, 3),
        ("spk_b0.1",          1.0, 0.1,  0.0, 0.0, 0.0,   2.0, 3),
        ("spk_b0.2",          1.0, 0.2,  0.0, 0.0, 0.0,   2.0, 3),
        ("spk_b0.1_clip1.5",  1.0, 0.1,  0.0, 0.0, 0.0,   1.5, 3),
        ("spk_b0.1_clip3",    1.0, 0.1,  0.0, 0.0, 0.0,   3.0, 3),
        ("spk_b0.1_f0_0.05",  1.0, 0.1,  0.05,0.0, 0.0,   2.0, 3),
        ("spk_b0.1_e_0.01",   1.0, 0.1,  0.0, 0.01,0.0,   2.0, 3),
        ("spk_b0.1_jp_0.001", 1.0, 0.1,  0.0, 0.0, 0.001, 2.0, 3),
        ("full_b0.1",         1.0, 0.1,  0.05,0.01,0.001, 2.0, 3),
        ("spk_b0.2_k1",       1.0, 0.2,  0.0, 0.0, 0.0,   2.0, 1),
        ("oracle_rr",         0.0, 0.0,  0.0, 0.0, 0.0,   2.0, 0),
    ]

    results = defaultdict(list)
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

        bank_mc, bank_f0, bank_reg = build_bank(p["tgt"], p["text"], n_utts=10)
        tgt_profile = get_speaker_profile(p["tgt"])

        src_keys = build_context_key(mc_s, src_mean, CTX, inv_fratio)
        bank_keys = build_context_key(bank_mc, tgt_mean, CTX, inv_fratio)

        tree = cKDTree(bank_keys)
        dist_knn, idx_knn = tree.query(src_keys, k=N_CAND)

        src_reg = compute_register(f0_s)

        # DTW oracle for reference
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

            for name, alpha, beta, gamma, delta, eta, clip_val, blend_k in configs:
                mc_pred = np.zeros((T, MC_DIM), dtype=np.float32)
                prev_bank = None

                for t in range(T):
                    cands_bank_idx = idx_knn[t]
                    cands_mc = bank_mc[cands_bank_idx]
                    cands_f0 = bank_f0[cands_bank_idx]
                    cands_reg = bank_reg[cands_bank_idx]
                    d_content = dist_knn[t]

                    if name == "oracle_rr":
                        cand_dist = np.sqrt(((cands_mc - mc_t_aligned[t])**2).sum(axis=1))
                        best3 = np.argsort(cand_dist)[:3]
                        w = np.exp(-cand_dist[best3] / 1.0)
                        w = w / w.sum()
                        mc_pred[t] = (cands_mc[best3] * w[:, None]).sum(axis=0)
                        prev_bank = cands_bank_idx[best3[0]]
                    else:
                        mc_out, prev_bank = rerank_select(
                            d_content, cands_mc, cands_bank_idx, cands_f0, cands_reg,
                            f0_s[t], mc_s[t, 0], src_reg[t],
                            tgt_profile, prev_bank,
                            alpha, beta, gamma, delta, eta, clip_val, blend_k)
                        mc_pred[t] = mc_out

                wav_out = synth(f0_shifted[:T], mc_pred[:T], ap_s[:T])
                sim = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
                results[name].append(sim)

        if (idx+1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] " + " | ".join(
                f"{n}={np.mean(results[n][-5:]):.3f}"
                for n, *_ in configs[:5]), flush=True)

    print(f"\n{'='*65}")
    print(f"{'config':<22} {'mean':>8} {'std':>8} {'vs base':>8}")
    print(f"{'-'*52}")
    base = np.mean(results["baseline"])
    for name, *_ in configs:
        arr = np.array(results[name])
        print(f"{name:<22} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.mean()-base:>+8.4f}")

    best_name = max([c[0] for c in configs], key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_name])
    print(f"\n最高: {best_name} = {best_score:.4f}")
    if best_score >= 0.48:
        print("→ learned reranker Go条件 (>= 0.48) 相当!")
    elif best_score >= 0.46:
        print("→ heuristic Go条件 (>= 0.46) クリア! learnedに進む")
    elif best_score >= 0.44:
        print("→ 改善あり、パラメータ調整で伸びる可能性")
    else:
        print("→ 改善不十分")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/reranker_heuristic.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
