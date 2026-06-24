"""
B diagnostic: Oracle reranker

top-N候補に正解（DTW targetに最も近い候補）があるか確認

もしoracle rerankで0.48超え → 学習可能rerankerで改善可能
超えない → 候補集合そのものが不適 → C方向（unit/sequence）へ

追加: top-1の命中率（正解がrank-1にある割合）も測る
"""
import sys, json
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
    return np.concatenate([np.load(f)["mc"] for f in files], axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    bank_mc = np.concatenate([np.load(f)["mc"] for f in files], axis=0).astype(np.float32)
    return bank_mc


def build_context_key(mc, spk_mean, ctx=8, weights=None):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


def main():
    DEVICE = torch.device("cuda")
    print("=== B Diagnostic: Oracle Reranker ===\n")

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

    N_CANDIDATES = 20

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

        bank_mc = build_bank(p["tgt"], p["text"], n_utts=10)

        src_keys = build_context_key(mc_s, src_mean, 8, inv_fratio)
        bank_keys = build_context_key(bank_mc, tgt_mean, 8, inv_fratio)

        tree = cKDTree(bank_keys)
        dist_knn, idx_knn = tree.query(src_keys, k=N_CANDIDATES)

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

            # 1. baseline: top-3 soft blend
            w3 = np.exp(-dist_knn[:, :3] / 1.0)
            w3 = w3 / w3.sum(axis=1, keepdims=True)
            mc_base = np.einsum('nk,nkd->nd', w3, bank_mc[idx_knn[:, :3]])
            wav_base = synth(f0_shifted[:T], mc_base[:T].astype(np.float32), ap_s[:T])
            sim_base = F.cosine_similarity(e_tgt, emb(wav_base), dim=-1).item()
            results["baseline_k3"].append(sim_base)

            # 2. oracle rerank: pick candidate closest to DTW-aligned target
            oracle_mc = np.zeros((T, MC_DIM), dtype=np.float32)
            top1_correct = 0
            for t in range(T):
                cands = bank_mc[idx_knn[t]]  # (N, mc_dim)
                target_frame = mc_t_aligned[t]
                cand_dist = np.sqrt(((cands - target_frame)**2).sum(axis=1))
                best_c = np.argmin(cand_dist)
                oracle_mc[t] = cands[best_c]
                if best_c == 0:
                    top1_correct += 1

            wav_oracle_rr = synth(f0_shifted[:T], oracle_mc[:T], ap_s[:T])
            sim_oracle_rr = F.cosine_similarity(e_tgt, emb(wav_oracle_rr), dim=-1).item()
            results["oracle_rerank"].append(sim_oracle_rr)
            results["top1_hit_rate"].append(top1_correct / T)

            # 3. oracle rerank with top-3 soft blend of best candidates
            oracle_blend = np.zeros((T, MC_DIM), dtype=np.float32)
            for t in range(T):
                cands = bank_mc[idx_knn[t]]
                target_frame = mc_t_aligned[t]
                cand_dist = np.sqrt(((cands - target_frame)**2).sum(axis=1))
                best3 = np.argsort(cand_dist)[:3]
                w = np.exp(-cand_dist[best3] / 1.0)
                w = w / w.sum()
                oracle_blend[t] = (cands[best3] * w[:, None]).sum(axis=0)

            wav_ob = synth(f0_shifted[:T], oracle_blend[:T], ap_s[:T])
            sim_ob = F.cosine_similarity(e_tgt, emb(wav_ob), dim=-1).item()
            results["oracle_rerank_blend3"].append(sim_ob)

            # 4. pure oracle (DTW-aligned target) for reference
            wav_pure = synth(f0_shifted[:T], mc_t_aligned[:T].astype(np.float32), ap_s[:T])
            sim_pure = F.cosine_similarity(e_tgt, emb(wav_pure), dim=-1).item()
            results["dtw_oracle"].append(sim_pure)

        print(f"  [{idx+1}/{len(pairs)}] base={sim_base:.3f} "
              f"oracle_rr={sim_oracle_rr:.3f} blend3={sim_ob:.3f} "
              f"dtw={sim_pure:.3f} hit={top1_correct/T:.2f}", flush=True)

    print(f"\n{'='*65}")
    print(f"{'config':<25} {'mean':>8} {'std':>8}")
    print(f"{'-'*45}")
    for name in ["baseline_k3", "oracle_rerank", "oracle_rerank_blend3", "dtw_oracle"]:
        arr = np.array(results[name])
        print(f"{name:<25} {arr.mean():>8.4f} {arr.std():>8.4f}")
    print(f"\ntop-1 hit rate (正解がkNN rank-1): {np.mean(results['top1_hit_rate']):.3f}")

    oracle_rr = np.mean(results["oracle_rerank"])
    print(f"\nOracle rerank = {oracle_rr:.4f}")
    if oracle_rr >= 0.48:
        print("→ 候補内に正解あり、reracker路線 viable")
    elif oracle_rr >= 0.45:
        print("→ 候補内に一定程度正解あり")
    else:
        print("→ 候補内に正解なし、候補生成そのもの要改善")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/b_oracle_rerank.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
