"""
Fixed retrieval v2 — 学習なしでkey設計を改善してSECSを伸ばす

テストする改善:
  1. context window (±N frames) — 単一フレーム→temporal context
  2. delta mcep — 時間変動特徴
  3. F0 register matching — pitch binで検索空間を絞る
  4. top-k blending with τ sweep
  5. register-stratified bank

baseline (single-frame norm mcep, top-5, τ=1.0): 0.333 (oracle, no leakage)
目標: >= 0.42
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

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000
FRAME_PERIOD = 5.0
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24
MC_DIM = 25
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
    all_mc = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
    return np.concatenate(all_mc, axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    all_mc = []
    all_f0 = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
        all_f0.append(d["f0"])
    return np.concatenate(all_mc, axis=0).astype(np.float32), np.concatenate(all_f0, axis=0).astype(np.float32)


def build_context_key(mc, f0, spk_mean, ctx=0, use_delta=False, use_f0_reg=False):
    T = len(mc)
    mc_norm = mc - spk_mean

    parts = []

    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        ctx_frames = np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1)  # (T, mc_dim, 2*ctx+1)
        ctx_flat = ctx_frames.reshape(T, -1)  # (T, mc_dim * (2*ctx+1))
        parts.append(ctx_flat)
    else:
        parts.append(mc_norm)

    if use_delta:
        delta = np.zeros_like(mc_norm)
        delta[1:] = mc_norm[1:] - mc_norm[:-1]
        parts.append(delta)

    if use_f0_reg:
        f0_norm = np.zeros((T, 1), dtype=np.float32)
        voiced = f0 > 0
        if voiced.any():
            f0_log = np.log(f0[voiced].mean())
            f0_norm[voiced, 0] = f0_log
        parts.append(f0_norm)

    return np.concatenate(parts, axis=1).astype(np.float32)


def retrieve(query_keys, bank_keys, bank_values, k=5, temp=1.0):
    tree = cKDTree(bank_keys)
    dist, idx = tree.query(query_keys, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = np.exp(-dist / (temp + 1e-10))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    return np.einsum('nk,nkd->nd', weights, bank_values[idx])


def retrieve_register(query_keys, query_f0, bank_keys, bank_f0, bank_values,
                      k=5, temp=1.0, n_bins=3):
    T = len(query_keys)
    mc_dim = bank_values.shape[1]
    result = np.zeros((T, mc_dim), dtype=np.float32)

    f0_edges = np.linspace(np.log(80), np.log(400), n_bins + 1)[1:-1]

    def get_bin(f0_val):
        if f0_val <= 0:
            return -1
        return int(np.digitize(np.log(f0_val), f0_edges))

    bank_bins = np.array([get_bin(f) for f in bank_f0])

    bin_trees = {}
    for b in range(-1, n_bins):
        mask = bank_bins == b
        if mask.sum() == 0:
            continue
        idx_map = np.where(mask)[0]
        tree = cKDTree(bank_keys[idx_map])
        bin_trees[b] = (tree, idx_map)

    fallback_tree = cKDTree(bank_keys) if len(bank_keys) > 0 else None

    for t in range(T):
        qb = get_bin(query_f0[t])
        if qb in bin_trees:
            tree, idx_map = bin_trees[qb]
            kk = min(k, len(idx_map))
        elif fallback_tree is not None:
            tree = fallback_tree
            idx_map = np.arange(len(bank_keys))
            kk = min(k, len(idx_map))
        else:
            continue

        d, ix = tree.query(query_keys[t], k=kk)
        d = np.atleast_1d(d); ix = np.atleast_1d(ix)
        w = np.exp(-d / (temp + 1e-10))
        w = w / (w.sum() + 1e-10)
        result[t] = (bank_values[idx_map[ix]] * w[:, None]).sum(axis=0)

    return result


def main():
    DEVICE = torch.device("cuda")
    print("=== Fixed Retrieval v2 — Key Design Experiment ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(N_PAIRS)
    print(f"Pairs: {len(pairs)}\n")

    configs = [
        ("baseline",         0, False, False, 5, 1.0, False),
        ("ctx1",             1, False, False, 5, 1.0, False),
        ("ctx2",             2, False, False, 5, 1.0, False),
        ("ctx4",             4, False, False, 5, 1.0, False),
        ("ctx2_delta",       2, True,  False, 5, 1.0, False),
        ("ctx2_f0reg",       2, False, True,  5, 1.0, False),
        ("ctx2_delta_f0reg", 2, True,  True,  5, 1.0, False),
        ("ctx4_delta_f0reg", 4, True,  True,  5, 1.0, False),
        ("ctx2_k3",          2, False, False, 3, 1.0, False),
        ("ctx2_k10",         2, False, False, 10, 1.0, False),
        ("ctx2_t05",         2, False, False, 5, 0.5, False),
        ("ctx2_t2",          2, False, False, 5, 2.0, False),
        ("ctx2_regbank",     2, False, False, 5, 1.0, True),
        ("ctx2_delta_regbank", 2, True, True, 5, 1.0, True),
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

        if p["src"] not in speaker_means:
            speaker_means[p["src"]] = compute_speaker_mean(p["src"])
        if p["tgt"] not in speaker_means:
            speaker_means[p["tgt"]] = compute_speaker_mean(p["tgt"])
        src_mean = speaker_means[p["src"]]
        tgt_mean = speaker_means[p["tgt"]]

        bank_mc, bank_f0 = build_bank(p["tgt"], p["text"], n_utts=10)

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for name, ctx, delta, f0reg, k, temp, use_regbank in configs:
                src_keys = build_context_key(mc_s, f0_s, src_mean, ctx, delta, f0reg)
                bank_keys = build_context_key(bank_mc, bank_f0, tgt_mean, ctx, delta, f0reg)

                if use_regbank:
                    mc_pred = retrieve_register(src_keys, f0_s, bank_keys, bank_f0, bank_mc, k=k, temp=temp)
                else:
                    mc_pred = retrieve(src_keys, bank_keys, bank_mc, k=k, temp=temp)

                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])
                sim = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
                results[name].append(sim)

        if (idx+1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] " + " | ".join(
                f"{n}={np.mean(results[n][-5:]):.3f}" for n, *_ in configs[:5]), flush=True)

    print(f"\n{'='*70}")
    print(f"{'config':<25} {'mean':>8} {'std':>8}")
    print(f"{'-'*45}")
    for name, *_ in configs:
        arr = np.array(results[name])
        print(f"{name:<25} {arr.mean():>8.4f} {arr.std():>8.4f}")

    print(f"\n目標: >= 0.42")
    best_name = max(results.keys(), key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_name])
    print(f"最高: {best_name} = {best_score:.4f}")
    if best_score >= 0.42:
        print("→ Go条件クリア!")
    elif best_score >= 0.38:
        print("→ 改善あり、もう一歩")
    else:
        print("→ 改善不十分、C方向も検討必要")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/retrieval_v2.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存: results/retrieval_v2.json")


if __name__ == "__main__":
    main()
