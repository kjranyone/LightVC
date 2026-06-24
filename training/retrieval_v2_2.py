"""
Retrieval v2.2 — ctx8_invfratio の微調整
k, temperature, bank size, context size の精密sweep
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
    all_mc = [np.load(f)["mc"] for f in files]
    return np.concatenate(all_mc, axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    all_mc = [np.load(f)["mc"] for f in files]
    all_f0 = [np.load(f)["f0"] for f in files]
    return (np.concatenate(all_mc, axis=0).astype(np.float32),
            np.concatenate(all_f0, axis=0).astype(np.float32))


def build_context_key(mc, spk_mean, ctx, weights):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


def retrieve(query_keys, bank_keys, bank_values, k=5, temp=1.0):
    tree = cKDTree(bank_keys)
    dist, idx = tree.query(query_keys, k=k)
    if k == 1:
        dist = dist[:, None]; idx = idx[:, None]
    weights = np.exp(-dist / (temp + 1e-10))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    return np.einsum('nk,nkd->nd', weights, bank_values[idx])


def main():
    DEVICE = torch.device("cuda")
    print("=== Retrieval v2.2 — Fine-tuning ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    fratio = np.load("data/mc_fratio_weights.npy")
    inv_fratio = (1.0 / (fratio + 1e-6)).astype(np.float32)
    inv_fratio = inv_fratio / inv_fratio.mean()
    sqrt_inv = np.sqrt(inv_fratio).astype(np.float32)

    pairs = find_pairs(N_PAIRS)
    print(f"Pairs: {len(pairs)}\n")

    configs = [
        ("ctx8_ifr_k3",    8, inv_fratio,  3, 1.0, 10),
        ("ctx8_ifr_k5",    8, inv_fratio,  5, 1.0, 10),
        ("ctx8_ifr_k10",   8, inv_fratio, 10, 1.0, 10),
        ("ctx8_ifr_k20",   8, inv_fratio, 20, 1.0, 10),
        ("ctx8_ifr_k5_t05",8, inv_fratio,  5, 0.5, 10),
        ("ctx8_ifr_k5_t2", 8, inv_fratio,  5, 2.0, 10),
        ("ctx6_ifr_k5",    6, inv_fratio,  5, 1.0, 10),
        ("ctx10_ifr_k5",  10, inv_fratio,  5, 1.0, 10),
        ("ctx8_sqrtifr_k5",8, sqrt_inv,    5, 1.0, 10),
        ("ctx8_ifr_k5_b20",8, inv_fratio,  5, 1.0, 20),
        ("ctx8_none_k5",   8, None,        5, 1.0, 10),
        ("ctx8_ifr_k10_t2",8, inv_fratio, 10, 2.0, 10),
        ("ctx8_ifr_k20_t2",8, inv_fratio, 20, 2.0, 10),
    ]

    results = defaultdict(list)
    speaker_means = {}

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])
        mc_s = feat_s["mc"]; f0_s = feat_s["f0"]; ap_s = feat_s["ap"]
        f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        for spk in [p["src"], p["tgt"]]:
            if spk not in speaker_means:
                speaker_means[spk] = compute_speaker_mean(spk)
        src_mean = speaker_means[p["src"]]
        tgt_mean = speaker_means[p["tgt"]]

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for name, ctx, w, k, temp, n_bank in configs:
                bank_mc, bank_f0 = build_bank(p["tgt"], p["text"], n_utts=n_bank)

                src_keys = build_context_key(mc_s, src_mean, ctx, w)
                bank_keys = build_context_key(bank_mc, tgt_mean, ctx, w)
                mc_pred = retrieve(src_keys, bank_keys, bank_mc, k=k, temp=temp)
                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])
                sim = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
                results[name].append(sim)

        if (idx+1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] " + " | ".join(
                f"{n}={np.mean(results[n][-5:]):.3f}" for n, *_ in configs[:5]), flush=True)

    print(f"\n{'='*60}")
    print(f"{'config':<22} {'mean':>8} {'std':>8}")
    print(f"{'-'*42}")
    for name, *_ in configs:
        arr = np.array(results[name])
        print(f"{name:<22} {arr.mean():>8.4f} {arr.std():>8.4f}")

    best_name = max(results.keys(), key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_name])
    print(f"\n最高: {best_name} = {best_score:.4f}")
    if best_score >= 0.42:
        print("→ Go条件クリア!")
    elif best_score >= 0.40:
        print("→ 0.42にほぼ到達")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/retrieval_v2_2.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
