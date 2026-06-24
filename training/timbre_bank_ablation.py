"""
Timbre bank oracle ablation: bank size, k, distance weighting
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


def build_bank(tgt_spk, exclude_text, n_utts):
    spk_dir = MC_CACHE / tgt_spk
    bank_files = sorted(spk_dir.glob("*.npz"))
    bank_files = [f for f in bank_files if exclude_text not in f.name][:n_utts]
    bank_mc = []
    for f in bank_files:
        d = np.load(f)
        bank_mc.append(d["mc"])
    return np.concatenate(bank_mc, axis=0)


def retrieve_weighted(query_keys, bank_keys, bank_values, k=10, temperature=1.0):
    tree = cKDTree(bank_keys)
    dist, idx = tree.query(query_keys, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = np.exp(-dist / (temperature + 1e-10))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    retrieved = np.einsum('nk,nkd->nd', weights, bank_values[idx])
    return retrieved


def compute_speaker_mean(spk_id):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:30]
    all_mc = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
    return np.concatenate(all_mc, axis=0).mean(axis=0)


def main():
    DEVICE = torch.device("cuda")

    print("=== Timbre Bank Oracle Ablation ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(N_PAIRS)
    print(f"ペア数: {len(pairs)}\n")

    ablation_configs = [
        ("ref5_k5",   5, 5, 1.0),
        ("ref10_k5",  10, 5, 1.0),
        ("ref10_k10", 10, 10, 1.0),
        ("ref20_k5",  20, 5, 1.0),
        ("ref20_k10", 20, 10, 1.0),
        ("ref20_k20", 20, 20, 1.0),
        ("ref50_k10", 50, 10, 1.0),
        ("ref10_k10_t05", 10, 10, 0.5),
        ("ref10_k10_t2",  10, 10, 2.0),
    ]

    results = defaultdict(list)

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])

        mc_s = feat_s["mc"]
        f0_s = feat_s["f0"]
        ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]
        f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        src_mean = compute_speaker_mean(p["src"])
        tgt_mean = compute_speaker_mean(p["tgt"])
        src_norm = mc_s - src_mean

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        # pre-build banks for all ref sizes
        banks = {}
        for n_ref in [5, 10, 20, 50]:
            bank_mc = build_bank(p["tgt"], p["text"], n_ref)
            bank_norm = bank_mc - tgt_mean
            banks[n_ref] = (bank_mc, bank_norm)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for name, n_ref, k, temp in ablation_configs:
                bank_mc, bank_norm = banks[n_ref]
                mc_pred = retrieve_weighted(src_norm, bank_norm, bank_mc, k=k, temperature=temp)
                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])
                e_out = emb(wav_out)
                sim = F.cosine_similarity(e_tgt, e_out, dim=-1).item()
                results[name].append(sim)

        status = " | ".join(f"{n}={results[n][-1]:.3f}" for n, _, _, _ in ablation_configs[:4])
        print(f"  [{idx+1}/{len(pairs)}] {p['src']}→{p['tgt']}: {status}", flush=True)

    print(f"\n=== Ablation結果 ===")
    print(f"{'config':<20} {'mean':>8} {'std':>8}")
    print("-" * 40)
    for name, _, _, _ in ablation_configs:
        arr = np.array(results[name])
        print(f"{name:<20} {arr.mean():>8.4f} {arr.std():>8.4f}")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/timbre_bank_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存: results/timbre_bank_ablation.json")


if __name__ == "__main__":
    main()
