"""
Timbre bank retrieval oracle (学習なし)

target話者の参照発話からmcep bankを構築し、
source frameからcontent/register近傍のtarget mcepを retrieves して合成。

複数のkey typeを比較:
  1. raw mcep: そのままL2距離
  2. speaker-normalized: mc - speaker_mean (話者DC除去)
  3. register-only: F0_bin + energy_bin + VUV (純register)

SECS >= 0.35-0.45 なら attention model化の価値あり。
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
N_REF_UTTS = 10


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


def build_bank(tgt_spk, exclude_text, n_utts=N_REF_UTTS):
    spk_dir = MC_CACHE / tgt_spk
    bank_files = sorted(spk_dir.glob("*.npz"))
    bank_files = [f for f in bank_files if exclude_text not in f.name][:n_utts]
    bank_mc = []
    bank_f0 = []
    for f in bank_files:
        d = np.load(f)
        bank_mc.append(d["mc"])
        bank_f0.append(d["f0"])
    bank_mc = np.concatenate(bank_mc, axis=0)
    bank_f0 = np.concatenate(bank_f0, axis=0)
    return bank_mc, bank_f0


def retrieve_nn(query_keys, bank_keys, bank_values):
    from scipy.spatial import cKDTree
    tree = cKDTree(bank_keys)
    _, idx = tree.query(query_keys, k=1)
    return bank_values[idx]


def retrieve_softavg(query_keys, bank_keys, bank_values, k=5):
    from scipy.spatial import cKDTree
    tree = cKDTree(bank_keys)
    _, idx = tree.query(query_keys, k=k)
    return bank_values[idx].mean(axis=1)


def compute_speaker_mean(spk_id):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:30]
    all_mc = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
    return np.concatenate(all_mc, axis=0).mean(axis=0)


def register_key(mc, f0):
    f0_voiced = f0[f0 > 0]
    f0_mean = np.mean(np.log(f0_voiced)) if len(f0_voiced) > 0 else np.log(100)
    f0_bin = int(np.clip((f0_mean - np.log(80)) / (np.log(400) - np.log(80)) * 8, 0, 7))
    energy = mc[:, 0]
    e_bin = np.clip((energy - energy.min()) / (energy.max() - energy.min() + 1e-6) * 2, 0, 1).astype(int)
    vuv = (f0 > 0).astype(int)
    keys = np.stack([np.full(len(mc), f0_bin), e_bin, vuv], axis=1).astype(float)
    return keys


def main():
    DEVICE = torch.device("cuda")

    print("=== Timbre Bank Retrieval Oracle ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(N_PAIRS)
    print(f"ペア数: {len(pairs)}\n")

    results = defaultdict(list)

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])

        mc_s = feat_s["mc"]
        f0_s = feat_s["f0"]
        ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]
        f0_t = feat_t["f0"]
        ap_t = feat_t["ap"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        tgt_text = p["text"]
        bank_mc, bank_f0 = build_bank(p["tgt"], tgt_text)

        src_mean = compute_speaker_mean(p["src"])
        tgt_mean = compute_speaker_mean(p["tgt"])

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        configs = {}

        # 1. raw mcep retrieval
        configs["raw_nn"] = retrieve_nn(mc_s, bank_mc, bank_mc)

        # 2. speaker-normalized retrieval
        src_norm = mc_s - src_mean
        bank_norm = bank_mc - tgt_mean
        configs["norm_nn"] = retrieve_nn(src_norm, bank_norm, bank_mc)
        configs["norm_soft5"] = retrieve_softavg(src_norm, bank_norm, bank_mc, k=5)

        # 3. DTW oracle (upper bound for same-text)
        from fastdtw import fastdtw
        dist, path = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]
        configs["dtw_oracle"] = mc_t_aligned[:T]

        # 4. target mean (constant)
        configs["tgt_mean_const"] = np.tile(tgt_mean, (T, 1)).astype(np.float32)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for name, mc_pred in configs.items():
                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])
                e_out = emb(wav_out)
                sim = F.cosine_similarity(e_tgt, e_out, dim=-1).item()
                results[name].append(sim)

        status = " | ".join(f"{n}={results[n][-1]:.3f}" for n in ["dtw_oracle", "raw_nn", "norm_nn", "norm_soft5", "tgt_mean_const"])
        print(f"  [{idx+1}/{len(pairs)}] {p['src']}→{p['tgt']}: {status}", flush=True)

    print(f"\n=== 結果 (timbre bank retrieval oracle) ===")
    print(f"{'method':<20} {'mean':>8} {'std':>8} {'vs oracle':>10}")
    print("-" * 50)
    oracle_mean = np.mean(results["dtw_oracle"])
    for name in ["dtw_oracle", "raw_nn", "norm_nn", "norm_soft5", "tgt_mean_const"]:
        arr = np.array(results[name])
        ratio = arr.mean() / oracle_mean if oracle_mean > 0 else 0
        print(f"{name:<20} {arr.mean():>8.4f} {arr.std():>8.4f} {ratio:>10.1%}")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v)), "scores": [float(x) for x in v]}
           for name, v in results.items()}
    with open("results/timbre_bank_oracle.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存: results/timbre_bank_oracle.json")


if __name__ == "__main__":
    main()
