"""DTW oracle再計算（200ペア、truncation bug修正）"""
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
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000; FRAME_PERIOD = 5.0; FFTL = 2048; ALPHA = 0.410
MC_ORDER = 24; MC_DIM = 25
VCTK_WAV = Path("../data/vctk_200")
N_PAIRS = 200


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


def find_pairs(n=200):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb: continue
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                if len(pairs) >= n: return pairs
    return pairs


def main():
    DEVICE = torch.device("cuda")
    print("=== DTW Oracle 再計算 (200ペア) ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(N_PAIRS)
    print(f"ペア数: {len(pairs)}\n")

    scores = []
    t0 = time.time()

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])
        mc_s = feat_s["mc"]; f0_s = feat_s["f0"]; ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]; f0_t = feat_t["f0"]
        T = min(len(mc_s), len(mc_t))

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        dist_dtw, path_dtw = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path_dtw:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]

        wav_dtw = synth(f0_shifted[:T], mc_t_aligned[:T].astype(np.float32), ap_s[:T])

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR:
            wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            tgt_t = torch.from_numpy(wav_tgt.astype(np.float32)).unsqueeze(0).to(DEVICE)
            if tgt_t.shape[1] < 8000: continue
            e_tgt = secs_model.encode_batch(tgt_t).squeeze(0)
            wav_d = torch.from_numpy(wav_dtw.astype(np.float32)).unsqueeze(0).to(DEVICE)
            if wav_d.shape[1] < 8000: continue
            e_d = secs_model.encode_batch(wav_d).squeeze(0)
            sim = F.cosine_similarity(e_tgt, e_d, dim=-1).item()
            scores.append(sim)

        if (idx+1) % 20 == 0:
            elapsed = time.time() - t0
            speed = (idx+1) / elapsed
            eta = (len(pairs) - idx - 1) / speed
            print(f"  [{idx+1}/{len(pairs)}] dtw_oracle={np.mean(scores[-20:]):.3f} "
                  f"| {speed:.1f}pair/s ETA {eta:.0f}s", flush=True)

    arr = np.array(scores)
    n = len(arr)
    boot_means = []
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        boot_means.append(arr[idx].mean())
    boot_means = np.array(boot_means)

    print(f"\n{'='*50}")
    print(f"DTW Oracle (200ペア)")
    print(f"  mean:    {arr.mean():.4f}")
    print(f"  std:     {arr.std():.4f}")
    print(f"  95% CI:  [{np.percentile(boot_means, 2.5):.4f}, {np.percentile(boot_means, 97.5):.4f}]")
    print(f"  CI幅:    {np.percentile(boot_means, 97.5) - np.percentile(boot_means, 2.5):.4f}")
    print(f"  n:       {n}")

    print(f"\n--- 訂正された全体像 ---")
    print(f"  ctx8_b10 retrieval:     0.328 (±0.031)")
    print(f"  oracle rerank b10:      0.377 (±0.029)")
    print(f"  oracle rerank b100:     0.400 (±0.029)")
    print(f"  DTW Oracle:             {arr.mean():.3f} (±{np.percentile(boot_means, 97.5) - np.percentile(boot_means, 2.5):.3f})")

    with open("results/dtw_oracle_200.json", "w") as f:
        json.dump({
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lo": float(np.percentile(boot_means, 2.5)),
            "ci_hi": float(np.percentile(boot_means, 97.5)),
            "raw_scores": scores,
        }, f, indent=2)


if __name__ == "__main__":
    main()
