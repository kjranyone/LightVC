"""
Phase 1: RVQ depth分析 + token組み替えoracle

1. DAC encode → RVQ tokens (9 codebooks × T frames)
2. depth別話者識別力 (F-ratio, classifier accuracy)
3. depth swap oracle:
   - target_all: 全depth置換 (上限)
   - src_coarse(0-2) + tgt_mid_fine(3-8)
   - src_coarse(0-2) + tgt_mid(3-5) + src_fine(6-8)
   - src_coarse_mid(0-5) + tgt_fine(6-8)
   - random_mix (negative control)
4. 200ペア評価、bootstrap CI

Go条件:
  same-text swap > 0.365 (WORLD ceiling)
  かつ mid識別力 > coarse
  かつ random_mixが明確に破綻
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
DAC_SR = 44100
SECS_SR = 16000
VCTK_WAV = Path("../data/vctk_200")
N_PAIRS = 200


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


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


@torch.no_grad()
def encode_to_tokens(dac, wav_44k):
    """wav [T] → (codes [9, T_frames], z [1024, T_frames])"""
    x = torch.from_numpy(wav_44k).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    z = dac.encoder(x)  # [1, 1024, T]
    result = dac.quantizer(z)
    codes = result[1]  # [1, 9, T] int64
    return (codes.squeeze(0).cpu().numpy(),
            z.squeeze(0).cpu().numpy())


@torch.no_grad()
def decode_from_codes(dac, codes):
    """codes [9, T] → wav [T_samples]"""
    codes_t = torch.from_numpy(codes).long().unsqueeze(0).to(DEVICE)
    z_q, _, _ = dac.quantizer.from_codes(codes_t)
    audio = dac.decoder(z_q)
    return audio.squeeze().cpu().numpy()


def swap_codes(codes_s, codes_t, src_depths, tgt_depths):
    """Create mixed codes from source and target"""
    mixed = codes_s.copy()
    for d in tgt_depths:
        mixed[d] = codes_t[d]
    return mixed


def align_codes(codes_s, codes_t, z_s, z_t):
    """DTW-align target codes to source timeline"""
    dist, path = fastdtw(z_s.T, z_t.T, radius=20)

    T_s = len(codes_s.T) if codes_s.ndim == 2 else codes_s.shape[1]
    T_t = codes_t.shape[1]

    src_to_tgt = np.zeros(T_s, dtype=int)
    for s, t in path:
        if s < T_s:
            src_to_tgt[s] = min(t, T_t - 1)
    for i in range(1, T_s):
        if src_to_tgt[i] == 0:
            src_to_tgt[i] = src_to_tgt[i-1]

    codes_t_aligned = np.zeros_like(codes_s)
    for s in range(T_s):
        codes_t_aligned[:, s] = codes_t[:, src_to_tgt[s]]

    return codes_t_aligned


def load_wav_44k(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def main():
    print("=== Phase 1: RVQ Depth Analysis + Token Swap Oracle ===\n")

    dac = load_dac()

    # Check quantizer structure
    quantizer = dac.quantizer
    n_cb = quantizer.n_codebooks
    cb_shape = quantizer.quantizers[0].codebook.weight.shape
    print(f"RVQ: {n_cb} codebooks, codebook shape={cb_shape}")

    # =========================================
    # Part 1: depth別話者識別力 (20 speakers × 5 utts)
    # =========================================
    print("\n--- Part 1: depth別話者識別力 ---")

    spk_dirs = sorted([d for d in VCTK_WAV.iterdir() if d.is_dir()])[:20]
    depth_tokens = defaultdict(lambda: defaultdict(list))  # {depth: {spk: [tokens]}}

    for si, spk_dir in enumerate(spk_dirs):
        wavs = sorted(spk_dir.glob("*.wav"))[:5]
        for w in wavs:
            wav_44k = load_wav_44k(w)
            if len(wav_44k) < DAC_SR * 0.5: continue
            codes, _ = encode_to_tokens(dac, wav_44k)
            for d in range(n_cb):
                depth_tokens[d][spk_dir.name].append(codes[d].flatten())

    # F-ratio per depth
    print(f"\n{'depth':>6} {'F-ratio':>10} {'n_unique':>10}")
    print("-" * 30)
    fratios = []
    for d in range(n_cb):
        all_tokens = []
        labels = []
        for spk in sorted(depth_tokens[d].keys()):
            tokens = np.concatenate(depth_tokens[d][spk])
            all_tokens.append(tokens)
            labels.extend([spk] * len(tokens))
        all_tokens = np.concatenate(all_tokens)
        labels = np.array(labels)

        speaker_means = {}
        for spk in sorted(depth_tokens[d].keys()):
            mask = labels == spk
            speaker_means[spk] = all_tokens[mask].mean()

        grand_mean = all_tokens.mean()
        ss_between = sum(len(depth_tokens[d][spk]) * (speaker_means[spk] - grand_mean)**2
                        for spk in speaker_means)
        ss_within = sum(((all_tokens[labels == spk] - speaker_means[spk])**2).sum()
                       for spk in speaker_means)
        ss_total = ((all_tokens - grand_mean)**2).sum()
        f_ratio = ss_between / (ss_total + 1e-10)
        n_unique = len(np.unique(all_tokens))
        fratios.append(f_ratio)
        print(f"{d:>6} {f_ratio:>10.6f} {n_unique:>10}")

    print(f"\ncoarse (0-2) avg F-ratio: {np.mean(fratios[:3]):.6f}")
    print(f"mid    (3-5) avg F-ratio: {np.mean(fratios[3:6]):.6f}")
    print(f"fine   (6-8) avg F-ratio: {np.mean(fratios[6:]):.6f}")

    # =========================================
    # Part 2: Token Swap Oracle (200 pairs)
    # =========================================
    print("\n--- Part 2: Token Swap Oracle ---\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(N_PAIRS)
    print(f"ペア数: {len(pairs)}")

    swap_configs = {
        "target_all":          list(range(9)),
        "src_coarse_tgt_rest": [3, 4, 5, 6, 7, 8],
        "src_coarse_mid_tgt_fine": [6, 7, 8],
        "src_coarse_tgt_mid":  [3, 4, 5],
        "tgt_coarse_src_rest": [0, 1, 2],
        "random_half":         "random",
    }

    results = defaultdict(list)
    t0 = time.time()

    for idx, p in enumerate(pairs):
        try:
            wav_s = load_wav_44k(p["src_wav"])
            wav_t = load_wav_44k(p["tgt_wav"])
            if len(wav_s) < DAC_SR or len(wav_t) < DAC_SR:
                continue

            codes_s, z_s = encode_to_tokens(dac, wav_s)
            codes_t, z_t = encode_to_tokens(dac, wav_t)

            codes_t_aligned = align_codes(codes_s, codes_t, z_s, z_t)

            # SECS reference: target wav at 16kHz
            wav_tgt_16k = librosa.resample(wav_t.astype(np.float64), orig_sr=DAC_SR, target_sr=SECS_SR)

            with torch.no_grad():
                if len(wav_tgt_16k) < 8000: continue
                e_tgt = secs_model.encode_batch(
                    torch.from_numpy(wav_tgt_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)
                ).squeeze(0)

                for name, tgt_depths in swap_configs.items():
                    if tgt_depths == "random":
                        tgt_depths = np.random.choice(9, size=4, replace=False).tolist()
                        mixed = codes_s.copy()
                        for d in tgt_depths:
                            mixed[d] = codes_t_aligned[d]
                    else:
                        mixed = swap_codes(codes_s, codes_t_aligned, list(range(9)), tgt_depths)

                    z_q_mixed = None  # not needed, decode directly from codes
                    audio_44k = decode_from_codes(dac, mixed)

                    if len(audio_44k) < DAC_SR * 0.5: continue
                    audio_16k = librosa.resample(audio_44k.astype(np.float64),
                                                orig_sr=DAC_SR, target_sr=SECS_SR)
                    if len(audio_16k) < 8000: continue

                    e_out = secs_model.encode_batch(
                        torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)
                    ).squeeze(0)
                    sim = F.cosine_similarity(e_tgt, e_out, dim=-1).item()
                    results[name].append(sim)

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx+1) % 20 == 0:
            elapsed = time.time() - t0
            speed = (idx+1) / elapsed
            eta = (len(pairs) - idx - 1) / speed
            ta = np.mean(results.get("target_all", [0])[-20:]) if results.get("target_all") else 0
            sc = np.mean(results.get("src_coarse_tgt_rest", [0])[-20:]) if results.get("src_coarse_tgt_rest") else 0
            print(f"  [{idx+1}/{len(pairs)}] tgt_all={ta:.3f} "
                  f"swap={sc:.3f} | {speed:.1f}pair/s ETA {eta:.0f}s", flush=True)

    # =========================================
    # Results
    # =========================================
    print(f"\n{'='*70}")
    print(f"{'config':<30} {'mean':>8} {'std':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print(f"{'-'*65}")

    for name in swap_configs:
        if name not in results or len(results[name]) == 0:
            print(f"{name:<30}  (no data)")
            continue
        arr = np.array(results[name])
        n = len(arr)
        boot = []
        for _ in range(1000):
            bi = np.random.choice(n, n, replace=True)
            boot.append(arr[bi].mean())
        boot = np.array(boot)
        m = arr.mean()
        lo = np.percentile(boot, 2.5)
        hi = np.percentile(boot, 97.5)
        print(f"{name:<30} {m:>8.4f} {arr.std():>8.4f} {lo:>8.4f} {hi:>8.4f}")

    print(f"\n--- 判定 ---")
    print(f"WORLD ceiling: 0.365")
    ta = np.mean(results.get("target_all", [0]))
    sc = np.mean(results.get("src_coarse_tgt_rest", [0]))
    rh = np.mean(results.get("random_half", [0]))

    mid_f = np.mean(fratios[3:6])
    coarse_f = np.mean(fratios[:3])

    print(f"target_all (上限):          {ta:.4f}")
    print(f"src_coarse + tgt_rest:      {sc:.4f}")
    print(f"random_half (neg control):  {rh:.4f}")
    print(f"mid F-ratio:                {mid_f:.6f}")
    print(f"coarse F-ratio:             {coarse_f:.6f}")

    go = True
    if sc > 0.365:
        print("✓ swap > WORLD ceiling (0.365)")
    else:
        print("✗ swap <= WORLD ceiling")
        go = False

    if mid_f > coarse_f:
        print("✓ mid識別力 > coarse")
    else:
        print("✗ mid識別力 <= coarse")
        go = False

    if rh < sc - 0.05:
        print("✓ random_mix が明確に破綻")
    else:
        print("✗ random_mix が破綻していない")
        go = False

    if go:
        print("\n→ Phase 1 Go条件クリア! CONCEPT v2 継続")
    else:
        print("\n→ Phase 1 Go条件未達。要再検討")

    out = {
        name: {"mean": float(np.mean(v)), "std": float(np.std(v)),
               "scores": [float(x) for x in v]}
        for name, v in results.items()
    }
    out["fratios"] = [float(f) for f in fratios]
    with open("results/phase1_rvq_swap.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存: results/phase1_rvq_swap.json")


if __name__ == "__main__":
    main()
