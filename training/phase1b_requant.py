"""
Phase 1b: Residual-chain-preserving token translation oracle

naive swap失敗(0.19≈random)の原因はresidual chain崩壊。
RVQ残差鎖を保ちながらsource coarseを固定し、残りをtargetから再量子化。

z ≈ q1 + q2 + ... + q9  (各q_dは1024-dim、残差は1024-dim空間)

re-quant(d, k):
  depths 0..k-1: sourceの量子化を保持
  residual = z_tgt_aligned - sum(q_src[0:k])
  depths k..8: residualから再量子化

実験:
  1. k sweep (0..9): どのdepthまでsource保持が最適か
  2. 逆方向: target coarse保持 + source rest再量子化
  3. depth 1個ずつablation: target_allからdepth dだけsourceに
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
def encode(dac, wav_44k):
    """wav [T] → z [1024, T_frames]"""
    x = torch.from_numpy(wav_44k).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return dac.encoder(x)  # [1, 1024, T]


@torch.no_grad()
def decode(dac, z):
    """z [1, 1024, T] → wav [T_samples]"""
    return dac.decoder(z).squeeze().cpu().numpy()


@torch.no_grad()
def quantize_full(dac, z):
    """z [1, 1024, T] → list of (q_d [1,1024,T], codes_d [1,T]) per depth"""
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    residual = z.clone()
    q_list = []; codes_list = []
    for d in range(n):
        q_out, _, _, codes_d, _ = quantizers[d](residual)
        q_list.append(q_out)
        codes_list.append(codes_d)
        residual = residual - q_out
    return q_list, codes_list, residual


@torch.no_grad()
def re_quant_source_coarse(dac, z_s, z_t_aligned, k):
    """
    Keep source depths 0..k-1, re-quantize k..N from z_t - sum(q_src[0:k])
    
    Returns: z_q [1, 1024, T]
    """
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks

    q_s_list, _, _ = quantize_full(dac, z_s)

    z_q = torch.zeros_like(z_s)
    for d in range(k):
        z_q = z_q + q_s_list[d]

    if k < n:
        residual = z_t_aligned - z_q
        for d in range(k, n):
            q_out, _, _, _, _ = quantizers[d](residual)
            z_q = z_q + q_out
            residual = residual - q_out

    return z_q


@torch.no_grad()
def re_quant_target_coarse(dac, z_s, z_t_aligned, k):
    """
    Keep target depths 0..k-1, re-quantize k..N from z_s - sum(q_tgt[0:k])
    """
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks

    q_t_list, _, _ = quantize_full(dac, z_t_aligned)

    z_q = torch.zeros_like(z_t_aligned)
    for d in range(k):
        z_q = z_q + q_t_list[d]

    if k < n:
        residual = z_s - z_q
        for d in range(k, n):
            q_out, _, _, _, _ = quantizers[d](residual)
            z_q = z_q + q_out
            residual = residual - q_out

    return z_q


@torch.no_grad()
def single_depth_swap_from_target(dac, z_s, z_t_aligned, swap_depth):
    """
    target_allから depth d だけ source に置換（residual chain保つ）
    """
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks

    q_t_list, _, _ = quantize_full(dac, z_t_aligned)
    q_s_list, _, _ = quantize_full(dac, z_s)

    z_q = torch.zeros_like(z_t_aligned)
    for d in range(n):
        if d == swap_depth:
            z_q = z_q + q_s_list[d]
        else:
            z_q = z_q + q_t_list[d]

    return z_q


def align_latents(z_s, z_t):
    """DTW-align z_t to z_s timeline. z_s, z_t: [1, 1024, T]"""
    z_s_np = z_s.squeeze(0).cpu().numpy().T  # [T_s, 1024]
    z_t_np = z_t.squeeze(0).cpu().numpy().T  # [T_t, 1024]

    dist, path = fastdtw(z_s_np, z_t_np, radius=15)

    T_s = len(z_s_np)
    T_t = len(z_t_np)
    src_to_tgt = np.zeros(T_s, dtype=int)
    for s, t in path:
        if s < T_s:
            src_to_tgt[s] = min(t, T_t - 1)
    for i in range(1, T_s):
        if src_to_tgt[i] == 0:
            src_to_tgt[i] = src_to_tgt[i-1]

    z_t_np_aligned = z_t_np[src_to_tgt].T  # [1024, T_s]
    return torch.from_numpy(z_t_np_aligned).float().unsqueeze(0).to(DEVICE)


def load_wav_44k(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def main():
    print("=== Phase 1b: Residual-Chain-Preserving Re-quant Oracle ===\n")

    dac = load_dac()
    n_cb = dac.quantizer.n_codebooks
    print(f"RVQ: {n_cb} codebooks\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(N_PAIRS)
    print(f"ペア数: {len(pairs)}\n")

    results = defaultdict(list)
    t0 = time.time()

    for idx, p in enumerate(pairs):
        try:
            wav_s = load_wav_44k(p["src_wav"])
            wav_t = load_wav_44k(p["tgt_wav"])
            if len(wav_s) < DAC_SR or len(wav_t) < DAC_SR:
                continue

            z_s = encode(dac, wav_s)
            z_t = encode(dac, wav_t)
            z_t_aligned = align_latents(z_s, z_t)

            wav_tgt_16k = librosa.resample(wav_t.astype(np.float64),
                                           orig_sr=DAC_SR, target_sr=SECS_SR)
            if len(wav_tgt_16k) < 8000: continue

            with torch.no_grad():
                e_tgt = secs_model.encode_batch(
                    torch.from_numpy(wav_tgt_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)
                ).squeeze(0)

                configs = {}

                # Baselines
                configs["target_all"] = z_t_aligned
                configs["source_all"] = z_s

                # Re-quant: source coarse k depths + target rest re-quantized
                for k in range(n_cb + 1):
                    z_q = re_quant_source_coarse(dac, z_s, z_t_aligned, k)
                    configs[f"src_k{k}"] = z_q

                # Re-quant: target coarse k depths + source rest re-quantized
                for k in [1, 3, 5]:
                    if k <= n_cb:
                        z_q = re_quant_target_coarse(dac, z_s, z_t_aligned, k)
                        configs[f"tgt_k{k}"] = z_q

                # Single depth ablation: target_all - depth d + source depth d
                for d in range(n_cb):
                    z_q = single_depth_swap_from_target(dac, z_s, z_t_aligned, d)
                    configs[f"tgt_minus_d{d}"] = z_q

                for name, z_q in configs.items():
                    audio_44k = decode(dac, z_q)
                    if len(audio_44k) < DAC_SR * 0.5: continue
                    audio_16k = librosa.resample(audio_44k.astype(np.float64),
                                                orig_sr=DAC_SR, target_sr=SECS_SR)
                    if len(audio_16k) < 8000: continue
                    e = secs_model.encode_batch(
                        torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)
                    ).squeeze(0)
                    sim = F.cosine_similarity(e_tgt, e, dim=-1).item()
                    results[name].append(sim)

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx+1) % 20 == 0:
            elapsed = time.time() - t0
            speed = (idx+1) / elapsed
            eta = (len(pairs) - idx - 1) / speed
            ta = np.mean(results.get("target_all", [0])[-20:]) if results.get("target_all") else 0
            s3 = np.mean(results.get("src_k3", [0])[-20:]) if results.get("src_k3") else 0
            t3 = np.mean(results.get("tgt_k3", [0])[-20:]) if results.get("tgt_k3") else 0
            print(f"  [{idx+1}/{len(pairs)}] tgt_all={ta:.3f} "
                  f"src_k3={s3:.3f} tgt_k3={t3:.3f} "
                  f"| {speed:.1f}pair/s ETA {eta:.0f}s", flush=True)

    # =========================================
    # Results
    # =========================================
    print(f"\n{'='*75}")
    print(f"{'config':<25} {'mean':>8} {'std':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print(f"{'-'*60}")

    all_names = ["target_all", "source_all"]
    all_names += [f"src_k{k}" for k in range(n_cb + 1)]
    all_names += [f"tgt_k{k}" for k in [1, 3, 5]]
    all_names += [f"tgt_minus_d{d}" for d in range(n_cb)]

    for name in all_names:
        if name not in results or len(results[name]) == 0:
            continue
        arr = np.array(results[name])
        n = len(arr)
        boot = [arr[np.random.choice(n, n, replace=True)].mean() for _ in range(500)]
        boot = np.array(boot)
        print(f"{name:<25} {arr.mean():>8.4f} {arr.std():>8.4f} "
              f"{np.percentile(boot, 2.5):>8.4f} {np.percentile(boot, 97.5):>8.4f}")

    print(f"\n--- 判定 ---")
    ta = np.mean(results.get("target_all", [0]))
    s3 = np.mean(results.get("src_k3", [0]))
    s5 = np.mean(results.get("src_k5", [0]))
    best_src_k = max(range(n_cb+1), key=lambda k: np.mean(results.get(f"src_k{k}", [0])))
    best_src_v = np.mean(results.get(f"src_k{best_src_k}", [0]))

    print(f"target_all (上限):       {ta:.4f}")
    print(f"best src_k (k={best_src_k}): {best_src_v:.4f}")
    print(f"WORLD ceiling:           0.365")

    if best_src_v >= 0.55:
        print("→ CONCEPT v2 強く継続 (>= 0.55)")
    elif best_src_v >= 0.45:
        print("→ Phase 2 継続 (>= 0.45)")
    elif best_src_v > 0.365:
        print("→ WORLD ceiling超え、改善の余地あり")
    else:
        print("→ frozen DAC token translation は厳しい")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                  "scores": [float(x) for x in v]}
           for name, v in results.items()}
    with open("results/phase1b_requant.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存: results/phase1b_requant.json")


if __name__ == "__main__":
    main()
