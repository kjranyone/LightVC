"""
T0: Noisy Latent Tolerance Sweep (DIAGNOSTIC)

Characterizes where RVQ cascade sensitivity lives by decoding noised target
latents through three paths:

  Path A: direct decode (skip RVQ entirely)
  Path B: hard RVQ re-quantize with fixed source q0
  Path C: soft RVQ (temperature sweep)

At sigma=0:
  Path B reproduces the src_K1 oracle (SECS ~0.686).
  Path A approximates continuous target decode (SECS ~0.589).
  Path C with small tau should approach Path B.

Flip rate is measured per-depth using the q0_s-fixed residual chain.
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
DAC_SR = 44100
SECS_SR = 16000
DATA_DIR = Path("../data/phase3/eval")
RESULTS_DIR = Path("../results")

SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
TAUS = [0.1, 0.5, 1.0, 2.0, 5.0]


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


def load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def resample_16k(audio_44k):
    B = audio_44k.shape[0]
    flat = audio_44k.reshape(B, 1, -1)
    out = F.interpolate(flat, scale_factor=SECS_SR / DAC_SR,
                        mode="linear", align_corners=False)
    return out.squeeze(1)


@torch.no_grad()
def compute_secs(ecapa, audio_16k, timbre):
    if audio_16k.shape[-1] < 8000:
        return float("nan")
    emb = ecapa.encode_batch(audio_16k)
    sim = F.cosine_similarity(emb.squeeze(0), timbre.unsqueeze(0), dim=-1)
    return sim.item()


@torch.no_grad()
def hard_rvq_requantize(dac, q0_s, z_input):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks

    z_q = q0_s.clone()
    residual = z_input - q0_s
    codes_all = []

    for d in range(1, n):
        q_out, _, _, codes_d, _ = qs[d](residual)
        z_q = z_q + q_out
        residual = residual - q_out
        codes_all.append(codes_d.squeeze(0))

    codes = torch.stack(codes_all)
    return z_q, codes


@torch.no_grad()
def soft_rvq_requantize(dac, q0_s, z_input, tau):
    z_q_sum = q0_s.clone()
    residual = z_input - q0_s

    for d in range(1, 9):
        quantizer = dac.quantizer.quantizers[d]

        z_e = quantizer.in_proj(residual)
        cb = quantizer.codebook.weight

        z_e_t = z_e.transpose(1, 2)
        dist = torch.cdist(z_e_t, cb.unsqueeze(0)).pow(2)

        weights = F.softmax(-dist / tau, dim=-1)

        z_q_soft = (weights @ cb).transpose(1, 2)
        z_q_1024 = quantizer.out_proj(z_q_soft)

        residual = residual - z_q_1024
        z_q_sum = z_q_sum + z_q_1024

    return z_q_sum


def run_sweep(args):
    print("=== T0: Noisy Latent Tolerance Sweep ===\n")

    torch.manual_seed(args.seed)

    dac = load_dac()
    print("DAC loaded")

    ecapa = load_ecapa()
    print("ECAPA loaded")

    files = sorted(DATA_DIR.glob("*.pt"))[: args.n_pairs]
    print(f"Pairs: {len(files)}")
    print(f"Sigmas: {SIGMAS}")
    print(f"Taus:   {TAUS}\n")

    secs_A = {s: [] for s in SIGMAS}
    secs_B = {s: [] for s in SIGMAS}
    secs_C = {(s, t): [] for s in SIGMAS for t in TAUS}
    flip = {s: [[] for _ in range(8)] for s in SIGMAS}

    t0 = time.time()

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu")
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        q0_s = d["q0_s"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().to(DEVICE)

        z_norm = z_t.norm().item()

        with torch.no_grad():
            _, codes_clean = hard_rvq_requantize(dac, q0_s, z_t)

        for sigma in SIGMAS:
            if sigma == 0.0:
                z_noisy = z_t.clone()
            else:
                eps = torch.randn_like(z_t)
                eps = eps * (z_norm / (eps.norm().item() + 1e-8))
                z_noisy = z_t + sigma * eps

            with torch.no_grad():
                audio_A = dac.decoder(z_noisy)
                secs_A[sigma].append(
                    compute_secs(ecapa, resample_16k(audio_A.squeeze(1)), timbre)
                )

                z_q_B, codes_noisy = hard_rvq_requantize(dac, q0_s, z_noisy)
                audio_B = dac.decoder(z_q_B)
                secs_B[sigma].append(
                    compute_secs(ecapa, resample_16k(audio_B.squeeze(1)), timbre)
                )

                for depth in range(8):
                    fr = (codes_clean[depth] != codes_noisy[depth]).float().mean().item()
                    flip[sigma][depth].append(fr)

                for tau in TAUS:
                    z_q_C = soft_rvq_requantize(dac, q0_s, z_noisy, tau)
                    audio_C = dac.decoder(z_q_C)
                    secs_C[(sigma, tau)].append(
                        compute_secs(ecapa, resample_16k(audio_C.squeeze(1)), timbre)
                    )

        elapsed = time.time() - t0
        eta = elapsed / (pi + 1) * (len(files) - pi - 1)
        print(
            f"  [{pi+1:>3}/{len(files)}] "
            f"A@0={secs_A[0.0][-1]:.3f} "
            f"B@0={secs_B[0.0][-1]:.3f} "
            f"B@0.05={secs_B[0.05][-1]:.3f} "
            f"C@0.05/t1={secs_C[(0.05, 1.0)][-1]:.3f} "
            f"[{elapsed:.0f}s, ETA {eta:.0f}s]",
            flush=True,
        )

    results = {
        "n_pairs": len(files),
        "sigmas": SIGMAS,
        "taus": TAUS,
        "path_A_secs": {
            str(s): {
                "mean": float(np.nanmean(v)),
                "std": float(np.nanstd(v)),
                "n_valid": int(np.sum(~np.isnan(v))),
            }
            for s, v in secs_A.items()
        },
        "path_B_secs": {
            str(s): {
                "mean": float(np.nanmean(v)),
                "std": float(np.nanstd(v)),
                "n_valid": int(np.sum(~np.isnan(v))),
            }
            for s, v in secs_B.items()
        },
        "path_C_secs": {
            f"sigma={s}_tau={t}": {
                "mean": float(np.nanmean(v)),
                "std": float(np.nanstd(v)),
                "n_valid": int(np.sum(~np.isnan(v))),
            }
            for (s, t), v in secs_C.items()
        },
        "flip_rate": {
            str(s): {str(d): float(np.mean(v)) for d, v in enumerate(per_d)}
            for s, per_d in flip.items()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"smoke{args.n_pairs}" if args.n_pairs < 200 else "full200"
    out_path = RESULTS_DIR / f"phase3b_tolerance_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*100}")
    print(f"{'σ':>8}  {'Path A':>8}  {'Path B':>8}  {'flip d0':>8}  {'flip d1':>8}  |  {'C τ.1':>8}  {'C τ.5':>8}  {'C τ1.0':>8}  {'C τ2.0':>8}  {'C τ5.0':>8}")
    print(f"{'-'*100}")
    for s in SIGMAS:
        a = results["path_A_secs"][str(s)]["mean"]
        b = results["path_B_secs"][str(s)]["mean"]
        f0 = results["flip_rate"][str(s)]["0"]
        f1 = results["flip_rate"][str(s)]["1"]
        cs = [results["path_C_secs"][f"sigma={s}_tau={t}"]["mean"] for t in TAUS]
        print(f"{s:>8.3f}  {a:>8.3f}  {b:>8.3f}  {f0:>8.3f}  {f1:>8.3f}  |  {cs[0]:>8.3f}  {cs[1]:>8.3f}  {cs[2]:>8.3f}  {cs[3]:>8.3f}  {cs[4]:>8.3f}")
    print(f"{'='*100}")

    print(f"\nSanity checks:")
    b0 = results["path_B_secs"]["0.0"]["mean"]
    a0 = results["path_A_secs"]["0.0"]["mean"]
    f0_check = results["flip_rate"]["0.0"]["0"]
    c0_t01 = results["path_C_secs"]["sigma=0.0_tau=0.1"]["mean"]
    print(f"  Path B @ σ=0 (src_K1 oracle):  SECS={b0:.3f}  (expect ~0.686)")
    print(f"  Path A @ σ=0 (continuous):     SECS={a0:.3f}  (expect ~0.589)")
    print(f"  Flip rate @ σ=0:                {f0_check:.6f}  (expect 0.0)")
    print(f"  Path C τ=0.1 @ σ=0 vs Path B:   {c0_t01:.3f} vs {b0:.3f}  (should be close)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T0: Noisy Latent Tolerance Sweep")
    parser.add_argument("--n_pairs", type=int, default=50, help="number of eval pairs (50=smoke, 200=full)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_sweep(args)
