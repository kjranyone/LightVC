"""
Phase 3b: Empirical Generator Error Diagnostic

Diagnoses WHERE the TLG_Embed generator's predicted depth embeddings diverge
from oracle, and classifies the failure mode.

Conditions:
  1. oracle:      source d0 + oracle d1..8 → from_codes → decode
  2. model_hard:  source d0 + pred nearest-code d1..8 → from_codes → decode
  3. model_soft:  q0_s + pred soft blend d1..8 (τ sweep) → manual z_q → decode

Per-depth diagnostics:
  - Embedding cosine:  cos(pred_emb_d, oracle_emb_d)
  - Code accuracy:     frac(argmin(pred) == oracle)
  - Distance rank:     rank of oracle code in pred's sorted NN list

Overall:
  - SECS vs target (conversion quality)
  - SECS vs source (leakage)
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
CKPT_PATH = Path("checkpoints/phase3/latest.pt")

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


def load_model():
    from phase3_model import TLG_Embed
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    args = ck["args"]
    model = TLG_Embed(
        content_dim=1024,
        hidden_dim=args["hidden_dim"],
        timbre_dim=192,
        n_heads=8,
        n_layers=args["n_layers"],
    ).to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Loaded TLG_Embed (epoch={ck['epoch']}, hidden={args['hidden_dim']}, layers={args['n_layers']})")
    return model


@torch.no_grad()
def resample_16k(audio_44k):
    B = audio_44k.shape[0]
    flat = audio_44k.reshape(B, 1, -1)
    out = F.interpolate(flat, scale_factor=SECS_SR / DAC_SR,
                        mode="linear", align_corners=False)
    return out.squeeze(1)


@torch.no_grad()
def compute_embedding(ecapa, audio_16k):
    if audio_16k.shape[-1] < 8000:
        return None
    return ecapa.encode_batch(audio_16k).squeeze(0)


@torch.no_grad()
def cosine_sim(emb_a, emb_b):
    return F.cosine_similarity(emb_a, emb_b, dim=-1).item()


@torch.no_grad()
def get_source_d0_code(dac, z_s):
    _, _, _, codes_0, _ = dac.quantizer.quantizers[0](z_s)
    return codes_0


@torch.no_grad()
def hard_rvq_residual(dac, q0_s, z_input):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    residual = z_input - q0_s
    codes_all = []
    for d in range(1, n):
        q_out, _, _, codes_d, _ = qs[d](residual)
        residual = residual - q_out
        codes_all.append(codes_d)
    codes = torch.stack(codes_all, dim=1)
    return codes


@torch.no_grad()
def decode_from_codes(dac, codes_full):
    z_q, _, _ = dac.quantizer.from_codes(codes_full)
    return dac.decoder(z_q)


@torch.no_grad()
def decode_model_soft(dac, q0_s, pred_embeds, tau):
    z_q = q0_s.clone()
    for d in range(8):
        quantizer = dac.quantizer.quantizers[d + 1]
        cb = quantizer.codebook.weight

        pred = pred_embeds[:, d, :, :]
        dist = torch.cdist(pred, cb.unsqueeze(0)).pow(2)
        weights = F.softmax(-dist / tau, dim=-1)
        soft_emb = (weights @ cb).transpose(1, 2)
        q_d = quantizer.out_proj(soft_emb)
        z_q = z_q + q_d

    return dac.decoder(z_q)


def run_diagnostic(args):
    print("=== Phase 3b: Empirical Generator Error Diagnostic ===\n")

    dac = load_dac()
    print("DAC loaded")

    ecapa = load_ecapa()
    print("ECAPA loaded")

    model = load_model()
    print("TLG_Embed loaded\n")

    codebooks = []
    for d in range(1, 9):
        codebooks.append(dac.quantizer.quantizers[d].codebook.weight)
    codebooks = torch.stack(codebooks).to(DEVICE)

    files = sorted(DATA_DIR.glob("*.pt"))[:args.n_pairs]
    print(f"Pairs: {len(files)}")
    print(f"Taus:  {TAUS}\n")

    per_depth_cos = [[] for _ in range(8)]
    per_depth_acc = [[] for _ in range(8)]
    per_depth_rank = [[] for _ in range(8)]

    secs_oracle_t = []
    secs_model_hard_t = []
    secs_model_soft_t = {t: [] for t in TAUS}
    secs_oracle_s = []
    secs_model_hard_s = []
    secs_model_soft_s = {t: [] for t in TAUS}

    t0 = time.time()

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu")
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        q0_s = d["q0_s"].float().unsqueeze(0).to(DEVICE)
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        f0 = d["f0"].float().unsqueeze(0).to(DEVICE)
        energy = d["energy"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_embeds = model(z_s.transpose(1, 2), f0, energy, timbre)

            oracle_codes = hard_rvq_residual(dac, q0_s, z_t)
            oracle_embeds = torch.stack([
                codebooks[d, oracle_codes[:, d].long()]
                for d in range(8)
            ], dim=1)

            for depth in range(8):
                pe = pred_embeds[:, depth, :, :]
                oe = oracle_embeds[:, depth, :, :]
                cos = F.cosine_similarity(pe, oe, dim=-1)
                per_depth_cos[depth].append(cos.mean().item())

                pred_codes = torch.cdist(pe, codebooks[depth].unsqueeze(0)).argmin(dim=-1)
                acc = (pred_codes == oracle_codes[:, depth]).float().mean().item()
                per_depth_acc[depth].append(acc)

                dist_all = torch.cdist(pe, codebooks[depth].unsqueeze(0))
                oracle_d = dist_all.gather(
                    2, oracle_codes[:, depth].long().unsqueeze(-1)
                )
                rank = (dist_all < oracle_d).float().sum(dim=-1)
                per_depth_rank[depth].append(rank.mean().item())

            codes_0 = get_source_d0_code(dac, z_s)

            oracle_full = torch.cat([
                codes_0.unsqueeze(1),
                oracle_codes
            ], dim=1)
            audio_oracle = decode_from_codes(dac, oracle_full)
            audio_oracle_16k = resample_16k(audio_oracle.squeeze(1))

            pred_codes_hard = torch.stack([
                torch.cdist(pred_embeds[:, d], codebooks[d].unsqueeze(0)).argmin(dim=-1)
                for d in range(8)
            ], dim=1)
            model_full = torch.cat([
                codes_0.unsqueeze(1),
                pred_codes_hard
            ], dim=1)
            audio_hard = decode_from_codes(dac, model_full)
            audio_hard_16k = resample_16k(audio_hard.squeeze(1))

            audio_source = decode_from_codes(
                dac, get_source_full_codes(dac, z_s)
            )
            audio_source_16k = resample_16k(audio_source.squeeze(1))

            src_emb = compute_embedding(ecapa, audio_source_16k)
            if src_emb is None:
                continue

            ot = compute_embedding(ecapa, audio_oracle_16k)
            ht = compute_embedding(ecapa, audio_hard_16k)

            if ot is not None:
                secs_oracle_t.append(cosine_sim(ot, timbre.unsqueeze(0)))
                secs_oracle_s.append(cosine_sim(ot, src_emb))
            if ht is not None:
                secs_model_hard_t.append(cosine_sim(ht, timbre.unsqueeze(0)))
                secs_model_hard_s.append(cosine_sim(ht, src_emb))

            for tau in TAUS:
                audio_soft = decode_model_soft(dac, q0_s, pred_embeds, tau)
                audio_soft_16k = resample_16k(audio_soft.squeeze(1))
                st = compute_embedding(ecapa, audio_soft_16k)
                if st is not None:
                    secs_model_soft_t[tau].append(
                        cosine_sim(st, timbre.unsqueeze(0)))
                    secs_model_soft_s[tau].append(
                        cosine_sim(st, src_emb))

        elapsed = time.time() - t0
        eta = elapsed / (pi + 1) * (len(files) - pi - 1)
        print(
            f"  [{pi+1:>3}/{len(files)}] "
            f"oracle={secs_oracle_t[-1]:.3f} "
            f"hard={secs_model_hard_t[-1]:.3f} "
            f"soft_t1={secs_model_soft_t[1.0][-1]:.3f} "
            f"d1cos={per_depth_cos[0][-1]:.3f} "
            f"d1acc={per_depth_acc[0][-1]:.3f} "
            f"[{elapsed:.0f}s, ETA {eta:.0f}s]",
            flush=True,
        )

    results = {
        "n_pairs": len(files),
        "checkpoint_epoch": 18,
        "per_depth_cos_mean": [float(np.mean(v)) for v in per_depth_cos],
        "per_depth_cos_std": [float(np.std(v)) for v in per_depth_cos],
        "per_depth_acc_mean": [float(np.mean(v)) for v in per_depth_acc],
        "per_depth_acc_std": [float(np.std(v)) for v in per_depth_acc],
        "per_depth_rank_mean": [float(np.mean(v)) for v in per_depth_rank],
        "per_depth_rank_std": [float(np.std(v)) for v in per_depth_rank],
        "secs_oracle_target": {
            "mean": float(np.mean(secs_oracle_t)),
            "std": float(np.std(secs_oracle_t)),
        },
        "secs_model_hard_target": {
            "mean": float(np.mean(secs_model_hard_t)),
            "std": float(np.std(secs_model_hard_t)),
        },
        "secs_model_soft_target": {
            str(t): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }
            for t, v in secs_model_soft_t.items()
        },
        "secs_oracle_source": {
            "mean": float(np.mean(secs_oracle_s)),
            "std": float(np.std(secs_oracle_s)),
        },
        "secs_model_hard_source": {
            "mean": float(np.mean(secs_model_hard_s)),
            "std": float(np.std(secs_model_hard_s)),
        },
        "secs_model_soft_source": {
            str(t): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }
            for t, v in secs_model_soft_s.items()
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"phase3b_empirical_error_n{args.n_pairs}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*90}")
    print(f"{'depth':>6}  {'cosine':>8}  {'acc':>8}  {'rank':>8}")
    print(f"{'-'*90}")
    for d in range(8):
        print(f"{'d'+str(d+1):>6}  "
              f"{results['per_depth_cos_mean'][d]:>8.3f}  "
              f"{results['per_depth_acc_mean'][d]:>8.3f}  "
              f"{results['per_depth_rank_mean'][d]:>8.1f}")

    print(f"\n{'='*90}")
    print(f"{'condition':>20}  {'SECS_tgt':>8}  {'SECS_src':>8}  {'tgt-src':>8}")
    print(f"{'-'*90}")
    ot_ = results["secs_oracle_target"]["mean"]
    os_ = results["secs_oracle_source"]["mean"]
    ht_ = results["secs_model_hard_target"]["mean"]
    hs_ = results["secs_model_hard_source"]["mean"]
    print(f"{'oracle':>20}  {ot_:>8.3f}  {os_:>8.3f}  {ot_-os_:>8.3f}")
    print(f"{'model_hard':>20}  {ht_:>8.3f}  {hs_:>8.3f}  {ht_-hs_:>8.3f}")
    for t in TAUS:
        st_ = results["secs_model_soft_target"][str(t)]["mean"]
        ss_ = results["secs_model_soft_source"][str(t)]["mean"]
        print(f"{'model_soft τ='+str(t):>20}  {st_:>8.3f}  {ss_:>8.3f}  {st_-ss_:>8.3f}")
    print(f"{'='*90}")

    print(f"\nFailure mode classification:")
    hard_secs = ht_
    best_soft = max(results["secs_model_soft_target"][str(t)]["mean"] for t in TAUS)
    d1_cos = results["per_depth_cos_mean"][0]
    d2_cos = results["per_depth_cos_mean"][1]
    d3_cos = results["per_depth_cos_mean"][2]
    all_cos = results["per_depth_cos_mean"]
    mean_cos = float(np.mean(all_cos))

    print(f"  hard SECS:     {hard_secs:.3f}")
    print(f"  best soft:     {best_soft:.3f}  (Δ={best_soft-hard_secs:+.3f})")
    print(f"  d1 cosine:     {d1_cos:.3f}")
    print(f"  d2 cosine:     {d2_cos:.3f}")
    print(f"  d3+ cosine:    {float(np.mean(all_cos[2:])):.3f}")
    print(f"  mean cosine:   {mean_cos:.3f}")

    if best_soft - hard_secs > 0.1:
        print(f"\n  → Pattern A: code selection is coarse. soft RVQ + audio-domain loss is the main line.")
    elif d1_cos < 0.3 or d2_cos < 0.3:
        print(f"\n  → Pattern C: d1-d2 speaker direction is broken. Need depth-weighted / speaker loss.")
    elif mean_cos > 0.5 and hard_secs < 0.2:
        print(f"\n  → Pattern D: embeddings are close but SECS is low. Audio-domain loss mandatory.")
    else:
        print(f"\n  → Pattern B: generator hasn't learned speaker-bearing residual direction.")
        print(f"    Need fundamental objective change (audio-domain SECS loss).")


@torch.no_grad()
def get_source_full_codes(dac, z_s):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    residual = z_s.clone()
    codes_all = []
    for d in range(n):
        _, _, _, codes_d, _ = qs[d](residual)
        q_out, _, _, _, _ = qs[d](residual)
        residual = residual - q_out
        codes_all.append(codes_d)
    return torch.stack(codes_all, dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b Empirical Error Diagnostic")
    parser.add_argument("--n_pairs", type=int, default=50)
    args = parser.parse_args()
    run_diagnostic(args)
