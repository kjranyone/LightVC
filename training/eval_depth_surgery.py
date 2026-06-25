"""
Depth surgery with residual-chain re-quantization.

For each same-text pair (DTW-aligned source/target):
  For each split point K (0..9):
    src_K: source depths 0..K-1 + target re-quantized at K..8
    tgt_K: target depths 0..K-1 + source re-quantized at K..8

Re-quantization: the residual after depths 0..K-1 is taken from the OTHER
speaker's latent, then quantized at depths K..8. This properly accounts for
the residual chain structure.

Learning-free. Pure codebook manipulation.

Usage:
  cd training
  uv run python eval_depth_surgery.py --n_pairs 200
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from train_phase3b import (
    DEVICE, DAC_SR, SECS_SR, load_dac, load_ecapa,
    resample_16k, ecapa_embed,
)

VCTK_TEXT_ROOT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")
N_DEPTHS = 9


@torch.no_grad()
def quantize_sequential(dac, z):
    """Quantize z through the full residual chain. Returns list of (q_vec, codes)."""
    layers = []
    residual = z.clone()
    for d in range(N_DEPTHS):
        out = dac.quantizer.quantizers[d](residual)
        q_vec = out[0]
        codes = out[3]
        layers.append((q_vec, codes))
        residual = residual - q_vec
    return layers


@torch.no_grad()
def requantize_from(dac, z_base, z_fill, keep_depths):
    """Build a mixed latent:
      - Depths 0..keep_depths-1: from z_base (quantized)
      - Depths keep_depths..8: re-quantize z_fill's residual

    The residual = z_fill - sum(q_base[0..keep_depths-1])
    This ensures the residual chain is coherent.
    """
    base_layers = quantize_sequential(dac, z_base)
    z_q = torch.zeros_like(z_base)
    for d in range(keep_depths):
        z_q = z_q + base_layers[d][0]
    residual = z_fill - z_q
    for d in range(keep_depths, N_DEPTHS):
        out = dac.quantizer.quantizers[d](residual)
        q_vec = out[0]
        z_q = z_q + q_vec
        residual = residual - q_vec
    return z_q


def compute_secs(ecapa, audio_tensor, timbre, source_emb):
    audio_16k = resample_16k(audio_tensor)
    if audio_16k.shape[-1] < 8000:
        return None
    emb = ecapa_embed(ecapa, audio_16k)
    t = F.cosine_similarity(emb, timbre, dim=-1).mean().item()
    s = F.cosine_similarity(emb, source_emb, dim=-1).mean().item()
    return {"secs_target": t, "secs_source": s, "margin": t - s}


def run_eval(args):
    print("=== Depth Surgery (Re-quantization) ===\n")
    dac = load_dac()
    dac.eval()
    ecapa = load_ecapa()

    whisper_pipe = None
    if not args.skip_whisper:
        from eval_streaming import load_whisper, load_vctk_text, compute_cer
        whisper_pipe = load_whisper()
    print()

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if args.n_pairs > 0:
        files = files[: args.n_pairs]
    print(f"Evaluating {len(files)} pairs\n")

    configs = []
    for k in range(N_DEPTHS + 1):
        configs.append((f"src_K{k}", "src", k))
    for k in range(N_DEPTHS + 1):
        configs.append((f"tgt_K{k}", "tgt", k))

    all_results = {name: [] for name, _, _ in configs}

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu", weights_only=False)
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)
        src_spk = d.get("src_spk", "")
        text_id = d.get("text_id", "")

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        ref_text = None
        if whisper_pipe and src_spk and text_id:
            ref_text = load_vctk_text(src_spk, text_id)

        for name, base, k in configs:
            z_base = z_s if base == "src" else z_t
            z_fill = z_t if base == "src" else z_s

            z_synth = requantize_from(dac, z_base, z_fill, k)
            with torch.no_grad():
                audio = dac.decoder(z_synth).squeeze(1)

            m = compute_secs(ecapa, audio, timbre, source_emb)
            if m is None:
                continue
            if whisper_pipe:
                cer_r = compute_cer(whisper_pipe, audio.squeeze().cpu().numpy(), ref_text)
                m["cer_vs_ref"] = cer_r.get("cer_vs_ref")
            all_results[name].append(m)

        if (pi + 1) % 10 == 0:
            parts = []
            for name, _, _ in configs:
                rs = all_results[name]
                if rs:
                    parts.append(f"{name}={np.mean([r['margin'] for r in rs]):+.3f}")
            print(f"  [{pi+1:>3}/{len(files)}] {' '.join(parts[:6])}", flush=True)

    # --- Summary ---
    print(f"\n{'='*90}")
    print(f"{'config':<10} {'target':>8} {'source':>8} {'margin':>8} {'cer':>6} {'n':>4}")
    print("-" * 90)

    summary = {}
    for name, _, _ in configs:
        rs = all_results[name]
        if not rs:
            continue
        def sm(key):
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        summary[name] = {
            "n": len(rs),
            "secs_target": sm("secs_target"),
            "secs_source": sm("secs_source"),
            "margin_mean": sm("margin"),
            "cer_vs_ref_mean": sm("cer_vs_ref"),
        }

        def fmt(v, sign=False):
            if v is None: return f"{'—':>8}"
            s = "+" if sign else ""
            return f"{v:{s}8.3f}"
        cer_str = f"{sm('cer_vs_ref'):6.2f}" if sm('cer_vs_ref') is not None else f"{'—':>6}"

        print(
            f"{name:<10} "
            f"{fmt(sm('secs_target'))} {fmt(sm('secs_source'))} {fmt(sm('margin'), sign=True)} "
            f"{cer_str} "
            f"{len(rs):>4}"
        )
    print(f"{'='*90}")

    print("""
Interpretation:
  src_K1 = source d0 + target d1-8 re-quantized (Phase 1b oracle)
  src_K0 = all-target (ceiling)
  src_K9 = all-source (floor = source identity)

  Look for the LARGEST margin jump between consecutive K values.
  That depth boundary is where speaker info concentrates.

  If margin is monotonically increasing with K (tgt series),
  speaker info is UNIFORMLY distributed across the residual chain.
  This would mean there is NO clean speaker/content split in RVQ structure.
""")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_pair": all_results}, f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth surgery with re-quantization")
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--output", default="../results/depth_surgery.json")
    parser.add_argument("--skip_whisper", action="store_true")
    args = parser.parse_args()
    run_eval(args)
