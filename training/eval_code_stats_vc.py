"""
Learning-free VC: depth-wise code statistics interpolation.

NO neural network. NO adapter. NO training.

1. Encode target reference audio → extract per-depth code histograms
2. Encode source audio → extract per-depth source codes
3. For each speaker-depth, sample replacement codes from target histogram
4. Mix source/target code distributions with a knob α ∈ [0,1]
5. Decode and measure SECS

The knob IS the voice changer. No black box.

Usage:
  cd training
  uv run python eval_code_stats_vc.py --n_pairs 200
  uv run python eval_code_stats_vc.py --knob 0.5 --speaker_depths 1,2,3
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

N_DEPTHS = 9
CODEBOOK_SIZE = 1024


@torch.no_grad()
def get_all_codes(dac, z):
    """Sequential residual quantization → list of code indices per depth."""
    codes = []
    residual = z.clone()
    for d in range(N_DEPTHS):
        out = dac.quantizer.quantizers[d](residual)
        codes.append(out[3].squeeze(0).cpu())
        residual = residual - out[0]
    return codes


@torch.no_grad()
def get_all_quantized(dac, z):
    """Sequential quantization → list of (q_vec, codes) per depth."""
    layers = []
    residual = z.clone()
    for d in range(N_DEPTHS):
        out = dac.quantizer.quantizers[d](residual)
        layers.append((out[0], out[3]))
        residual = residual - out[0]
    return layers


@torch.no_grad()
def build_histogram(codes_per_depth):
    """Build per-depth code usage histograms from code sequences."""
    hists = []
    for d in range(N_DEPTHS):
        c = codes_per_depth[d].numpy()
        hist = np.bincount(c, minlength=CODEBOOK_SIZE).astype(np.float32)
        hist = hist / hist.sum()
        hists.append(hist)
    return hists


@torch.no_grad()
def soft_decode_depth(dac, depth, probs):
    """Decode one depth from code probabilities (soft codebook lookup).

    probs: [T, 1024] probability distribution over codebook entries
    Returns: [1, 1024, T] quantized latent for this depth
    """
    q = dac.quantizer.quantizers[depth]
    cb = q.codebook.weight  # [1024, 8]
    projected = probs @ cb  # [T, 8]
    q_vec = q.out_proj(projected.T.unsqueeze(0))  # [1, 1024, T]
    return q_vec


@torch.no_grad()
def stats_vc(dac, source_layers, target_hists, knob, speaker_depths, n_frames):
    """Learning-free VC via code statistics interpolation.

    For speaker depths: mix source onehot with target histogram by knob α.
    For non-speaker depths: keep source codes exactly.
    """
    z_total = None
    for d in range(N_DEPTHS):
        q_src, codes_src = source_layers[d]
        T = codes_src.shape[1]
        cb_size = CODEBOOK_SIZE

        if d in speaker_depths and knob > 0:
            src_onehot = F.one_hot(codes_src.squeeze(0).long(), cb_size).float()  # [T, 1024]
            tgt_dist = torch.from_numpy(target_hists[d]).to(DEVICE).unsqueeze(0).expand(T, -1)

            probs = (1 - knob) * src_onehot + knob * tgt_dist

            q_vec = soft_decode_depth(dac, d, probs)
        else:
            q_vec = q_src

        z_total = q_vec if z_total is None else z_total + q_vec

    return z_total


def compute_secs(ecapa, audio_tensor, timbre, source_emb):
    audio_16k = resample_16k(audio_tensor)
    if audio_16k.shape[-1] < 8000:
        return None
    emb = ecapa_embed(ecapa, audio_16k)
    t = F.cosine_similarity(emb, timbre, dim=-1).mean().item()
    s = F.cosine_similarity(emb, source_emb, dim=-1).mean().item()
    return {"secs_target": t, "secs_source": s, "margin": t - s}


def run_eval(args):
    print("=== Learning-Free Code Statistics VC ===\n")
    dac = load_dac()
    dac.eval()
    ecapa = load_ecapa()
    print()

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if args.n_pairs > 0:
        files = files[: args.n_pairs]
    print(f"Evaluating {len(files)} pairs\n")

    speaker_depths = set(int(x) for x in args.speaker_depths.split(","))

    knobs = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_results = {f"k{k}": [] for k in knobs}

    # Also compute oracle (src_K1) for comparison
    all_results["oracle_src_K1"] = []

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu", weights_only=False)
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        source_layers = get_all_quantized(dac, z_s)
        target_codes = get_all_codes(dac, z_t)
        target_hists = build_histogram(target_codes)

        # Oracle: src_K1 (source d0 + target d1-8 re-quantized)
        from eval_depth_surgery import requantize_from
        z_oracle = requantize_from(dac, z_s, z_t, 1)
        with torch.no_grad():
            oracle_audio = dac.decoder(z_oracle).squeeze(1)
        oracle_m = compute_secs(ecapa, oracle_audio, timbre, source_emb)
        if oracle_m:
            all_results["oracle_src_K1"].append(oracle_m)

        # Stats VC at each knob value
        for knob in knobs:
            z_vc = stats_vc(dac, source_layers, target_hists, knob, speaker_depths, 0)
            with torch.no_grad():
                vc_audio = dac.decoder(z_vc).squeeze(1)
            m = compute_secs(ecapa, vc_audio, timbre, source_emb)
            if m:
                all_results[f"k{knob}"].append(m)

        if (pi + 1) % 20 == 0:
            r = all_results["k0.5"]
            om = all_results["oracle_src_K1"]
            k05_margin = np.mean([x["margin"] for x in r]) if r else 0
            oracle_margin = np.mean([x["margin"] for x in om]) if om else 0
            print(
                f"  [{pi+1:>3}/{len(files)}] "
                f"stats k=0.5: m={k05_margin:+.3f}  "
                f"oracle: m={oracle_margin:+.3f}",
                flush=True,
            )

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"{'config':<18} {'target':>8} {'source':>8} {'margin':>8} {'n':>4}")
    print("-" * 80)

    summary = {}
    for cond in ["oracle_src_K1"] + [f"k{k}" for k in knobs]:
        rs = all_results[cond]
        if not rs:
            continue
        def sm(key):
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        summary[cond] = {
            "n": len(rs),
            "secs_target": sm("secs_target"),
            "secs_source": sm("secs_source"),
            "margin_mean": sm("margin"),
        }

        def fmt(v, sign=False):
            if v is None: return f"{'—':>8}"
            s = "+" if sign else ""
            return f"{v:{s}8.3f}"

        print(
            f"{cond:<18} "
            f"{fmt(sm('secs_target'))} {fmt(sm('secs_source'))} {fmt(sm('margin'), sign=True)} "
            f"{len(rs):>4}"
        )
    print(f"{'='*80}")

    oracle_margin = summary.get("oracle_src_K1", {}).get("margin_mean", 0)
    best_knob = None
    best_margin = -999
    for k in knobs:
        m = summary.get(f"k{k}", {}).get("margin_mean")
        if m is not None and m > best_margin:
            best_margin = m
            best_knob = k

    print(f"\nOracle src_K1 margin: {oracle_margin:+.3f}")
    print(f"Best stats knob: {best_knob} → margin {best_margin:+.3f}")
    if oracle_margin > 0:
        ratio = best_margin / oracle_margin if best_margin > 0 else 0
        print(f"Stats/oracle ratio: {ratio:.1%}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "per_pair": all_results,
            "config": {
                "speaker_depths": sorted(speaker_depths),
                "knobs": knobs,
            },
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning-free code statistics VC")
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--output", default="../results/code_stats_vc.json")
    parser.add_argument("--knob", type=float, default=None,
                        help="single knob value (overrides sweep)")
    parser.add_argument("--speaker_depths", type=str, default="1,2,3",
                        help="comma-separated depth indices to modify")
    args = parser.parse_args()
    run_eval(args)
