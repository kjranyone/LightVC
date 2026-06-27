"""
Per-depth knob evaluation for DepthAwareAdapter.

Tests different knob configurations:
  identity_only:  knobs = [1,1,1,0,0]  (d1-3 on, d4-5 off)
  character_only: knobs = [0,0,0,1,1]  (d1-3 off, d4-5 on)
  full:           knobs = [1,1,1,1,1]  (all on)
  identity_half:  knobs = [0.5,0.5,0.5,0,0]

Measures SECS to verify:
  - d1-3 knobs control speaker conversion (turning them off kills margin)
  - d4-5 knobs add character (turning them off slightly reduces margin)
  - knobs are partially independent

Usage:
  cd training
  uv run python eval_depth_knobs.py --ckpt checkpoints/phase3c_depth_v2/best.pt
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
    DEVICE, DAC_SR, load_dac, load_ecapa,
    resample_16k, ecapa_embed, soft_rvq_requantize,
    hard_quantize_all,
)
from train_phase3c_adapter import DepthAwareAdapter, depth_aware_soft_rvq


KNOB_CONFIGS = {
    "full":              [1.0, 1.0, 1.0, 1.0, 1.0],
    "identity_only":     [1.0, 1.0, 1.0, 0.0, 0.0],
    "character_only":    [0.0, 0.0, 0.0, 1.0, 1.0],
    "identity_half":     [0.5, 0.5, 0.5, 0.0, 0.0],
    "character_half":    [1.0, 1.0, 1.0, 0.5, 0.5],
    "off":               [0.0, 0.0, 0.0, 0.0, 0.0],
}


def load_depth_adapter(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ca = ck["args"]
    sp_depths = ca.get("speaker_depths", (1, 2, 3, 4, 5))
    if isinstance(sp_depths, str):
        sp_depths = tuple(int(x) for x in sp_depths.split(","))
    adapter = DepthAwareAdapter(
        n_speaker_depths=len(sp_depths),
        latent_dim=1024,
        bottleneck=ca.get("bottleneck", 256),
        timbre_dim=192,
        n_tokens=ca.get("n_tokens", 32),
        n_heads=ca.get("n_heads", 4),
        kernel=ca.get("kernel", 3),
    ).to(DEVICE)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    return adapter, sp_depths, ca


def compute_secs(ecapa, audio_tensor, timbre, source_emb):
    audio_16k = resample_16k(audio_tensor)
    if audio_16k.shape[-1] < 8000:
        return None
    emb = ecapa_embed(ecapa, audio_16k)
    t = F.cosine_similarity(emb, timbre, dim=-1).mean().item()
    s = F.cosine_similarity(emb, source_emb, dim=-1).mean().item()
    return {"secs_target": t, "secs_source": s, "margin": t - s}


def run_eval(args):
    print("=== Depth Knob Evaluation ===\n")
    dac = load_dac()
    ecapa = load_ecapa()
    adapter, sp_depths, ca = load_depth_adapter(args.ckpt)
    tau = ca.get("tau", 5.0)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Speaker depths: {sp_depths}")
    print(f"Tau: {tau}\n")

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if args.n_pairs > 0:
        files = files[: args.n_pairs]
    print(f"Evaluating {len(files)} pairs\n")

    all_results = {name: [] for name in KNOB_CONFIGS}

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu", weights_only=False)
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        q0_s = d["q0_s"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        for name, knobs in KNOB_CONFIGS.items():
            with torch.no_grad():
                z_q = depth_aware_soft_rvq(
                    dac, q0_s, z_s, tau, adapter, timbre,
                    speaker_depths=sp_depths, knobs=knobs,
                )
                audio = dac.decoder(z_q).squeeze(1)
            m = compute_secs(ecapa, audio, timbre, source_emb)
            if m:
                all_results[name].append(m)

        if (pi + 1) % 20 == 0:
            full_m = np.mean([r["margin"] for r in all_results["full"]]) if all_results["full"] else 0
            id_m = np.mean([r["margin"] for r in all_results["identity_only"]]) if all_results["identity_only"] else 0
            print(f"  [{pi+1:>3}/{len(files)}] full={full_m:+.3f} id_only={id_m:+.3f}", flush=True)

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"{'config':<20} {'target':>8} {'source':>8} {'margin':>8} {'n':>4}")
    print("-" * 80)

    summary = {}
    for name in KNOB_CONFIGS:
        rs = all_results[name]
        if not rs:
            continue
        def sm(key):
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        summary[name] = {"target": sm("secs_target"), "source": sm("secs_source"), "margin": sm("margin")}
        def fmt(v, sign=False):
            if v is None: return f"{'—':>8}"
            s = "+" if sign else ""
            return f"{v:{s}8.3f}"
        print(f"{name:<20} {fmt(sm('secs_target'))} {fmt(sm('secs_source'))} {fmt(sm('margin'), sign=True)} {len(rs):>4}")
    print(f"{'='*80}")

    full_m = summary.get("full", {}).get("margin", 0)
    id_m = summary.get("identity_only", {}).get("margin", 0)
    char_m = summary.get("character_only", {}).get("margin", 0)
    off_m = summary.get("off", {}).get("margin", 0)
    print(f"\nAnalysis:")
    print(f"  Full vs Off:         {full_m - off_m:+.3f} (total adapter contribution)")
    print(f"  Identity contribution: {id_m - off_m:+.3f} (d1-3 alone)")
    print(f"  Character contribution: {char_m - off_m:+.3f} (d4-5 alone)")
    print(f"  Identity share:       {(id_m - off_m) / (full_m - off_m) * 100:.0f}%" if full_m != off_m else "")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_pair": all_results,
                   "knob_configs": KNOB_CONFIGS}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-depth knob evaluation")
    parser.add_argument("--ckpt", default="checkpoints/phase3c_depth_v2/best.pt")
    parser.add_argument("--n_pairs", type=int, default=200)
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--output", default="../results/depth_knobs.json")
    args = parser.parse_args()
    run_eval(args)
