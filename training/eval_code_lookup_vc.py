"""
Code lookup VC: P(target_code | source_code, depth) from training data.

Build a co-occurrence matrix from training pairs:
  For each frame t, depth d: M[d][src_code][tgt_code] += 1

At inference:
  Given source code at frame t, depth d:
    target_dist = normalize(M[d][src_code])
    Mix source onehot with target_dist by knob α

This is a lookup table — no neural network. But it captures content-conditional
code patterns (unlike the marginal histogram).

Usage:
  cd training
  uv run python eval_code_lookup_vc.py --n_train 500 --n_eval 200
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
CB_SIZE = 1024


@torch.no_grad()
def get_codes(dac, z):
    codes = []
    residual = z.clone()
    for d in range(N_DEPTHS):
        out = dac.quantizer.quantizers[d](residual)
        codes.append(out[3].squeeze(0).cpu())
        residual = residual - out[0]
    return codes


@torch.no_grad()
def build_lookup(dac, train_files, max_pairs):
    """Build co-occurrence lookup: M[d][i][j] = count(src=i, tgt=j)."""
    M = np.zeros((N_DEPTHS, CB_SIZE, CB_SIZE), dtype=np.float32)
    files = sorted(train_files)[:max_pairs]
    for fi, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        codes_s = get_codes(dac, z_s)
        codes_t = get_codes(dac, z_t)
        for depth in range(N_DEPTHS):
            cs = codes_s[depth].numpy()
            ct = codes_t[depth].numpy()
            np.add.at(M[depth], (cs, ct), 1.0)
        if (fi + 1) % 100 == 0:
            print(f"  lookup build: [{fi+1}/{len(files)}]", flush=True)

    row_sums = M.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    M_norm = M / row_sums
    return M_norm


@torch.no_grad()
def lookup_vc(dac, z_s, lookup, knob, speaker_depths):
    """Apply code lookup VC with knob control."""
    z_total = None
    residual = z_s.clone()
    for d in range(N_DEPTHS):
        q = dac.quantizer.quantizers[d]
        out = q(residual)
        q_vec = out[0]
        codes = out[3].squeeze(0)  # [T]

        if d in speaker_depths and knob > 0:
            T = codes.shape[0]
            src_onehot = F.one_hot(codes.long(), CB_SIZE).float().to(DEVICE)  # [T, 1024]

            tgt_dists = torch.from_numpy(lookup[d][codes.cpu().numpy()]).to(DEVICE)  # [T, 1024]

            probs = (1 - knob) * src_onehot + knob * tgt_dists

            cb = q.codebook.weight  # [1024, 8]
            projected = probs @ cb  # [T, 8]
            q_vec = q.out_proj(projected.T.unsqueeze(0))  # [1, 1024, T]

        z_total = q_vec if z_total is None else z_total + q_vec
        residual = residual - out[0]

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
    print("=== Code Lookup VC ===\n")
    dac = load_dac()
    dac.eval()
    ecapa = load_ecapa()

    train_dir = Path(args.train_dir)
    train_files = sorted(train_dir.glob("*.pt"))
    print(f"Building lookup from {min(args.n_train, len(train_files))} train pairs...")
    lookup = build_lookup(dac, train_files, args.n_train)
    print(f"Lookup built: [{N_DEPTHS}, {CB_SIZE}, {CB_SIZE}]\n")

    eval_files = sorted(Path(args.eval_dir).glob("*.pt"))
    if args.n_eval > 0:
        eval_files = eval_files[: args.n_eval]
    print(f"Evaluating on {len(eval_files)} pairs\n")

    speaker_depths = set(int(x) for x in args.speaker_depths.split(","))
    knobs = [0.0, 0.25, 0.5, 0.75, 1.0]
    all_results = {f"k{k}": [] for k in knobs}
    all_results["oracle"] = []

    from eval_depth_surgery import requantize_from

    for pi, fpath in enumerate(eval_files):
        d = torch.load(fpath, map_location="cpu", weights_only=False)
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        z_t = d["z_t_aligned"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        z_oracle = requantize_from(dac, z_s, z_t, 1)
        with torch.no_grad():
            oracle_audio = dac.decoder(z_oracle).squeeze(1)
        om = compute_secs(ecapa, oracle_audio, timbre, source_emb)
        if om:
            all_results["oracle"].append(om)

        for knob in knobs:
            z_vc = lookup_vc(dac, z_s, lookup, knob, speaker_depths)
            with torch.no_grad():
                vc_audio = dac.decoder(z_vc).squeeze(1)
            m = compute_secs(ecapa, vc_audio, timbre, source_emb)
            if m:
                all_results[f"k{knob}"].append(m)

        if (pi + 1) % 20 == 0:
            k5 = all_results["k0.5"]
            k5m = np.mean([x["margin"] for x in k5]) if k5 else 0
            print(f"  [{pi+1:>3}/{len(eval_files)}] k=0.5: m={k5m:+.3f}", flush=True)

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"{'config':<14} {'target':>8} {'source':>8} {'margin':>8} {'n':>4}")
    print("-" * 80)

    summary = {}
    for cond in ["oracle"] + [f"k{k}" for k in knobs]:
        rs = all_results[cond]
        if not rs:
            continue
        def sm(key):
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        summary[cond] = {"n": len(rs), "secs_target": sm("secs_target"),
                         "secs_source": sm("secs_source"), "margin_mean": sm("margin")}
        def fmt(v, sign=False):
            if v is None: return f"{'—':>8}"
            s = "+" if sign else ""
            return f"{v:{s}8.3f}"
        print(f"{cond:<14} {fmt(sm('secs_target'))} {fmt(sm('secs_source'))} {fmt(sm('margin'), sign=True)} {len(rs):>4}")
    print(f"{'='*80}")

    oracle_m = summary.get("oracle", {}).get("margin_mean", 0)
    best_k, best_m = None, -999
    for k in knobs:
        m = summary.get(f"k{k}", {}).get("margin_mean")
        if m is not None and m > best_m:
            best_m, best_k = m, k
    ratio = best_m / oracle_m if oracle_m > 0 and best_m > 0 else 0
    print(f"\nOracle: {oracle_m:+.3f} | Best knob {best_k}: {best_m:+.3f} ({ratio:.0%} of oracle)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_pair": all_results,
                   "config": {"speaker_depths": sorted(speaker_depths), "n_train": args.n_train}},
                  f, indent=2, default=str)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code lookup VC")
    parser.add_argument("--n_train", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--train_dir", default="../data/phase3_10k/train")
    parser.add_argument("--eval_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--output", default="../results/code_lookup_vc.json")
    parser.add_argument("--speaker_depths", type=str, default="1,2,3")
    args = parser.parse_args()
    run_eval(args)
