"""
Cross-text evaluation: test whether the B1 adapter generalizes beyond same-text.

Takes eval pairs and crosses them: source latent from pair i, target timbre from pair j
(where i and j have different text_ids). Measures SECS/CER to check if:
  - Timbre conversion still works (SECS_target high)
  - Content is preserved (CER vs source text low)
  - Source speaker is suppressed (SECS_source low)

If cross-text margin collapses vs same-text, the adapter exploits same-text alignment.

Usage:
  cd training
  uv run python eval_cross_text.py --n_pairs 200
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
    resample_16k, ecapa_embed, soft_rvq_requantize,
)
from export_streaming_samples import load_adapter, offline_pipeline, quantize_q0
from eval_streaming import (
    compute_secs, compute_f0_metrics, compute_mcd,
    load_whisper, load_vctk_text, compute_cer,
)

TAU = 5.0
VCTK_TEXT_ROOT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")


def build_cross_pairs(files, n_cross, seed=42):
    rng = np.random.default_rng(seed)
    n = len(files)
    pairs = []
    for i in range(min(n_cross, n)):
        for _ in range(20):
            j = rng.integers(0, n)
            if j == i:
                continue
            di = torch.load(files[i], map_location="cpu", weights_only=False)
            dj = torch.load(files[j], map_location="cpu", weights_only=False)
            if di.get("text_id") != dj.get("text_id"):
                pairs.append((i, j, files[i], files[j]))
                break
    return pairs


def run_eval(args):
    print("=== Cross-Text Evaluation ===\n")
    dac = load_dac()
    ecapa = load_ecapa()
    adapter = load_adapter(args.adapter_ckpt)
    whisper_pipe = None if args.skip_whisper else load_whisper()
    print(f"Adapter: {args.adapter_ckpt}\n")

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if args.n_pairs > 0:
        files = files[: args.n_pairs]
    print(f"Eval pairs available: {len(files)}")

    cross_pairs = build_cross_pairs(files, args.n_cross)
    print(f"Cross-text pairs: {len(cross_pairs)}\n")

    conditions = ["same_text", "cross_text"]
    all_results = {c: [] for c in conditions}

    for idx, (i, j, fi, fj) in enumerate(cross_pairs):
        di = torch.load(fi, map_location="cpu", weights_only=False)
        dj = torch.load(fj, map_location="cpu", weights_only=False)

        z_s = di["z_s"].float().unsqueeze(0).to(DEVICE)
        src_spk = di.get("src_spk", "")
        src_text_id = di.get("text_id", "")

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))
        pcm_np = source_audio.squeeze().cpu().numpy()

        ref_text = None
        if whisper_pipe and src_spk and src_text_id:
            ref_text = load_vctk_text(src_spk, src_text_id)

        pair_results = {}

        # --- Same-text: source z_s + own target timbre ---
        timbre_same = di["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            same_audio = offline_pipeline(dac, adapter, z_s, timbre_same)
        same_m = compute_secs(ecapa, same_audio, timbre_same, source_emb)
        if same_m:
            same_mcd = compute_mcd(pcm_np, same_audio.squeeze().cpu().numpy())
            same_m.update(same_mcd)
            if whisper_pipe:
                same_cer = compute_cer(whisper_pipe, same_audio.squeeze().cpu().numpy(), ref_text)
                same_m["cer_vs_ref"] = same_cer.get("cer_vs_ref")
            all_results["same_text"].append(same_m)
            pair_results["same_text"] = same_m

        # --- Cross-text: source z_s + DIFFERENT target timbre ---
        timbre_cross = dj["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            cross_audio = offline_pipeline(dac, adapter, z_s, timbre_cross)
        cross_m = compute_secs(ecapa, cross_audio, timbre_cross, source_emb)
        if cross_m:
            cross_mcd = compute_mcd(pcm_np, cross_audio.squeeze().cpu().numpy())
            cross_m.update(cross_mcd)
            if whisper_pipe:
                cross_cer = compute_cer(whisper_pipe, cross_audio.squeeze().cpu().numpy(), ref_text)
                cross_m["cer_vs_ref"] = cross_cer.get("cer_vs_ref")
            all_results["cross_text"].append(cross_m)
            pair_results["cross_text"] = cross_m

        s_margin = pair_results.get("same_text", {}).get("margin", 0)
        c_margin = pair_results.get("cross_text", {}).get("margin", 0)
        s_tgt = pair_results.get("same_text", {}).get("secs_target", 0)
        c_tgt = pair_results.get("cross_text", {}).get("secs_target", 0)
        s_cer = pair_results.get("same_text", {}).get("cer_vs_ref")
        c_cer = pair_results.get("cross_text", {}).get("cer_vs_ref")
        print(
            f"  [{idx+1:>3}/{len(cross_pairs)}] {di.get('src_spk','?')}→{dj.get('tgt_spk','?')} "
            f"text {src_text_id}→{dj.get('text_id','?')}  "
            f"same: tgt={s_tgt:.3f} m={s_margin:+.3f}  "
            f"cross: tgt={c_tgt:.3f} m={c_margin:+.3f}"
            + (f"  CER:{s_cer:.2f}/{c_cer:.2f}" if s_cer is not None and c_cer is not None else ""),
            flush=True,
        )

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"{'condition':<14} {'target':>8} {'source':>8} {'margin':>8} {'mcd':>6} {'cer':>6} {'n':>4}")
    print("-" * 80)

    summary = {}
    for cond in conditions:
        rs = all_results[cond]
        if not rs:
            continue
        def sm(key):
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        t_mean = sm("secs_target")
        s_mean = sm("secs_source")
        m_mean = sm("margin")
        mcd_mean = sm("mcd")
        cer_mean = sm("cer_vs_ref")

        summary[cond] = {
            "n": len(rs),
            "secs_target": t_mean,
            "secs_source": s_mean,
            "margin_mean": m_mean,
            "mcd_mean": mcd_mean,
            "cer_vs_ref_mean": cer_mean,
        }

        def fmt(v, sign=False):
            if v is None: return f"{'—':>8}"
            s = "+" if sign else ""
            return f"{v:{s}8.3f}"
        def fmt_s(v):
            return f"{v:6.2f}" if v is not None else f"{'—':>6}"

        print(
            f"{cond:<14} "
            f"{fmt(t_mean)} {fmt(s_mean)} {fmt(m_mean, sign=True)} "
            f"{fmt_s(mcd_mean)} "
            f"{fmt_s(cer_mean) if cer_mean is not None else fmt_s(None)} "
            f"{len(rs):>4}"
        )
    print(f"{'='*80}")

    if "same_text" in summary and "cross_text" in summary:
        delta_target = summary["cross_text"]["secs_target"] - summary["same_text"]["secs_target"]
        delta_margin = summary["cross_text"]["margin_mean"] - summary["same_text"]["margin_mean"]
        print(f"\n  Δ target SECS (cross - same): {delta_target:+.3f}")
        print(f"  Δ margin      (cross - same): {delta_margin:+.3f}")
        if delta_margin < -0.10:
            print("  >> MARGIN COLLAPSE: adapter does NOT generalize to cross-text")
        elif delta_margin < -0.05:
            print("  >> MODERATE DEGRADATION: partial generalization")
        else:
            print("  >> GENERALIZES: cross-text performance comparable to same-text")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_pair": all_results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-text evaluation")
    parser.add_argument("--n_pairs", type=int, default=200,
                        help="number of eval pairs to load")
    parser.add_argument("--n_cross", type=int, default=100,
                        help="number of cross-text pairs to test")
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--adapter_ckpt",
                        default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    parser.add_argument("--output", default="../results/cross_text_eval.json")
    parser.add_argument("--skip_whisper", action="store_true")
    args = parser.parse_args()
    run_eval(args)
