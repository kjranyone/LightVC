"""
Streaming evaluation: SECS/margin/SNR on offline vs streaming pipeline.

Usage:
  cd training
  uv run python eval_streaming.py --n_pairs 25
  uv run python eval_streaming.py --n_pairs 200 --data_dir ../data/phase3_10k/eval
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
    hard_rvq_requantize,
)
from train_phase3c_adapter import TimbreAdapter
from export_streaming_samples import (
    streaming_pipeline, offline_pipeline, load_adapter,
    quantize_q0, aligned_snr,
)

HOP = 512
TAU = 5.0


def compute_secs(ecapa, audio_tensor, timbre, source_emb):
    audio_16k = resample_16k(audio_tensor)
    if audio_16k.shape[-1] < 8000:
        return None
    emb = ecapa_embed(ecapa, audio_16k)
    t = F.cosine_similarity(emb, timbre, dim=-1).mean().item()
    s = F.cosine_similarity(emb, source_emb, dim=-1).mean().item()
    return {"secs_target": t, "secs_source": s, "margin": t - s}


def run_eval(args):
    print("=== Streaming Evaluation ===\n")
    dac = load_dac()
    ecapa = load_ecapa()
    adapter = load_adapter(args.adapter_ckpt)
    print(f"Models loaded (adapter: {args.adapter_ckpt})\n")

    files = sorted(Path(args.data_dir).glob("*.pt"))[: args.n_pairs]
    print(f"Evaluating {len(files)} pairs\n")

    conditions = ["offline", "balanced_4f"]
    all_results = {c: [] for c in conditions}

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu")
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))
        pcm_np = source_audio.squeeze().cpu().numpy()

        pair_results = {}

        with torch.no_grad():
            off_audio = offline_pipeline(dac, adapter, z_s, timbre)
        off_m = compute_secs(ecapa, off_audio, timbre, source_emb)
        if off_m:
            all_results["offline"].append(off_m)
            pair_results["offline"] = off_m

        off_np = off_audio.squeeze().cpu().numpy()

        bal_audio = streaming_pipeline(dac, adapter, pcm_np, timbre, 4, 4)
        bal_tensor = torch.from_numpy(bal_audio).float().unsqueeze(0).to(DEVICE)
        bal_m = compute_secs(ecapa, bal_tensor, timbre, source_emb)
        snr_bal, lag_bal = aligned_snr(off_np, bal_audio)
        if bal_m:
            bal_m["snr_vs_offline"] = snr_bal
            bal_m["align_lag"] = lag_bal
            all_results["balanced_4f"].append(bal_m)
            pair_results["balanced_4f"] = bal_m

        print(
            f"  [{pi+1:>3}/{len(files)}] {fpath.stem}  "
            f"off: margin={pair_results.get('offline', {}).get('margin', 0):+.3f}  "
            f"bal: margin={pair_results.get('balanced_4f', {}).get('margin', 0):+.3f}  "
            f"SNR={snr_bal:.1f}dB",
            flush=True,
        )

    print(f"\n{'='*70}")
    print(f"{'condition':<16} {'target':>8} {'source':>8} {'margin':>8} {'snr':>8} {'n':>4}")
    print(f"{'-'*70}")

    summary = {}
    for cond in conditions:
        rs = all_results[cond]
        if not rs:
            continue
        t_mean = np.mean([r["secs_target"] for r in rs])
        s_mean = np.mean([r["secs_source"] for r in rs])
        m_mean = np.mean([r["margin"] for r in rs])
        m_std = np.std([r["margin"] for r in rs])
        snr_vals = [r.get("snr_vs_offline", float("nan")) for r in rs]
        snr_mean = np.nanmean(snr_vals) if snr_vals else float("nan")

        summary[cond] = {
            "n": len(rs),
            "secs_target": float(t_mean),
            "secs_source": float(s_mean),
            "margin_mean": float(m_mean),
            "margin_std": float(m_std),
            "snr_vs_offline_mean": float(snr_mean) if not np.isnan(snr_mean) else None,
        }
        snr_str = f"{snr_mean:>7.1f}dB" if not np.isnan(snr_mean) else f"{'—':>8}"
        print(
            f"{cond:<16} {t_mean:>8.3f} {s_mean:>8.3f} {m_mean:>+8.3f} "
            f"{snr_str} {len(rs):>4}"
        )
    print(f"{'='*70}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"summary": summary, "per_pair": all_results},
            f,
            indent=2,
            default=float,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming pipeline evaluation")
    parser.add_argument("--n_pairs", type=int, default=25,
                        help="number of eval pairs (-1 for all)")
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--adapter_ckpt", default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    parser.add_argument("--output", default="../results/streaming_eval.json")
    args = parser.parse_args()
    run_eval(args)
