"""
Full Phase 3c evaluation with paired bootstrap CI.
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from phase3_model import TLG
from train_phase3b import (
    DEVICE,
    DATA_DIR,
    PairDataset,
    collate,
    load_dac,
    load_ecapa,
    resample_16k,
    ecapa_embed,
    soft_rvq_requantize,
    hard_rvq_requantize,
    hard_quantize_all,
)
from train_phase3c_adapter import TimbreAdapter


RESULTS_DIR = Path("../results")


def ci_bootstrap(values, n_boot=1000, seed=1234):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "std": 0.0}
    means = []
    n = len(arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        means.append(float(arr[idx].mean()))
    return {
        "mean": float(arr.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "std": float(arr.std()),
    }


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]

    generator = None
    if not a.get("adapter_only", False):
        generator = TLG(
            content_dim=1024,
            hidden_dim=a.get("hidden_dim", 512),
            timbre_dim=192,
            n_heads=8,
            n_layers=a.get("n_layers", 6),
            causal=True,
        ).to(DEVICE)
        generator.load_state_dict(ck["generator"])
        generator.eval()

    adapter = TimbreAdapter(
        latent_dim=1024,
        timbre_dim=192,
        bottleneck=a.get("bottleneck", 256),
        kernel=a.get("kernel", 3),
        n_blocks=a.get("n_blocks", 1),
        utte_mode=a.get("utte_mode", "none"),
        film_mode=a.get("film_mode", "full"),
        n_tokens=a.get("n_tokens", 32),
        n_heads=a.get("n_heads", 4),
    ).to(DEVICE)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    return ck, generator, adapter


@torch.no_grad()
def eval_full(args):
    print("=== Phase 3c Full Eval ===")
    print(f"device={DEVICE}")
    print(f"checkpoint={args.checkpoint}")

    ck, generator, adapter = load_checkpoint(args.checkpoint)
    ck_args = ck["args"]
    tau = args.tau if args.tau > 0 else ck_args.get("tau", 5.0)
    max_frames = args.max_frames if args.max_frames > 0 else ck_args.get("max_frames", 256)
    data_dir = Path(args.data_dir) if args.data_dir else Path(ck_args.get("data_dir", DATA_DIR))
    print(f"epoch={ck.get('epoch')} tau={tau} max_frames={max_frames}")
    print(f"data_dir={data_dir}")

    dac = load_dac()
    ecapa = load_ecapa()
    ds = PairDataset(data_dir / "eval", max_frames)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=args.num_workers)

    rows = []
    t0 = time.time()

    for bi, batch in enumerate(dl):
        z_s, q0_s, z_t, f0, energy, timbre = [x.to(DEVICE) for x in batch]

        if generator:
            z_pred = generator(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
        else:
            z_pred = z_s

        z_q = soft_rvq_requantize(dac, q0_s, z_pred, tau)
        z_q_adapted = adapter(z_q, timbre, z_t)
        audio = dac.decoder(z_q_adapted).squeeze(1)
        emb = ecapa_embed(ecapa, resample_16k(audio))

        source_z = hard_quantize_all(dac, z_s)
        source_audio = dac.decoder(source_z).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        oracle_z = hard_rvq_requantize(dac, q0_s, z_t)
        oracle_audio = dac.decoder(oracle_z).squeeze(1)
        oracle_emb = ecapa_embed(ecapa, resample_16k(oracle_audio))

        target = F.cosine_similarity(emb, timbre, dim=-1).detach().cpu().numpy()
        source = F.cosine_similarity(emb, source_emb, dim=-1).detach().cpu().numpy()
        oracle_target = F.cosine_similarity(oracle_emb, timbre, dim=-1).detach().cpu().numpy()
        oracle_source = F.cosine_similarity(oracle_emb, source_emb, dim=-1).detach().cpu().numpy()
        delta = (z_q_adapted - z_q).pow(2).mean(dim=(1, 2)).sqrt().detach().cpu().numpy()

        start = bi * args.batch_size
        for i in range(len(target)):
            rows.append({
                "index": start + i,
                "target": float(target[i]),
                "source": float(source[i]),
                "margin": float(target[i] - source[i]),
                "oracle_target": float(oracle_target[i]),
                "oracle_source": float(oracle_source[i]),
                "oracle_margin": float(oracle_target[i] - oracle_source[i]),
                "delta_norm": float(delta[i]),
            })

        elapsed = time.time() - t0
        done = min((bi + 1) * args.batch_size, len(ds))
        eta = elapsed / max(done, 1) * (len(ds) - done)
        print(
            f"  [{done:>3}/{len(ds)}] "
            f"target={np.mean([r['target'] for r in rows]):.3f} "
            f"source={np.mean([r['source'] for r in rows]):.3f} "
            f"margin={np.mean([r['margin'] for r in rows]):+.3f} "
            f"ETA={eta:.0f}s",
            flush=True,
        )

    metrics = {}
    for key in ("target", "source", "margin", "delta_norm",
                "oracle_target", "oracle_source", "oracle_margin"):
        metrics[key] = ci_bootstrap([r[key] for r in rows], args.n_boot)

    metrics["n_pairs"] = len(rows)
    metrics["checkpoint"] = args.checkpoint
    metrics["epoch"] = ck.get("epoch")
    metrics["tau"] = tau
    metrics["oracle_ratio"] = (
        metrics["target"]["mean"] / metrics["oracle_target"]["mean"]
        if metrics["oracle_target"]["mean"] else 0.0
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"summary": metrics, "per_pair": rows}
    out_path = RESULTS_DIR / args.output
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\nSummary")
    print(f"{'metric':<15} {'mean':>8} {'CI_lo':>8} {'CI_hi':>8} {'std':>8}")
    print("-" * 55)
    for key in ("target", "source", "margin", "delta_norm",
                "oracle_target", "oracle_source", "oracle_margin"):
        m = metrics[key]
        print(f"{key:<15} {m['mean']:>8.3f} {m['ci_lo']:>8.3f} {m['ci_hi']:>8.3f} {m['std']:>8.3f}")
    print(f"oracle_ratio={metrics['oracle_ratio']:.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3c full evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/phase3c/best.pt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output", default="phase3c_full_eval.json")
    eval_full(parser.parse_args())
