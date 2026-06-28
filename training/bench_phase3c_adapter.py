"""
Microbenchmark Phase 3c TimbreAdapter inference cost.
"""
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from train_phase3b import DEVICE
from train_phase3c_adapter import TimbreAdapter


RESULTS_DIR = Path("../results")


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def load_adapter(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    adapter = TimbreAdapter(
        latent_dim=1024,
        timbre_dim=192,
        bottleneck=args.get("bottleneck", 256),
        kernel=args.get("kernel", 3),
        n_blocks=args.get("n_blocks", 1),
        utte_mode=("ecapa" if args.get("utte_mode", "none") == "ecpa" else args.get("utte_mode", "none")),
        film_mode=args.get("film_mode", "full"),
        n_tokens=args.get("n_tokens", 32),
        n_heads=args.get("n_heads", 4),
    ).to(DEVICE)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    return args, adapter


@torch.no_grad()
def bench(args):
    ck_args, adapter = load_adapter(args.checkpoint)
    print("=== Phase 3c Adapter Benchmark ===")
    print(f"device={DEVICE}")
    print(f"checkpoint={args.checkpoint}")
    print(
        "config: "
        f"bottleneck={ck_args.get('bottleneck')} "
        f"utte={ck_args.get('utte_mode')} "
        f"tokens={ck_args.get('n_tokens')} "
        f"heads={ck_args.get('n_heads')} "
        f"blocks={ck_args.get('n_blocks')}"
    )

    results = {}
    for frames in args.frames:
        z_q = torch.randn(args.batch_size, 1024, frames, device=DEVICE)
        timbre = torch.randn(args.batch_size, 192, device=DEVICE)

        for _ in range(args.warmup):
            _ = adapter(z_q, timbre, None)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elif DEVICE.type == "xpu":
            torch.xpu.synchronize()

        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            _ = adapter(z_q, timbre, None)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elif DEVICE.type == "xpu":
                torch.xpu.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        stats = {
            "mean_ms": float(np.mean(times)),
            "p50_ms": percentile(times, 50),
            "p95_ms": percentile(times, 95),
            "p99_ms": percentile(times, 99),
            "frames": frames,
            "batch_size": args.batch_size,
        }
        results[str(frames)] = stats
        print(
            f"T={frames:>4}: mean={stats['mean_ms']:.3f}ms "
            f"p50={stats['p50_ms']:.3f}ms p95={stats['p95_ms']:.3f}ms "
            f"p99={stats['p99_ms']:.3f}ms"
        )

    out = {
        "checkpoint": args.checkpoint,
        "device": str(DEVICE),
        "checkpoint_args": ck_args,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.output
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Phase 3c adapter")
    parser.add_argument("--checkpoint", default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    parser.add_argument("--frames", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--output", default="phase3c_adapter_bench.json")
    bench(parser.parse_args())
