from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from causal_codec import CausalCodec, LATENT_DIM, architecture_stats


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[index]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="../results/ys1/s1_0_probe.json")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    codec = CausalCodec().eval()
    z = torch.randn(1, LATENT_DIM, args.frames)
    stream = codec.decoder.stream()

    with torch.inference_mode():
        full = codec.decode(z)
        stepped = stream.decode_chunk(z)
        parity = float((full - stepped).abs().max())
        for _ in range(args.warmup):
            stream.reset()
            stream.decode_chunk(z)
        elapsed: list[float] = []
        for _ in range(args.runs):
            stream.reset()
            start = time.perf_counter()
            stream.decode_chunk(z)
            elapsed.append((time.perf_counter() - start) * 1000 / args.frames)

    stats = architecture_stats(codec)
    report = {
        "schema": 1,
        "sample_rate": 48000,
        "hop_length": 480,
        "latent_dim": LATENT_DIM,
        "frames_per_run": args.frames,
        "threads": args.threads,
        "host": platform.node(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "encoder_parameters": stats.encoder_parameters,
        "decoder_parameters": stats.decoder_parameters,
        "total_parameters": stats.total_parameters,
        "decoder_macs_per_second": stats.decoder_macs_per_second,
        "full_step_max_abs": parity,
        "step_ms_p50": statistics.median(elapsed),
        "step_ms_p95": percentile(elapsed, 0.95),
        "step_ms_p99": percentile(elapsed, 0.99),
        "step_ms_max": max(elapsed),
        "rtf_mean": statistics.mean(elapsed) / 10.0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
