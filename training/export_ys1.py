"""Y-S1 codec decoder の重み export（Rust/Candle 用・flat binary）。

s1_3 ckpt（EMA 優先）から decoder のみ抽出し、Rust 実装が読みやすい
逐次 layout の .bin + manifest JSON へ。weight_norm は本 model では未使用
（plain Conv1d）なのでそのまま。

    CUDA_VISIBLE_DEVICES="" uv run python export_ys1.py
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"
STAGES = [(512, 256, 3), (256, 128, 4), (128, 64, 5), (64, 32, 8)]  # 逆順実行
DILS = (1, 3, 9)


def main() -> int:
    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location="cpu")
    net = ck.get("ema") or ck["net"]
    OUT.mkdir(exist_ok=True)
    buf = bytearray()
    manifest: list[tuple[str, list[int], int, int]] = []   # (key, shape, offset, len)

    def put(key: str, t: torch.Tensor):
        t = t.detach().float().contiguous()
        off = len(buf)
        buf.extend(struct.pack(f"<{t.numel()}f", *t.flatten().tolist()))
        manifest.append((key, list(t.shape), off, t.numel()))

    put("decoder.pre.weight", net["decoder.pre.weight"])
    put("decoder.pre.bias", net["decoder.pre.bias"])
    for i, (cin, cout, stride) in enumerate(STAGES):
        put(f"decoder.stages.{i}.up.weight", net[f"decoder.stages.{i}.up.weight"])
        put(f"decoder.stages.{i}.up.bias", net[f"decoder.stages.{i}.up.bias"])
        for j, d in enumerate(DILS):
            for part in ("act1.alpha", "act1.beta", "conv1.weight", "conv1.bias",
                         "act2.alpha", "act2.beta", "conv2.weight", "conv2.bias"):
                put(f"decoder.stages.{i}.res.{j}.{part}",
                    net[f"decoder.stages.{i}.res.{j}.{part}"])
    put("decoder.post_act.alpha", net["decoder.post_act.alpha"])
    put("decoder.post_act.beta", net["decoder.post_act.beta"])
    put("decoder.post.weight", net["decoder.post.weight"])
    put("decoder.post.bias", net["decoder.post.bias"])

    binp = OUT / "ys1_decoder.bin"
    binp.write_bytes(bytes(buf))
    meta = {
        "schema": 1,
        "sample_rate": 48000,
        "hop_length": 480,
        "latent_dim": 32,
        "strides": [8, 5, 4, 3],
        "channels": [32, 64, 128, 256, 512],
        "snake_log_domain": True,
        "causal": True,
        "source_ckpt": "results/s1_3_c32/s1_3_c32_last.pt (ema)",
        "step": int(ck.get("step", -1)),
        "tensors": [{"key": k, "shape": s, "offset": o, "count": n}
                    for k, s, o, n in manifest],
    }
    (OUT / "ys1_decoder.json").write_text(json.dumps(meta, indent=1))
    print(f"  {binp.name}: {len(buf)/1e6:.1f} MB / {len(manifest)} tensors"
          f" (step {meta['step']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
