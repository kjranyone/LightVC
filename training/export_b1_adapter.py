"""
Export B1 UTTE adapter weights to safetensors for Rust/Candle.

Splits nn.MultiheadAttention in_proj into separate Q/K/V tensors.
Skips unused FiLM params (film_mode="none" in B1).
"""
import sys
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))


def export(args):
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["adapter"]
    ck_args = ck.get("args", {})
    bottleneck = ck_args.get("bottleneck", 256)
    n_tokens = ck_args.get("n_tokens", 32)
    n_heads = ck_args.get("n_heads", 4)
    n_blocks = ck_args.get("n_blocks", 1)

    print(f"checkpoint: {args.checkpoint}")
    print(f"epoch: {ck.get('epoch')}")
    print(f"bottleneck={bottleneck} n_tokens={n_tokens} n_heads={n_heads} n_blocks={n_blocks}")

    out = {}

    for b in range(n_blocks):
        prefix = f"blocks.{b}." if n_blocks > 1 else ""

        for name in ("conv_in", "conv_out"):
            for suffix in ("weight", "bias"):
                key = f"blocks.{b}.{name}.{suffix}"
                if key in sd:
                    out[f"{prefix}{name}.{suffix}"] = sd[key].contiguous().half()

        for name in ("film_gamma", "film_beta"):
            for suffix in ("weight", "bias"):
                key = f"blocks.{b}.{name}.{suffix}"
                if key in sd and ck_args.get("film_mode", "full") == "full":
                    out[f"{prefix}{name}.{suffix}"] = sd[key].contiguous().half()

    if "ecapa_to_tokens.weight" in sd:
        out["token_mlp.weight"] = sd["ecapa_to_tokens.weight"].contiguous().half()
        out["token_mlp.bias"] = sd["ecapa_to_tokens.bias"].contiguous().half()

    if "token_proj.weight" in sd:
        out["token_proj.weight"] = sd["token_proj.weight"].contiguous().half()
        out["token_proj.bias"] = sd["token_proj.bias"].contiguous().half()

    if "cross_attn.in_proj_weight" in sd:
        ipw = sd["cross_attn.in_proj_weight"]
        ipb = sd["cross_attn.in_proj_bias"]
        d = bottleneck
        out["attn.q.weight"] = ipw[:d].contiguous().half()
        out["attn.q.bias"] = ipb[:d].contiguous().half()
        out["attn.k.weight"] = ipw[d:2*d].contiguous().half()
        out["attn.k.bias"] = ipb[d:2*d].contiguous().half()
        out["attn.v.weight"] = ipw[2*d:].contiguous().half()
        out["attn.v.bias"] = ipb[2*d:].contiguous().half()
        out["attn.o.weight"] = sd["cross_attn.out_proj.weight"].contiguous().half()
        out["attn.o.bias"] = sd["cross_attn.out_proj.bias"].contiguous().half()

    print(f"\nExported {len(out)} tensors:")
    for k, v in sorted(out.items()):
        print(f"  {k:30s} {str(list(v.shape)):20s} {v.dtype}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_path))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export B1 UTTE adapter to safetensors")
    parser.add_argument("--checkpoint", default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    parser.add_argument("--output", default="../models/utte_adapter_b1.safetensors")
    export(parser.parse_args())
