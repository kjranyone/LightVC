"""
Phase D: Export PyTorch converter weights to safetensors.

Supports both Converter (warm-start) and FlowConverter (flow matching).
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

from converter import Converter, FlowConverter, ConverterConfig


def export(checkpoint_path, output_path, config_override=None, model_type="auto"):
    """Export converter checkpoint to safetensors."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if config_override:
        import yaml

        with open(config_override) as f:
            cfg = yaml.safe_load(f)
        config = ConverterConfig(**cfg["model"])
    elif "config" in ckpt:
        config = ConverterConfig(**ckpt["config"]["model"])
    else:
        config = ConverterConfig()

    # Auto-detect model type from checkpoint keys
    if model_type == "auto":
        if any("time_embed" in k for k in ckpt["model"].keys()):
            model_type = "flow"
        else:
            model_type = "converter"

    if model_type == "flow":
        model = FlowConverter(config)
        print("Model type: FlowConverter")
    else:
        model = Converter(config)
        print("Model type: Converter")

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Config: {config}")
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"State dict keys: {len(state_dict)}")

    save_file(state_dict, output_path)

    print(f"\nExported to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1e6:.1f} MB")

    config_path = output_path.replace(".safetensors", "_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "latent_dim": config.latent_dim,
                "hidden_dim": config.hidden_dim,
                "n_conv_blocks": config.n_conv_blocks,
                "speaker_embed_dim": config.speaker_embed_dim,
                "n_timbre_tokens": config.n_timbre_tokens,
                "n_attn_heads": config.n_attn_heads,
                "enable_timbre": config.enable_timbre,
                "bottleneck_dim": config.bottleneck_dim,
                "time_embed_dim": config.time_embed_dim,
                "n_depth_groups": config.n_depth_groups,
                "model_type": model_type,
            },
            f,
            indent=2,
        )
    print(f"Config: {config_path}")

    print("\nSample keys (first 10):")
    for k in list(state_dict.keys())[:10]:
        print(f"  {k}: {state_dict[k].shape} {state_dict[k].dtype}")


def main():
    parser = argparse.ArgumentParser(description="Export converter to safetensors")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--model-type",
        choices=["auto", "converter", "flow"],
        default="auto",
    )
    args = parser.parse_args()

    export(args.checkpoint, args.output, args.config, args.model_type)


if __name__ == "__main__":
    main()
