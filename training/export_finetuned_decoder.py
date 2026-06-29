"""
Export fine-tuned DAC decoder as a complete safetensors file.

Merges fine-tuned block.2/block.3 weights into the original DAC model,
then saves the complete model as models/dac_44khz_finetuned.safetensors.

Also generates a PyTorch reference output for Rust parity testing.

Usage:
  cd training
  uv run python export_finetuned_decoder.py
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR, load_dac


def export_merged_weights(ckpt_path, output_path):
    print("=== Export Fine-Tuned Decoder ===\n")
    dac = load_dac()

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    delta = ck["decoder_state"]
    print(f"Fine-tune checkpoint: {ckpt_path}")
    print(f"Delta tensors: {len(delta)}")

    full_sd = dac.state_dict()
    n_patched = 0
    for k, v in delta.items():
        if k in full_sd:
            full_sd[k] = v.cpu()
            n_patched += 1
    print(f"Patched: {n_patched} tensors")

    dac.load_state_dict(full_sd)
    dac.eval()

    export_sd = {}
    for k, v in full_sd.items():
        export_sd[k] = v.contiguous().cpu()
    save_file(export_sd, str(output_path))
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    return dac


def generate_parity_ref(dac, output_path, T=32):
    print(f"\nGenerating parity reference (T={T})...")
    torch.manual_seed(42)
    z = torch.randn(1, 1024, T, device=DEVICE)

    with torch.no_grad():
        audio = dac.decoder(z).squeeze().cpu().numpy()

    np.save(str(output_path), audio)
    torch.save(z.cpu(), str(output_path.with_suffix(".pt")))
    print(f"Reference audio: {output_path} ({audio.shape})")
    print(f"Reference latent: {output_path.with_suffix('.pt')} ({z.shape})")
    print(f"Audio mean={audio.mean():.6f} std={audio.std():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/decoder_finetune/best.pt")
    parser.add_argument("--output", default="../models/dac_44khz_finetuned.safetensors")
    parser.add_argument("--ref_output", default="../results/decoder_finetuned_parity_ref.npy")
    args = parser.parse_args()

    output_path = Path(args.output)
    ref_path = Path(args.ref_output)

    dac = export_merged_weights(args.ckpt, output_path)
    generate_parity_ref(dac, ref_path)
