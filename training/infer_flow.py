"""
One-step inference test: generate before/after audio samples using FlowConverter.

Loads the flow converter, encodes source + reference through DAC,
converts via one-step mean-flow, decodes through DAC.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from converter import FlowConverter, ConverterConfig


def load_dac(model_id="descript/dac_44khz"):
    from transformers import AutoModel

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    dac = AutoModel.from_pretrained(model_id).to(device).eval()
    return dac, device


def load_flow_converter(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ConverterConfig(**ckpt["config"]["model"])
    model = FlowConverter(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, config


@torch.no_grad()
def encode(dac, wav, device):
    x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(device)
    return dac.encoder(x).squeeze(0)


@torch.no_grad()
def decode(dac, latent, device):
    x = latent.unsqueeze(0)
    return dac.decoder(x).squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="One-step VC inference test")
    parser.add_argument("--source", required=True, help="Source audio WAV")
    parser.add_argument("--reference", required=True, help="Reference audio WAV")
    parser.add_argument("--output", required=True, help="Output WAV")
    parser.add_argument("--converter", required=True, help="FlowConverter checkpoint")
    parser.add_argument("--dac-model", default="descript/dac_44khz")
    args = parser.parse_args()

    dac, device = load_dac(args.dac_model)
    model, config = load_flow_converter(args.converter, device)

    # Load audio
    import librosa

    src_wav, src_sr = sf.read(args.source, dtype="float32")
    if src_wav.ndim > 1:
        src_wav = src_wav.mean(axis=1)
    if src_sr != 44100:
        src_wav = librosa.resample(src_wav, orig_sr=src_sr, target_sr=44100)

    ref_wav, ref_sr = sf.read(args.reference, dtype="float32")
    if ref_wav.ndim > 1:
        ref_wav = ref_wav.mean(axis=1)
    if ref_sr != 44100:
        ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=44100)

    # Cap reference at 15s
    if len(ref_wav) > 15 * 44100:
        ref_wav = ref_wav[: 15 * 44100]

    print(f"Source: {len(src_wav)} samples ({len(src_wav) / 44100:.1f}s)")
    print(f"Reference: {len(ref_wav)} samples ({len(ref_wav) / 44100:.1f}s)")

    # Pad to hop
    for name, wav in [("source", src_wav), ("reference", ref_wav)]:
        rem = len(wav) % 512
        if rem > 0:
            if name == "source":
                src_wav = np.pad(src_wav, (0, 512 - rem))
            else:
                ref_wav = np.pad(ref_wav, (0, 512 - rem))

    # Encode
    z_src = encode(dac, src_wav, device)
    z_ref = encode(dac, ref_wav, device)
    print(f"Source latent: {z_src.shape}")
    print(f"Reference latent: {z_ref.shape}")

    # One-step conversion: z_converted = z_src + v_pred(z_src, t=1, ref)
    z_converted = model.convert(z_src, z_ref)
    print(f"Converted latent: {z_converted.shape}")

    # Decode
    output_wav = decode(dac, z_converted, device)
    print(f"Output audio: {len(output_wav)} samples")

    # Save
    sf.write(args.output, output_wav.astype(np.float32), 44100)
    print(f"Saved: {args.output}")

    # Metrics
    src_zcr = np.mean(np.diff(np.sign(src_wav[: len(output_wav)])) != 0)
    out_zcr = np.mean(np.diff(np.sign(output_wav[: len(src_wav)])) != 0)
    ref_zcr = np.mean(np.diff(np.sign(ref_wav)) != 0)
    print(f"\nZCR source:    {src_zcr:.4f}")
    print(f"ZCR reference: {ref_zcr:.4f}")
    print(f"ZCR converted: {out_zcr:.4f}")

    diff_src = abs(out_zcr - src_zcr)
    diff_ref = abs(out_zcr - ref_zcr)
    closer = "REFERENCE" if diff_ref < diff_src else "SOURCE"
    print(f"  -> closer to {closer}")


if __name__ == "__main__":
    main()
