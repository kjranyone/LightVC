"""
Generate reference tensors for Rust/Candle parity testing.
Saves z_s, q0_s, timbre, z_q (soft RVQ), z_q_adapted to safetensors.
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))

from train_phase3b import (
    DEVICE, load_dac, soft_rvq_requantize,
)
from train_phase3c_adapter import TimbreAdapter

TAU = 5.0
EVAL_PAIR = Path("../data/phase3_10k/eval/pair_00000.pt")
ADAPTER_CKPT = Path("checkpoints/phase3c_ao_b1_ecapa/best.pt")
OUTPUT = Path("../models/rust_parity_ref.safetensors")


def main():
    print("=== Generating Rust Parity Reference ===\n")

    d = torch.load(EVAL_PAIR, map_location="cpu")
    z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
    q0_s = d["q0_s"].float().unsqueeze(0).to(DEVICE)
    timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)
    print(f"Loaded {EVAL_PAIR.name}: z_s={z_s.shape}, q0_s={q0_s.shape}, timbre={timbre.shape}")

    dac = load_dac()
    print("DAC loaded")

    ck = torch.load(ADAPTER_CKPT, map_location="cpu", weights_only=False)
    ca = ck["args"]
    adapter = TimbreAdapter(
        latent_dim=1024, timbre_dim=192,
        bottleneck=ca.get("bottleneck", 256),
        kernel=ca.get("kernel", 3),
        n_blocks=ca.get("n_blocks", 1),
        utte_mode=ca.get("utte_mode", "none").replace("ecpa", "ecapa") if ca.get("utte_mode", "none") == "ecpa" else ca.get("utte_mode", "none"),
        film_mode=ca.get("film_mode", "full"),
        n_tokens=ca.get("n_tokens", 32),
        n_heads=ca.get("n_heads", 4),
    ).to(DEVICE)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    print(f"Adapter loaded (epoch={ck['epoch']}, utte={ca.get('utte_mode')})")

    with torch.no_grad():
        z_q = soft_rvq_requantize(dac, q0_s, z_s, TAU)
        z_q_adapted = adapter(z_q, timbre)
        audio_ref = dac.decoder(z_q_adapted)

    print(f"\nz_q:          {z_q.shape}  norm={z_q.norm().item():.2f}")
    print(f"z_q_adapted:  {z_q_adapted.shape}  norm={z_q_adapted.norm().item():.2f}")
    print(f"audio_ref:    {audio_ref.shape}  norm={audio_ref.norm().item():.2f}")
    delta = z_q_adapted - z_q
    print(f"delta:        norm={delta.norm().item():.4f}")

    tensors = {
        "z_s": z_s.cpu(),
        "q0_s": q0_s.cpu(),
        "timbre": timbre.cpu(),
        "z_q_ref": z_q.cpu(),
        "z_q_adapted_ref": z_q_adapted.cpu(),
        "audio_ref": audio_ref.cpu(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUTPUT))
    print(f"\nSaved: {OUTPUT}")
    print(f"T={z_s.shape[2]}")


if __name__ == "__main__":
    main()
