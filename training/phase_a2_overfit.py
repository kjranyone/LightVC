import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from converter import FlowConverter, ConverterConfig
from infer_flow import encode, load_dac

DEVICE = torch.device("cuda")

def timbre_shift(wav, sr=44100):
    import librosa
    shifted = librosa.effects.pitch_shift(wav, sr=sr, n_steps=5.0)
    shifted = librosa.effects.preemphasis(shifted, coef=0.97)
    return shifted

def main():
    print("=== A2: timbre_shift(src)→src overfit test ===\n")

    dac, device = load_dac()

    wavs = sorted(Path("../data/vctk_200/p225").glob("*.wav"))[:10]
    print(f"Using {len(wavs)} utterances from p225")

    pairs = []
    for wav_path in wavs:
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != 44100:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=44100)
        rem = len(wav) % 512
        if rem > 0:
            wav = np.pad(wav, (0, 512 - rem))
        if len(wav) < 44100:
            continue

        shifted = timbre_shift(wav)
        rem = len(shifted) % 512
        if rem > 0:
            shifted = np.pad(shifted, (0, 512 - rem))

        z_clean = encode(dac, wav, device).unsqueeze(0)
        z_shifted = encode(dac, shifted, device).unsqueeze(0)
        if z_clean.shape[-1] != z_shifted.shape[-1]:
            min_t = min(z_clean.shape[-1], z_shifted.shape[-1])
            z_clean = z_clean[:, :, :min_t]
            z_shifted = z_shifted[:, :, :min_t]
        pairs.append((z_shifted, z_clean))

    print(f"Pairs: {len(pairs)}")

    config = ConverterConfig(latent_dim=1024, hidden_dim=1024, n_conv_blocks=4,
                             speaker_embed_dim=256, enable_timbre=True,
                             n_timbre_tokens=32, n_attn_heads=8,
                             bottleneck_dim=256, time_embed_dim=128)
    model = FlowConverter(config).to(device)

    ckpt = torch.load("checkpoints/phase_b_utte_1024/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded warm-start checkpoint")

    model.eval()
    with torch.no_grad():
        z_0, z_1 = pairs[0]
        z_0, z_1 = z_0.to(device), z_1.to(device)
        t = torch.ones(1, device=device)
        ref = z_1
        v = model.forward_velocity(z_0, t, ref)
        v_target = z_1 - z_0
        fm_init = F.mse_loss(v, v_target).item()
        print(f"Initial FM loss: {fm_init:.4f}")
        print(f"v_target std: {v_target.std().item():.4f}")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.8, 0.99), weight_decay=0.01)
    model.train()

    print(f"\nTraining 1000 steps (overfit on {len(pairs)} pairs)...")
    for step in range(1, 1001):
        z_0, z_1 = pairs[(step - 1) % len(pairs)]
        z_0, z_1 = z_0.to(device), z_1.to(device)
        t = torch.rand(1, device=device)
        t_expand = t[:, None, None]
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        v_target = z_1 - z_0

        v_pred = model.forward_velocity(z_t, t, z_1)
        loss = F.mse_loss(v_pred, v_target)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        if step % 100 == 0:
            with torch.no_grad():
                v_at_t1 = model.forward_velocity(z_0, torch.ones(1, device=device), z_1)
                fm_t1 = F.mse_loss(v_at_t1, v_target).item()
                cos = F.cosine_similarity(v_at_t1.flatten().unsqueeze(0),
                                          v_target.flatten().unsqueeze(0), dim=-1).item()
            print(f"step {step}: fm={loss.item():.4f}  fm(t=1)={fm_t1:.4f}  cos(v,v_tgt)={cos:.4f}")

    print(f"\n=== A2 Result ===")
    print(f"Initial FM: {fm_init:.4f} → Final FM: {loss.item():.4f}")
    if loss.item() < fm_init * 0.5:
        print("PASS: FM loss decreased significantly → FM works in DAC latent space")
        print("→ The Phase C failure was due to training objective, not FM itself")
    elif loss.item() < fm_init * 0.9:
        print("MARGINAL: FM loss decreased slightly → FM partially works")
    else:
        print("FAIL: FM loss did not decrease → FM itself may not work in DAC latent space")
        print("→ Consider C-5 (RVQ) or γ re-evaluation")

if __name__ == "__main__":
    main()
