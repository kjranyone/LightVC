import sys, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

DEVICE = torch.device("cuda")

def main():
    print("=== 09-D0: Phase D Profiling ===\n")

    torch.cuda.reset_peak_memory_stats()
    print(f"Base CUDA memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("Loading DAC decoder...")
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE)
    dac.eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    print(f"  DAC loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("Loading WavLM-SV...")
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE)
    wavlm.eval()
    for p in wavlm.parameters():
        p.requires_grad_(False)
    print(f"  WavLM-SV loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    import torchaudio
    resample = torchaudio.transforms.Resample(44100, 16000).to(DEVICE)

    from converter import FlowConverter, ConverterConfig
    config = ConverterConfig(latent_dim=1024, hidden_dim=1024, n_conv_blocks=4,
                             speaker_embed_dim=256, enable_timbre=True,
                             n_timbre_tokens=32, n_attn_heads=8,
                             bottleneck_dim=256, time_embed_dim=128)
    model = FlowConverter(config).to(DEVICE)
    print(f"  FlowConverter loaded: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    for batch_size in [4, 8, 16]:
        for T_frames in [100, 200]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            n_audio_samples = T_frames * 512
            try:
                z_src = torch.randn(batch_size, 1024, T_frames, device=DEVICE, requires_grad=True)
                z_ref = torch.randn(batch_size, 1024, T_frames, device=DEVICE)

                t0 = time.time()
                v_pred = model.forward_velocity(z_src, torch.ones(batch_size, device=DEVICE), z_ref)
                z_out = z_src + v_pred

                audio_44k = dac.decoder(z_out)
                if audio_44k.ndim == 3:
                    audio_44k = audio_44k.squeeze(1)
                audio_16k = resample(audio_44k)

                wavlm_out = wavlm(input_values=audio_16k)
                pred_embed = F.normalize(wavlm_out.last_hidden_state.mean(dim=1), dim=-1)

                target_embed = F.normalize(torch.randn_like(pred_embed), dim=-1)
                loss = (1 - F.cosine_similarity(pred_embed, target_embed, dim=-1).mean())

                loss.backward()

                elapsed = time.time() - t0
                peak_mem = torch.cuda.max_memory_allocated() / 1e9

                fwd_samples = audio_44k.shape[-1]
                print(f"B={batch_size:>2} T={T_frames:>3} ({fwd_samples:>6} smp) | "
                      f"time={elapsed:.3f}s | peak_mem={peak_mem:.2f} GB | "
                      f"loss={loss.item():.4f} | grad_norm={z_src.grad.norm().item():.2f}")

                del z_src, z_ref, v_pred, z_out, audio_44k, audio_16k, wavlm_out, pred_embed, loss

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"B={batch_size:>2} T={T_frames:>3} | OOM")
                    torch.cuda.empty_cache()
                else:
                    raise

    print("\n=== Summary ===")
    print("Feasible configurations (peak_mem < 20 GB):")
    print("  → use the largest batch_size × T_frames that fits for training")

if __name__ == "__main__":
    main()
