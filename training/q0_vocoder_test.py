import json, torch, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import soundfile as sf
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from bigvgan import BigVGAN
from bigvgan.env import AttrDict
from bigvgan import mel_spectrogram

DEVICE = torch.device("cuda")

def load_bigvgan():
    model_dir = snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x")
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    h = AttrDict(config)
    if h.fmax is None:
        h.fmax = h.sampling_rate // 2
    m = BigVGAN(h, use_cuda_kernel=False)
    weights_path = os.path.join(model_dir, "bigvgan_generator.pt")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "generator" in state:
        state = state["generator"]
    clean = {}
    for k, v in state.items():
        if not k.startswith("_"):
            clean[k] = v
    missing, unexpected = m.load_state_dict(clean, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, h

def main():
    print("=== Q0: Vocoder Upper Bound ===\n")
    print("Loading BigVGAN v2...")
    vocoder, h = load_bigvgan()
    print(f"  SR={h.sampling_rate} Mel={h.num_mels} Hop={h.hop_size} Win={h.win_size}")
    print(f"  fmin={h.fmin} fmax={h.fmax}")
    print(f"  Params: {sum(p.numel() for p in vocoder.parameters())/1e6:.1f}M")

    import librosa
    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    def get_secs_embed(wav_44k):
        wav16k = librosa.resample(wav_44k.astype(np.float32), orig_sr=44100, target_sr=16000)
        t = torch.from_numpy(wav16k).float().unsqueeze(0).to(DEVICE)
        return secs_model.encode_batch(t).squeeze().cpu()

    test_files = sorted(Path("../data/vctk_200/p225").glob("*.wav"))[:5]
    print(f"\nTest files: {len(test_files)}")

    secs_scores = []
    for wav_path in test_files:
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != h.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=h.sampling_rate)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
        if wav_t.shape[-1] % h.hop_size != 0:
            wav_t = wav_t[..., :wav_t.shape[-1] // h.hop_size * h.hop_size]

        with torch.no_grad():
            mel = mel_spectrogram(
                wav_t, h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax,
            )
            recon = vocoder(mel).squeeze(0).squeeze(0).cpu().numpy()

        real_embed = get_secs_embed(wav)
        recon_embed = get_secs_embed(recon)
        secs = F.cosine_similarity(real_embed.unsqueeze(0), recon_embed.unsqueeze(0), dim=-1).item()
        secs_scores.append(secs)

        mel_flat = mel.squeeze(0)
        print(f"  {wav_path.name}: mel={mel_flat.shape}, "
              f"recon_dur={len(recon)/h.sampling_rate:.2f}s, "
              f"SECS(self-recon)={secs:.4f}")

    print(f"\n=== Q0 Result ===")
    print(f"Vocoder self-reconstruction SECS: {np.mean(secs_scores):.4f}")
    print(f"Target: > 0.90 (near-perfect reconstruction)")
    if np.mean(secs_scores) > 0.90:
        print("PASS: Vocoder faithfully reconstructs speaker identity")
    elif np.mean(secs_scores) > 0.70:
        print("MARGINAL: Some speaker information lost in mel/vocoder roundtrip")
    else:
        print("FAIL: Vocoder loses significant speaker information")

if __name__ == "__main__":
    main()
