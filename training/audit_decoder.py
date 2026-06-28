"""
Step 1: decoder named_parameters audit
Step 2: frozen full vs frozen short alignment audit

Usage:
  cd training
  uv run python audit_decoder.py --n_utts 20
"""
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR, load_dac

VCTK_WAV = Path("../data/vctk_200")
HOP = 512


def load_wav_44k(p):
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def audit_named_parameters(dac):
    print("=== Step 1: Decoder named_parameters audit ===\n")
    target_prefix = ("decoder.block.2.", "decoder.block.3.")
    trainable = []
    frozen = []
    for name, p in dac.named_parameters():
        if any(name.startswith(pre) for pre in target_prefix):
            trainable.append((name, p))
        else:
            frozen.append((name, p))

    print(f"Trainable (block.2 + block.3): {sum(p.numel() for _, p in trainable):,} params")
    for name, p in trainable[:5]:
        print(f"  {name} {list(p.shape)}")
    if len(trainable) > 5:
        print(f"  ... and {len(trainable) - 5} more")
    print(f"\nFrozen: {sum(p.numel() for _, p in frozen):,} params")
    print(f"Total decoder params: {sum(p.numel() for _, p in trainable + frozen):,}")
    return trainable


def aligned_snr(ref, est, max_lag=200):
    if len(ref) < len(est):
        ref = np.pad(ref, (0, len(est) - len(ref)))
    elif len(est) < len(ref):
        est = np.pad(est, (0, len(ref) - len(est)))
    corr = np.correlate(ref, est, mode="full")
    center = len(ref) - 1
    best_lag = np.argmax(np.abs(corr[center - max_lag:center + max_lag + 1])) - max_lag
    est_shifted = np.roll(est, best_lag)
    noise = ref - est_shifted
    signal_power = np.sum(ref ** 2) + 1e-10
    noise_power = np.sum(noise ** 2) + 1e-10
    snr = 10 * np.log10(signal_power / noise_power)
    return snr, best_lag


def audit_alignment(dac, n_utts=20):
    print("\n=== Step 2: Alignment audit (frozen full vs frozen short) ===\n")
    dac.eval()

    wavs = sorted(VCTK_WAV.glob("*/*.wav"))
    rng = np.random.default_rng(42)
    selected = rng.choice(len(wavs), size=min(n_utts, len(wavs)), replace=False)

    results = defaultdict(lambda: {"snr": [], "lag": [], "snr_noalign": []})

    for idx, wi in enumerate(selected):
        wav = load_wav_44k(wavs[wi])
        if len(wav) < DAC_SR * 3:
            continue

        x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z = dac.encoder(x)
        T = z.shape[2]

        with torch.no_grad():
            audio_full = dac.decoder(z).squeeze().cpu().numpy()

        for w in [4, 8]:
            if w > T - 1:
                continue
            for s in [T // 4]:
                z_chunk = z[:, :, s:s + w]
                with torch.no_grad():
                    audio_short = dac.decoder(z_chunk).squeeze().cpu().numpy()

                start_sample = s * HOP
                end_sample = (s + w) * HOP
                ref_region = audio_full[start_sample:end_sample]

                min_len = min(len(ref_region), len(audio_short))
                ref_region = ref_region[:min_len]
                audio_short = audio_short[:min_len]

                snr_noalign, _ = aligned_snr(ref_region, audio_short, max_lag=0)
                snr_aligned, lag = aligned_snr(ref_region, audio_short, max_lag=200)

                results[w]["snr"].append(snr_aligned)
                results[w]["lag"].append(lag)
                results[w]["snr_noalign"].append(snr_noalign)

        print(f"  [{idx+1}/{len(selected)}] {wavs[wi].stem} T={T}", flush=True)

    print(f"\n{'window':<8} {'SNR unaligned':>14} {'SNR aligned':>12} {'mean lag':>10} {'std lag':>10} {'n':>4}")
    print("-" * 60)
    for w in sorted(results.keys()):
        r = results[w]
        print(
            f"{w}f      "
            f"{np.mean(r['snr_noalign']):>13.1f}dB "
            f"{np.mean(r['snr']):>11.1f}dB "
            f"{np.mean(r['lag']):>10.1f} "
            f"{np.std(r['lag']):>10.1f} "
            f"{len(r['snr']):>4}"
        )

    all_lags_4f = results[4]["lag"] if 4 in results else []
    if all_lags_4f:
        mode_lag = int(np.median(all_lags_4f))
        print(f"\nMedian lag (4f): {mode_lag} samples ({mode_lag / DAC_SR * 1000:.2f}ms)")
        print(f"Lag range: [{min(all_lags_4f)}, {max(all_lags_4f)}]")
        if abs(mode_lag) > 10:
            print(f"WARNING: lag > 10 samples. Use fixed lag={mode_lag} in training.")
        else:
            print(f"OK: lag < 10 samples. Fixed lag={mode_lag} recommended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder audit: params + alignment")
    parser.add_argument("--n_utts", type=int, default=20)
    args = parser.parse_args()

    dac = load_dac()
    audit_named_parameters(dac)
    audit_alignment(dac, args.n_utts)
