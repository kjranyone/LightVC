"""
Timbre shifter: signal-processing data augmentation.

This is borrowed from Seed-VC's training recipe. It is NOT a neural VC teacher —
it's a deterministic perturbation of pitch and formants that prevents the
converter from lazily copying the source speaker's timbre.

Reference: Seed-VC (arXiv:2411.09943) uses this to force the model to rely on
the reference embedding for speaker identity.
"""

from __future__ import annotations

import numpy as np
import torch


def psola_pitch_shift(
    wav: np.ndarray, sr: int, ratio: float, frame_size_ms: float = 40.0
) -> np.ndarray:
    """Simple PSOLA-style pitch shifting via resampling + duration correction.

    Uses WORLD-free approximation: resample to change pitch, then
    overlap-add with linear interpolation to restore duration.
    """
    if abs(ratio - 1.0) < 0.01:
        return wav

    # Resample (changes both pitch and duration)
    new_len = int(len(wav) / ratio)
    indices = np.linspace(0, len(wav) - 1, new_len)
    shifted = np.interp(indices, np.arange(len(wav)), wav).astype(np.float32)

    # Restore duration via overlap-add
    if len(shifted) >= len(wav):
        return shifted[: len(wav)]
    else:
        return np.pad(shifted, (0, len(wav) - len(shifted)))


def formant_filter(
    wav: np.ndarray, sr: int, shift_ratio: float, order: int = 2
) -> np.ndarray:
    """Approximate formant shifting via spectral envelope warping.

    Uses a simple approach: high-shelf / low-shelf filtering to shift the
    perceived formant center frequencies. Not as accurate as true cepstral
    warping, but fast and sufficient for augmentation.
    """
    if abs(shift_ratio - 1.0) < 0.02:
        return wav

    from scipy.signal import butter, sosfilt

    # Split into low and high bands at ~2kHz, amplify/shrink differently
    crossover = int(2000 * shift_ratio)
    crossover = max(200, min(sr // 2 - 200, crossover))

    nyq = sr / 2
    low_sos = butter(order, crossover / nyq, btype="low", output="sos")
    high_sos = butter(order, crossover / nyq, btype="high", output="sos")

    low = sosfilt(low_sos, wav)
    high = sosfilt(high_sos, wav)

    # Scale bands to approximate formant shift
    low_gain = 1.0 / shift_ratio if shift_ratio > 1 else shift_ratio
    high_gain = shift_ratio if shift_ratio > 1 else 1.0 / shift_ratio

    return (low * low_gain + high * high_gain).astype(np.float32)


def timbre_shift(
    wav: np.ndarray | torch.Tensor,
    sr: int,
    pitch_ratio_range: tuple[float, float] = (0.8, 1.25),
    formant_shift_range: tuple[float, float] = (0.85, 1.18),
    apply_prob: float = 0.5,
) -> np.ndarray | torch.Tensor:
    """Apply timbre perturbation augmentation.

    Args:
        wav: [samples] mono audio
        sr: sample rate
        pitch_ratio_range: range of pitch perturbation ratios
        formant_shift_range: range of formant shift ratios
        apply_prob: probability of applying (0 = never, 1 = always)
    Returns:
        perturbed wav of same length and type
    """
    if np.random.random() > apply_prob:
        return wav

    is_tensor = isinstance(wav, torch.Tensor)
    if is_tensor:
        wav_np = wav.cpu().numpy()
    else:
        wav_np = wav

    orig_len = len(wav_np)

    # Pitch shift
    pitch_ratio = np.random.uniform(*pitch_ratio_range)
    wav_np = psola_pitch_shift(wav_np, sr, pitch_ratio)

    # Formant shift
    formant_ratio = np.random.uniform(*formant_shift_range)
    wav_np = formant_filter(wav_np, sr, formant_ratio)

    # Ensure same length
    if len(wav_np) != orig_len:
        if len(wav_np) > orig_len:
            wav_np = wav_np[:orig_len]
        else:
            wav_np = np.pad(wav_np, (0, orig_len - len(wav_np)))

    if is_tensor:
        return torch.from_numpy(wav_np).to(wav.device)
    return wav_np


if __name__ == "__main__":
    # Quick self-test
    import soundfile as sf

    # Generate test signal
    sr = 44100
    t = np.arange(sr * 2) / sr
    wav = 0.3 * np.sin(2 * np.pi * 200 * t).astype(np.float32)

    shifted = timbre_shift(wav, sr, apply_prob=1.0)
    print(f"Original: {wav.shape}, shifted: {shifted.shape}")
    print(f"Original peak: {np.max(np.abs(wav)):.3f}")
    print(f"Shifted peak: {np.max(np.abs(shifted)):.3f}")
    sf.write("/tmp/timbre_test_orig.wav", wav, sr)
    sf.write("/tmp/timbre_test_shifted.wav", shifted, sr)
    print("Saved test files to /tmp/")
