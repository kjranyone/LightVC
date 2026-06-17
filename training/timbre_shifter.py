"""
Timbre shifter: signal-processing data augmentation.

This is borrowed from Seed-VC's training recipe. It is NOT a neural VC teacher —
it's a deterministic perturbation of pitch and formants that prevents the
converter from lazily copying the source speaker's timbre.

Uses pyworld for pitch-synchronous overlap-add (PSOLA) pitch shifting and
spectral envelope warping for formant shifting — the same approach as
Seed-VC's original recipe.

Reference: Seed-VC (arXiv:2411.09943) uses this to force the model to rely on
the reference embedding for speaker identity.
"""

from __future__ import annotations

import numpy as np
import pyworld as pw
import torch


def _ensure_double(wav: np.ndarray) -> np.ndarray:
    return wav.astype(np.double)


def psola_pitch_shift(
    wav: np.ndarray, sr: int, ratio: float, frame_period: float = 5.0
) -> np.ndarray:
    """True PSOLA pitch shifting via pyworld.

    Decomposes the signal into F0, spectral envelope, and aperiodicity,
    scales F0 by ``ratio``, then resynthesises. Duration is preserved.
    """
    if abs(ratio - 1.0) < 0.01:
        return wav

    wav_d = _ensure_double(wav)

    f0, t = pw.dio(wav_d, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav_d, f0, t, sr)
    sp = pw.cheaptrick(wav_d, f0, t, sr)
    ap = pw.d4c(wav_d, f0, t, sr)

    f0_shifted = f0 * ratio
    f0_shifted[np.isnan(f0_shifted)] = 0.0

    shifted = pw.synthesize(f0_shifted, sp, ap, sr, frame_period=frame_period)
    return shifted[: len(wav)].astype(np.float32)


def formant_shift(
    wav: np.ndarray, sr: int, shift_ratio: float, frame_period: float = 5.0
) -> np.ndarray:
    """Formant shifting via spectral envelope warping in log-frequency space.

    Decomposes with pyworld, warps the spectral envelope on the
    mel-frequency axis, then resynthesises.
    """
    if abs(shift_ratio - 1.0) < 0.02:
        return wav

    wav_d = _ensure_double(wav)

    f0, t = pw.dio(wav_d, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav_d, f0, t, sr)
    sp = pw.cheaptrick(wav_d, f0, t, sr)
    ap = pw.d4c(wav_d, f0, t, sr)

    fft_size = sp.shape[1]
    freqs = np.arange(fft_size) * (sr / (2.0 * (fft_size - 1)))
    mel_freqs = 1127.0 * np.log1p(freqs / 700.0)
    shifted_mel = mel_freqs / shift_ratio
    shifted_hz = 700.0 * (np.expm1(shifted_mel / 1127.0))

    indices = np.clip(
        (shifted_hz / (sr / 2.0) * (fft_size - 1)).astype(int),
        0,
        fft_size - 1,
    )
    sp_warped = np.ascontiguousarray(sp[:, indices])

    shifted = pw.synthesize(f0, sp_warped, ap, sr, frame_period=frame_period)
    return shifted[: len(wav)].astype(np.float32)


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

    pitch_ratio = np.random.uniform(*pitch_ratio_range)
    wav_np = psola_pitch_shift(wav_np, sr, pitch_ratio)

    formant_ratio = np.random.uniform(*formant_shift_range)
    wav_np = formant_shift(wav_np, sr, formant_ratio)

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
