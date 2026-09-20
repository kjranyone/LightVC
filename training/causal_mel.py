"""Z0-A: left-aligned (0-lookahead) causal mel, ABI-matched to BigVGAN's mel
except framing. Same librosa slaney filterbank / hann / magnitude / log-clamp as
bigvgan.meldataset.mel_spectrogram; ONLY the padding/framing differs.

Framing (win = n_fft):
  centered (path A, = BigVGAN): pad (n_fft-hop)//2 both sides, center=False.
  causal   (path B):           pad (n_fft-hop) LEFT, 0 RIGHT, center=False.
  -> causal frame t covers original samples [t*hop-(n_fft-hop), t*hop+hop):
     a TRAILING window (n_fft of PAST) whose rightmost sample is t*hop+hop-1.
     Lookahead beyond the emitted block = 0 (only the current hop, unavoidable).

Reviewer point A: with a LEFT-aligned window, n_fft(=win) is the transient-smear
knob (2048 was tuned for centered). Z0-B sweeps n_fft in {512,1024,2048}.

Run `python causal_mel.py` for the parity check (centered == BigVGAN) and the
causality CI (frames whose window ends before T are invariant to y[T:]).
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

_fb_cache: dict = {}
_win_cache: dict = {}


def _mel_fb(sr, n_fft, num_mels, fmin, fmax, device):
    key = (sr, n_fft, num_mels, fmin, fmax, str(device))
    if key not in _fb_cache:
        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _fb_cache[key] = torch.from_numpy(fb).float().to(device)
    return _fb_cache[key]


def _win(win, device):
    key = (win, str(device))
    if key not in _win_cache:
        _win_cache[key] = torch.hann_window(win).to(device)
    return _win_cache[key]


def _mel(y, n_fft, hop, num_mels, sr, fmin, fmax, left_pad, right_pad):
    """Core: pad (left,right) reflect, center=False stft, slaney mel, log-clamp.
    Matches bigvgan.meldataset.mel_spectrogram magnitude/log exactly."""
    if y.dim() == 1:
        y = y.unsqueeze(0)
    win = n_fft
    device = y.device
    fb = _mel_fb(sr, n_fft, num_mels, fmin, fmax, device)
    window = _win(win, device)
    # ⚠ **零詰め**（2.1 の作るもの #9）。`reflect` の頭は t=0 で未来サンプルを読む
    #   （左寄せ運用では先読み `n_fft - hop`）。製品のブロック実行系は起動時に
    #   実サンプルだけを解析するので、学習側も同じ頭にする。
    yp = torch.nn.functional.pad(y.unsqueeze(1), (left_pad, right_pad)).squeeze(1)
    spec = torch.stft(yp, n_fft, hop_length=hop, win_length=win, window=window,
                      center=False, pad_mode="reflect", normalized=False,
                      onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel = torch.matmul(fb, spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def causal_mel(y, n_fft=2048, hop=128, num_mels=128, sr=44100, fmin=0, fmax=None):
    """Left-aligned, 0-lookahead. win=n_fft. Frame t rightmost sample = t*hop+hop-1."""
    return _mel(y, n_fft, hop, num_mels, sr, fmin, fmax, left_pad=n_fft - hop, right_pad=0)


def centered_mel(y, n_fft=2048, hop=128, num_mels=128, sr=44100, fmin=0, fmax=None):
    """BigVGAN-parity centered mel (path A). For verification only."""
    pad = (n_fft - hop) // 2
    return _mel(y, n_fft, hop, num_mels, sr, fmin, fmax, left_pad=pad, right_pad=pad)


def mel_la(y, n_fft=1024, hop=128, look=0, num_mels=128, sr=44100, fmin=0, fmax=None):
    """Mel with a tunable FUTURE lookahead of `look` samples (framing sweep).
    look=0 -> causal (0-lookahead). look=(n_fft-hop)//2 -> centered.
    Frame t rightmost sample = t*hop + hop - 1 + look; lookahead = look samples.
    """
    left = (n_fft - hop) - look
    assert 0 <= look <= (n_fft - hop), f"look {look} out of [0, {n_fft-hop}]"
    return _mel(y, n_fft, hop, num_mels, sr, fmin, fmax, left_pad=left, right_pad=look)


def abi_spec(n_fft=2048, hop=128, num_mels=128, sr=44100, fmin=0, fmax=None):
    """Z0-A: the frozen mel-analysis ABI (path B). Persist this next to freec_B."""
    return {
        "framing": "left_aligned_causal",
        "n_fft": n_fft, "win": n_fft, "hop": hop, "num_mels": num_mels,
        "sr": sr, "fmin": fmin, "fmax": fmax if fmax is not None else sr / 2,
        "window": "hann", "magnitude": "sqrt(re^2+im^2+1e-9)",
        "log": "log(clamp(x, min=1e-5))  # slaney mel, dynamic_range_compression",
        "pad": {"left": n_fft - hop, "right": 0, "mode": "reflect", "center": False},
        "frame_time_samples": "frame t reads original [t*hop-(n_fft-hop), t*hop+hop); rightmost = t*hop+hop-1",
        "lookahead_beyond_emitted_block": 0,
        "note": "synthesis grid (FreeC nfft/win/hop=256/256/128) is SEPARATE; do NOT build mel with n_fft=256",
    }


def _selfcheck():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    import json
    SNAP = Path("/home/kojirotanaka/.cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/"
                "snapshots/95a9d1dcb12906c03edd938d77b9333d6ded7dfb")
    h = AttrDict(json.loads((SNAP / "config.json").read_text()))
    h["hop_size"] = 128

    y = (torch.rand(1, 44100) * 2 - 1).to(dev) * 0.5

    # (1) parity: our centered_mel == bigvgan get_mel_spectrogram (hop128)
    m_ours = centered_mel(y, n_fft=2048, hop=128).cpu()
    m_bv = get_mel_spectrogram(y.cpu(), h)
    T = min(m_ours.shape[-1], m_bv.shape[-1])
    d = (m_ours[..., :T] - m_bv[..., :T]).abs().max().item()
    print(f"[parity] centered_mel vs BigVGAN get_mel_spectrogram  max|Δ| = {d:.2e}  ({'PASS' if d < 1e-4 else 'FAIL'})")

    # (2) causality CI: randomize y[T0:], frames with rightmost < T0 must be identical
    for nf in (512, 1024, 2048):
        mel_full = causal_mel(y, n_fft=nf, hop=128)
        T0 = 20000
        y2 = y.clone(); y2[:, T0:] = (torch.rand_like(y2[:, T0:]) * 2 - 1) * 0.5
        mel2 = causal_mel(y2, n_fft=nf, hop=128)
        n_safe = (T0 - nf) // 128  # frames t with t*128+128-1 < T0-(nf-128) conservatively
        n_safe = max(0, n_safe)
        dd = (mel_full[..., :n_safe] - mel2[..., :n_safe]).abs().max().item() if n_safe else 0.0
        # transient smear proxy: how many past-ms the window spans
        smear_ms = 1000.0 * nf / 44100
        print(f"[causal CI] n_fft={nf:4d} win_past={smear_ms:5.1f}ms  safe_frames={n_safe:4d}  max|Δ|={dd:.2e}  "
              f"({'PASS' if dd < 1e-5 else 'FAIL'})")

    print("[abi]", json.dumps(abi_spec(), ensure_ascii=False))


if __name__ == "__main__":
    _selfcheck()
