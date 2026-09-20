"""G0-4: 8 kHz 以上を見る指標。PESQ が構造的に盲目な帯域を埋める。

`rddsp_loop.score_one` は 16 kHz へ再サンプルして PESQ-wb を取る（同 44-47 行）ので、
**8–22 kHz が全数値から消えている**。息・空気感・サ行は README / ROADMAP §0 が
「商品」と書いている当のもので、その帯域を機械が一切見ないまま 200k step を焼くのは
盲点そのもの。

指標は新発明せず、E0 期に**耳と較正済み**のものを流用する:

  mod_8_16k_dist  8–16 kHz 帯域包絡の変調スペクトル距離（vs gt）
                  `e0_discriminator_hunt.py:140-154`。RESEARCH の記録:
                  「耳が『world より劣化』と言うのに contrast が盲目だった穴を塞ぐ指標
                   （5/5 発話で耳と同順）」
  mod_2_8k_dist   補助
  lsd_b8_16k      8–16 kHz の log-spectral distance（有音フレームのみ）
  hfc_10_16k      10–16 kHz の peak−median コントラスト比（gt 係留）

**レベル整合を必ず先に行う**。[[pesq-blind-band-level]]: 正規化を揃えずに帯域量を並べると
広帯域ゲインの差が band 指標に化けて偽の発見になる（EQ の BLE 改善 0.32 dB のうち 0.24 が
広帯域ゲインだった実例）。ここでは発話ごとに RMS を gt に合わせてから測る。

    uv run python hf_gate.py <tag> [<tag> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R

EPS = 1e-8
SR = R.SR
NFFT = 2048
HOP = 256


def level_match(y: np.ndarray, gt: np.ndarray):
    n = min(len(y), len(gt))
    y, gt = y[:n].astype(np.float64), gt[:n].astype(np.float64)
    return y * np.sqrt((gt ** 2).mean() / max((y ** 2).mean(), 1e-20)), gt


def _subenv(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    s = librosa.stft(y, n_fft=NFFT, hop_length=64)
    f = np.fft.rfftfreq(NFFT, 1.0 / SR)
    return np.sqrt((np.abs(s[(f >= lo) & (f < hi)]) ** 2).sum(0) + EPS)


def _mspec(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    env = np.log(_subenv(x, lo, hi))
    fs_env = SR / 64.0
    n8 = min(len(env), int(fs_env * 8))
    e = env[:n8] - env[:n8].mean()
    spec = np.abs(np.fft.rfft(e * np.hanning(len(e)))) + EPS
    fm = np.fft.rfftfreq(len(e), 1.0 / fs_env)
    s = np.log(spec[(fm >= 2) & (fm <= 400)])
    return s - s.mean()


def _active(gt: np.ndarray, n: int) -> np.ndarray:
    """有音フレーム。低レベルを切り捨てすぎない（息=商品）ので閾値は gt ピーク比 -55 dB。"""
    rms = np.sqrt((gt[: n * HOP].reshape(n, HOP) ** 2).mean(-1) + EPS)
    return rms > rms.max() * 10 ** (-55 / 20)


def hf_metrics(y: np.ndarray, gt: np.ndarray) -> dict:
    y, gt = level_match(y, gt)
    out = {}
    for name, lo, hi in (("mod_2_8k", 2000, 8000), ("mod_8_16k", 8000, 16000)):
        out[f"{name}_dist"] = float(np.abs(_mspec(y, lo, hi) - _mspec(gt, lo, hi)).mean())

    sy = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    sg = np.abs(librosa.stft(gt, n_fft=NFFT, hop_length=HOP))
    f = np.fft.rfftfreq(NFFT, 1.0 / SR)
    nf = min(sy.shape[1], sg.shape[1], len(gt) // HOP)
    act = _active(gt, nf)
    for name, lo, hi in (("b4_8k", 4000, 8000), ("b8_16k", 8000, 16000),
                         ("b16_22k", 16000, 22050)):
        sel = (f >= lo) & (f < hi)
        a = np.log(sy[sel][:, :nf][:, act] + EPS)
        b = np.log(sg[sel][:, :nf][:, act] + EPS)
        out[f"lsd_{name}"] = float(np.abs(a - b).mean())

    for name, lo, hi in (("hfc_5_10k", 5000, 10000), ("hfc_10_16k", 10000, 16000)):
        sel = (f >= lo) & (f < hi)

        def contrast(s):
            b = np.log(s[sel][:, :nf][:, act] + EPS)
            return (b.max(0) - np.median(b, 0)).mean() if b.shape[1] > 2 else np.nan
        out[name] = float(contrast(sy) / (contrast(sg) + EPS))
    return out


def main() -> None:
    import train_gvoc as TG
    import ship_front as SF
    from render_z0c import load, render_pathA, render_pathB
    from rddsp_gpu import build as build_small

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = TG.mel_to_linear(dev)
    _, te = build_small(80, 12)
    items = te[:8]
    netA, _ = load("gvoc_nhv", dev)
    netB, _ = load("gvoc_ship", dev)

    acc = {}
    for x in items:
        w = x["gt"].to(dev).float()
        gt = w.cpu().numpy()
        itA = TG.to_gpu_pre(dict(w=w, mel=x["mel"].to(dev), f0=x["f0"]), dev, W)
        clips = {
            "pathA": render_pathA(netA, itA, dev,
                                  torch.Generator(device=dev).manual_seed(999)),
            "pathB": render_pathB(netB, w, dev,
                                  torch.Generator(device=dev).manual_seed(999), W),
        }
        for k, y in clips.items():
            m = hf_metrics(y, gt)
            acc.setdefault(k, []).append(m)

    keys = list(next(iter(acc.values()))[0].keys())
    print(f"\n  {'指標':16s}" + "".join(f"{k:>12s}" for k in acc))
    print("  " + "-" * (16 + 12 * len(acc)))
    for kk in keys:
        row = "".join(f"{np.mean([d[kk] for d in v]):12.4f}" for v in acc.values())
        print(f"  {kk:16s}{row}")
    print("\n  dist 系は小さいほど良い / hfc は 1.0 が gt 一致（>1 は過鋭, <1 は鈍い）\n")


if __name__ == "__main__":
    main()
