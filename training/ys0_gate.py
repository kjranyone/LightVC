"""Y-S0: 凍結 DACVAE decoder の streaming 適性計測（CFM 本学習前の関門）。

測定:
  S0-a: latent 1フレーム摂動の波形影響の時間広がり（receptive field）
  S0-b: 未来文脈を切り詰めた decode の品質（lookahead k フレーム掃引・
        mel-L1 vs full decode）＝「必要未来文脈」の定量
  S0-c: 厳密 causal decode（k=0）の品質

    cd Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/ys0_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.codec import DACVAECodec

LVC = Path("/home/kojirotanaka/kjranyone/LightVC")
OUT = LVC / "results/ys0"
SR = 48000
HOP = 1920
FPS = 25.0


def mel128(w: np.ndarray) -> torch.Tensor:
    m = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, win_length=2048,
        n_mels=128, f_min=0.0, f_max=None, power=1.0,
        norm="slaney", mel_scale="slaney")(torch.from_numpy(w)[None])
    return torch.log(m.clamp(min=1e-5))[0]


def s0a_receptive(codec: DACVAECodec, z: torch.Tensor) -> None:
    """frame t0 を摂動したとき波形差が nonzero になる時間範囲。"""
    t0 = z.shape[1] // 2
    zp = z.clone()
    zp[0, t0] += 0.5 * zp[0, t0].std().clamp(min=0.1)
    with torch.no_grad():
        y0 = codec.decode_latent(z)[0, 0].cpu()
        yp = codec.decode_latent(zp)[0, 0].cpu()
    d = (yp - y0).abs()
    nz = torch.where(d > 1e-6)[0]
    if len(nz) == 0:
        print("  S0-a: 摂動が波形に現れず（要再検討）")
        return
    s0 = t0 * HOP
    pre = (s0 - nz.min().item()) / SR * 1000
    post = (nz.max().item() - s0) / SR * 1000
    print(f"  S0-a: frame {t0} 摂動の影響 = 位置より {pre:.0f}ms 前 〜 {post:.0f}ms 後")


def s0b_lookahead(codec: DACVAECodec, z: torch.Tensor, w: np.ndarray,
                  name: str) -> list[tuple[int, float]]:
    """S0-c の chunked で代替（関数は廃止・main を参照）。"""
    return []


def chunked_decode(codec: DACVAECodec, z: torch.Tensor, past: int,
                   future: int) -> np.ndarray:
    """frame i を出すのに [i-past, i+future] の窓だけで decode し連結。

    実装: 窓ごとに独立 decode し中央 1 フレーム分の波形を採取して連結。
    （境界アーティファクト込み＝保守評価）"""
    T = z.shape[1]
    out = []
    for i in range(T):
        lo = max(0, i - past)
        hi = min(T, i + future + 1)
        seg = z[:, lo:hi]
        with torch.no_grad():
            y = codec.decode_latent(seg)[0, 0].cpu().numpy()
        # この窓内で frame i に対応する中央波形 = (i - lo) フレーム目の hop 区間
        s = (i - lo) * HOP
        e = s + HOP
        if e > len(y):
            e = len(y)
            out.append(np.zeros(HOP))
            continue
        out.append(y[s:e])
    return np.concatenate(out)[: T * HOP]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    codec = DACVAECodec.load(device="cuda", normalize_db=None)
    src = LVC / "female-dataset/fe659435bbd284e8/fe659435bbd284e8_00005555.wav"
    w, _ = librosa.load(str(src), sr=SR, mono=True)
    x = torch.from_numpy(w)[None]
    z = codec.encode_waveform(x, SR)
    print(f"  latent {tuple(z.shape)} ({z.shape[1]/FPS:.1f}s @25fps)")

    s0a_receptive(codec, z)

    with torch.no_grad():
        y_full = codec.decode_latent(z)[0, 0].cpu().numpy()
    m_full = mel128(y_full[: len(w)])
    import soundfile
    print("\n  S0-b: chunked decode（frame i のみ past/future 窓使用・境界込み保守）")
    for past, future, tag in [(64, 0, "causal(k=0)"), (64, 1, "look+1"),
                              (64, 2, "look+2"), (64, 4, "look+4"),
                              (64, 8, "look+8"), (64, 16, "look+16")]:
        # 10 秒分だけ（低速なため）
        T10 = min(z.shape[1], 250)
        yc = chunked_decode(codec, z[:, :T10], past, future)
        wc = w[: len(yc)]
        m_c = mel128(yc)
        T = min(m_c.shape[-1], m_full.shape[-1])
        l1 = float((m_c[:, :T] - m_full[:, :T]).abs().mean())
        sf_path = OUT / f"chunk_{tag}.wav"
        soundfile.write(sf_path, np.clip(yc, -1, 1), SR)
        print(f"    {tag:10s}: mel-L1 vs offline {l1:.4f}  -> {sf_path.name}")
    soundfile.write(OUT / "offline_full.wav", np.clip(y_full, -1, 1), SR)
    soundfile.write(OUT / "gt.wav", w, SR)
    print("  offline_full.wav / gt.wav 保存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
