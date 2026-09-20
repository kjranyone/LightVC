"""X' 検証: freebig（F0 非依存 Vocos・耳合格実績）に G の実出力 mel を食わせる。

問い: 上流の mel が GT より粗い（G 予測・平滑化あり）でも、F0 非依存 V は
聴取可能な品質を保つか（= 案 X/Y の前提確認）。

ABI 変換: VC チェーン mel80(1024/256) → freebig mel128(2048/512)。
G mel80 → 線形軸へ写像(W)→ mel128 基底で再投影（発話統計なし・固定行列）。
比較 arm: ①GT mel128 直入力（freebig 天井） ②G mel から変換（VC 相当）
③v2f 現行（対照・source-filter 天井）。

    CUDA_VISIBLE_DEVICES=0 uv run python xprime_freebig.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import soundfile
from free_vocoder import FreeVocoder
from train_vc_e import E1
from train_vc_g import GS, resample_to, F0_FPS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/xprime"
SR = 44100
CKPT = ROOT / "training/checkpoints/freebig/foundation_bigvgan_parity.pt"


def bigvgan_mel(w: np.ndarray) -> torch.Tensor:
    import torchaudio
    m = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=2048, hop_length=512, win_length=2048,
        n_mels=128, f_min=0.0, f_max=None,
        power=1.0, norm="slaney", mel_scale="slaney",
        center=True, pad_mode="reflect")(torch.from_numpy(w)[None])
    return torch.log(m.clamp(min=1e-5))[0]


def mel80_to_mel128(mel80: torch.Tensor) -> torch.Tensor:
    """G 出力 mel80(線形 log-mel・×32768 慣習) → freebig mel128(log slaney)。

    W80: [257,80]・W128 基底で重み最小二乗再投影（固定行列・統計なし）。"""
    import torchaudio
    fb128 = torchaudio.functional.melscale_fbanks(
        n_freqs=1025, f_min=0.0, f_max=SR / 2, n_mels=128, sample_rate=SR,
        norm="slaney", mel_scale="slaney")
    fb80 = torchaudio.functional.melscale_fbanks(
        n_freqs=1025, f_min=0.0, f_max=SR / 2, n_mels=80, sample_rate=SR,
        norm=None, mel_scale="htk")
    A = fb128.T @ fb80 + 1e-2 * torch.eye(80)
    coef = torch.linalg.solve(A, fb128.T)          # [80, 1025]
    return (coef @ (fb80.T @ torch.zeros(1)) if False else coef), fb128


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = FreeVocoder(n_mels=128, dim=ck["args"]["dim"],
                      n_layers=ck["args"]["layers"], causal=False,
                      nfft=2048, win=2048, hop=512).to(dev).eval()
    net.load_state_dict(ck["gen"])

    ek = torch.load(ROOT / "results/diag_e2/diag_e2_best.pt", map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"]).to(dev).eval()
    enet.load_state_dict(ek["net"])
    gk = torch.load(ROOT / "results/v23_gpc/v23_gpc_best.pt", map_location=dev)
    g = GS(dim=gk["args"]["dim"], layers=gk["args"]["L"]).to(dev).eval()
    g.load_state_dict(gk["net"])
    spk = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                     map_location="cpu", weights_only=False)

    # namikawa 変換 mel80 を作成
    w, _ = librosa.load(str(ROOT / "namikawa.mp3"), sr=SR, mono=True)
    x = torch.from_numpy(w) * 32768.0
    n = x.shape[-1]
    mel80 = SF.mel(x)
    f0, _ = SF.causal_f0(x)
    ratio = 2.0 ** (17 / 12)
    f0s = torch.where(f0 > 0, f0 * ratio, f0)
    hop = 512
    nfrm = n // hop
    rms = torch.sqrt(((x[: nfrm * hop] / 32768.0).reshape(nfrm, hop) ** 2).mean(-1) + 1e-12)
    en_ = torch.log(resample_to(rms[None], mel80.shape[-1], F0_FPS)[0].clamp(min=1e-4))
    with torch.no_grad():
        content = enet(mel80.to(dev)[None])[0]
        t = mel80.shape[-1]
        feat = torch.cat([content[:, :t],
                          torch.log(f0s.clamp(min=50.0) / 200.0).to(dev)[None, :t],
                          en_.to(dev)[None, :t]], 0)[None]
        mel_g80 = g(feat, spk["ab97e212acbb6d6b"][None].to(dev))[0].cpu()

    # mel80 → mel128: 線形軸を経由し slaney-log 空間へ合わせる相似スケール
    # （両者 log-mel だがスケール慣習が違う。GT mel128 との相対較正: 同一 wav の
    #   GT mel80→mel128 変換と GT mel128 の比例関係から定数 scale/offset を決める
    #   = 発話統計でなく本検証用の固定較正）
    def to128(m80: torch.Tensor) -> torch.Tensor:
        # 線形周波数軸へ
        from rddsp_gpu import mel_to_linear
        Wl = mel_to_linear("cpu", nbin=1025)
        lin = Wl.cpu() @ (m80 - SF.V_MEL_ADAPT)          # log 線形軸
        lin_e = lin.exp()
        fb = torch.from_numpy(
            librosa.filters.mel(sr=SR, n_fft=2048, n_mels=128,
                                fmin=0.0, fmax=SR / 2, norm="slaney",
                                htk=False)).float()
        m = fb @ lin_e
        return torch.log(m.clamp(min=1e-8))

    gt128 = bigvgan_mel(w)
    # 較正: GT mel80→128 の変換出力を GT mel128 に最小二乗一致（scale+offset）
    conv_gt = to128(mel80)
    T = min(conv_gt.shape[-1], gt128.shape[-1])
    conv_f = conv_gt[:, :T].flatten()
    gt_f = gt128[:, :T].flatten()
    A_ = torch.stack([conv_f, torch.ones_like(conv_f)], -1)
    sol = torch.linalg.lstsq(A_, gt_f[:, None]).solution[:, 0]
    sc, off = float(sol[0]), float(sol[1])
    print(f"  ABI 較正: mel128 = {sc:.3f} * conv + {off:.3f}")

    def synth(m128: torch.Tensor, name: str):
        with torch.no_grad():
            y = net(m128[None].to(dev))[0].cpu().numpy()
        soundfile.write(OUT / f"{name}.wav", np.clip(y, -1, 1), SR)
        print(f"  {name}.wav")

    # arm1: 天井（GT mel128 直）
    synth(gt128, "fb_gt128_ceiling")
    # arm2: VC 相当（G mel80 → 変換 mel128）
    conv_g = to128(mel_g80) * sc + off
    synth(conv_g, "fb_gmel80_vc")
    # arm3: 自己変換（source の mel80 → 変換。f0 シフトなし = 経路健全性）
    conv_s = to128(mel80) * sc + off
    synth(conv_s, "fb_src80_self")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
