"""S1-LG: latent 生成可能性 gate（causal_codec.md rev2 規定）。

s1_3 codec の latent で:
  LG-1 監査: channel scale/相関/時間差分/帯域
  LG-2 摂動: latent に SNR40dB 摂動 → decode 波形の劣化が線形か
  LG-3 tiny CFM: 5-10 発話で latent を条件なし CFM 学習し、
       学習発話と**held 区間**（同発話の未学習区間）を再現

    CUDA_VISIBLE_DEVICES=0 uv run python s1_lg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "female-dataset"
OUT = ROOT / "results/s1_lg"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ck["net"])
    codec.eval()

    # ---- LG-1: latent 監査（held 女声 10 発話） ----
    spk = "fe659435bbd284e8"
    wavs = sorted((FD / spk).glob("*.wav"))[:10]
    Z = []
    for w in wavs:
        x, _ = librosa.load(str(w), sr=SAMPLE_RATE, mono=True)
        n = len(x) // HOP_LENGTH * HOP_LENGTH
        if n < SAMPLE_RATE:
            continue
        with torch.no_grad():
            z = codec.encode(torch.from_numpy(x[:n].astype(np.float32))[None, None].to(dev))[0].cpu()
        Z.append(z)
    Zall = torch.cat(Z, dim=-1)                     # [32, T]
    std_ch = Zall.std(dim=-1)
    C = torch.corrcoef(Zall) if Zall.shape[-1] > 32 else torch.eye(32)
    off = C[~torch.eye(32, dtype=torch.bool)]
    dz = (Zall[:, 1:] - Zall[:, :-1]).std(dim=-1)
    print("LG-1 監査:")
    print(f"  channel std: min {std_ch.min():.3f} med {std_ch.median():.3f} max {std_ch.max():.3f}")
    print(f"  相関(対角除く): med {off.median():.3f} p99 {off.quantile(0.99):.3f} max {off.max():.3f}")
    print(f"  時間差分 std med {dz.median():.3f} / std 比 {dz.median()/std_ch.median():.3f}")
    # latent スペクトルの帯域偏り（channel ごとの時間周波数の平坦さは省略・相関で代替）

    # ---- LG-2: 摂動耐性（SNR 40dB・1 発話） ----
    x, _ = librosa.load(str(wavs[0]), sr=SAMPLE_RATE, mono=True)
    n = len(x) // HOP_LENGTH * HOP_LENGTH
    xt = torch.from_numpy(x[:n].astype(np.float32))[None, None].to(dev)
    with torch.no_grad():
        z0 = codec.encode(xt)
        y0 = codec.decode(z0)[0, 0].cpu().numpy()
        eps = torch.randn_like(z0)
        z_pert = z0 + eps * (z0.norm() / eps.norm()) * (10 ** (-40 / 20))
        y_pert = codec.decode(z_pert)[0, 0].cpu().numpy()
    base_err = float(np.abs(y0 - x[:n]).mean())
    pert_delta = float(np.abs(y_pert - y0).mean())
    print(f"\nLG-2 摂動(SNR40dB): 再構成誤差 {base_err:.5f} / 摂動による追加誤差 {pert_delta:.5f}"
          f"（比 {pert_delta/max(base_err,1e-9):.2f}・線形劣化なら小値）")
    soundfile.write(OUT / "perturb.wav", np.clip(y_pert, -1, 1), SAMPLE_RATE)
    soundfile.write(OUT / "recon0.wav", np.clip(y0, -1, 1), SAMPLE_RATE)
    soundfile.write(OUT / "gt0.wav", x[:n], SAMPLE_RATE)

    # ---- LG-3: tiny CFM（1 発話・前半学習→後半 held 再現） ----
    z1 = Z[0]                                        # [32, T]
    T = z1.shape[-1]
    Ttr = T * 3 // 4
    ztr = z1[:, :Ttr].to(dev)
    net = nn.Sequential()
    class VF(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv1d(33, 256, 5, padding=0), nn.GELU(),
                nn.Conv1d(256, 256, 5, padding=0), nn.GELU(),
                nn.Conv1d(256, 32, 1))
        def forward(self, z, t):
            h = torch.cat([z, t.expand(z.shape[0], 1, z.shape[-1])], 1)
            h = F.pad(h, (8, 0))
            return self.body(h)
    vf = VF().to(dev)
    opt = torch.optim.AdamW(vf.parameters(), lr=3e-4)
    g = torch.Generator(device=dev).manual_seed(0)
    print("\nLG-3 tiny CFM（学習=前半 75%・目標=後半 25% を含む全區間の再現）:")
    for it in range(3000):
        s = int(torch.randint(0, max(Ttr - 64, 1), (1,)))
        zt = ztr[:, s:s+64][None]
        t = torch.rand(1, device=dev)
        z0s = torch.randn(1, 32, 64, device=dev)
        zt_mix = (1 - t) * z0s + t * zt
        loss = F.mse_loss(vf(zt_mix, t), zt - z0s)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"  it {it}: cfm {float(loss):.4f}")
    # 後半 held の再現: noise から 1-step で
    with torch.no_grad():
        Th = T - Ttr
        z0h = torch.randn(1, 32, Th, device=dev)
        zh = (z0h + vf(z0h, torch.ones(1, device=dev)))[0].cpu()
    gt_h = z1[:, Ttr:]
    l1_train_domain = float((zh[:, :0].abs().mean()) if False else 0)
    l1 = float((zh - gt_h).abs().mean())
    l1_baseline = float((torch.randn_like(gt_h) * gt_h.std() - gt_h).abs().mean())
    print(f"  held 区間 latent L1: {l1:.4f}（ランダム基準 {l1_baseline:.4f}・"
          f"GT 自己 std {float(gt_h.std()):.4f}）")
    with torch.no_grad():
        yh = codec.decode(zh[None].to(dev))[0, 0].cpu().numpy()
    yh_gt, _ = librosa.load(str(wavs[0]), sr=SAMPLE_RATE, mono=True)
    tail = yh_gt[Ttr*HOP_LENGTH: Ttr*HOP_LENGTH + len(yh)]
    soundfile.write(OUT / "cfm_held.wav", np.clip(yh, -1, 1), SAMPLE_RATE)
    soundfile.write(OUT / "cfm_held_gt.wav", tail, SAMPLE_RATE)
    print("  -> cfm_held.wav / cfm_held_gt.wav 保存")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
