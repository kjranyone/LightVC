"""V2-3 深掘り: 「視聴に耐えない」の瓶颈分解（FM着手前の最低限の解析）。

arm 構成（同一条件で分離）:
  GT女声          : 自然音声（参照）
  cs_dom          : GT女声 mel -> V          = ボコーダ天井
  cs17            : 入力男声 mel + 因果f0×2^(17/12) -> V  = f0検出+V
  cs17_pyin       : 同 mel + pyin f0(頑健)×shift -> V     = f0検出の寄与分離
  pyv_w2_s2000    : G予測 mel + 因果f0 -> V               = G の寄与分離

計器: pyworld harvest f0 + aperiodicity（かすれ＝非周期性の標準計量）・jitter。

    CUDA_VISIBLE_DEVICES="" uv run python v23_decompose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
import v2f as V2
from eval_g1 import render
from rddsp_gpu import mel_to_linear

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/v23_decompose"
SR = 44100


def measure(path: str) -> dict:
    w, _ = librosa.load(path, sr=SR, mono=True)
    w16 = librosa.resample(w, orig_sr=SR, target_sr=16000)
    w16 = w16.astype(np.float64)
    f0, t = pyworld.harvest(w16, 16000, f0_floor=50, f0_ceil=600)
    sp = pyworld.cheaptrick(w16, f0, t, 16000)
    ap = pyworld.d4c(w16, f0, t, 16000)
    v = f0 > 0
    ap_v = ap[v]
    aperiodicity = float((1.0 - ap_v).mean())          # 0=完全周期, 1=完全非周期
    ap5k = float((1.0 - ap_v[:, ap.shape[1] // 2:]).mean())  # 高域側
    vf = f0[v]
    if len(vf) > 3:
        d = np.abs(np.diff(vf)) / vf[:-1]
        jitter = float(np.median(d[d > 0]) * 100) if (d > 0).any() else 0.0
    else:
        jitter = float("nan")
    return {"aperiodicity": round(aperiodicity, 3),
            "ap_hi": round(ap5k, 3),
            "jitter_pct": round(jitter, 2),
            "f0med": round(float(np.median(vf)), 1) if v.any() else None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cpu"
    vk = torch.load(ROOT / "results/v2f_prior_20260819_013905/"
                    "v2f_prior_20260819_013905_best.pt", map_location=dev)
    vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                  norm=vk["args"]["norm"]).to(dev).eval()
    vnet.load_state_dict(vk["net"])
    W = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)

    # --- 女声 GT と cs_dom（ボコーダ天井） ---
    from train_vc_g import FEATS, load_item
    files = []
    for root in FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    spk_all = sorted({f.parent.name for f in files})
    held = set(spk_all[-24:])
    ev = [f for f in files if f.parent.name in held][:3]
    for i, f in enumerate(ev):
        d, mel = load_item(f)
        wp = ROOT / str(d["path"]).lstrip("./")
        if not wp.exists():
            wp = Path(str(d["path"]))
        w, _ = librosa.load(str(wp), sr=SR, mono=True)
        gt = torch.from_numpy(w) * 32768.0
        n = gt.shape[-1]
        f0, _ = SF.causal_f0(gt)
        y = render(mel, f0, vnet, W, n, dev)
        soundfile.write(OUT / f"gt_{i}.wav", w, SR)
        soundfile.write(OUT / f"cs_dom_{i}.wav",
                        y.clamp(-1, 1).numpy(), SR)
        print(f"  gt_{i} / cs_dom_{i}", flush=True)

    # --- namikawa 系 ---
    w, _ = librosa.load(str(ROOT / "namikawa.mp3"), sr=SR, mono=True)
    x = torch.from_numpy(w) * 32768.0
    n = x.shape[-1]
    mel = SF.mel(x)
    f0c, _ = SF.causal_f0(x)
    ratio = 2.0 ** (17 / 12)

    w16 = librosa.resample(w, orig_sr=SR, target_sr=16000)
    fp, vp, _ = librosa.pyin(w16.astype(np.float64), fmin=50, fmax=500, sr=16000)
    # pyin f0 を mel グリッド(172fps)へ因果リサンプル
    T = mel.shape[-1]
    idx_src = ((np.arange(T) + 1) * 50.0 / (SR / SF.HOP_A) - 1).clip(0, len(fp) - 1)
    i0 = np.floor(idx_src).astype(int)
    frac = idx_src - i0
    fp_i = np.where(np.isnan(fp), 0.0, fp)
    fp_fill = np.zeros(len(fp_i))
    last = 0.0
    for k in range(len(fp_i)):
        if fp_i[k] > 0:
            last = fp_i[k]
        fp_fill[k] = last
    f0p = torch.from_numpy(fp_fill[i0] * (1 - frac) + fp_fill[np.minimum(i0 + 1, len(fp_fill) - 1)] * frac).float()
    f0p = torch.where(torch.isnan(f0p), torch.zeros_like(f0p), f0p)

    f0s_c = torch.where(f0c > 0, f0c * ratio, f0c)
    f0s_p = torch.where(f0p > 0, f0p * ratio, f0p)

    y_c = render(mel, f0s_c, vnet, W, n, dev)
    soundfile.write(OUT / "cs17_causalf0.wav", y_c.clamp(-1, 1).numpy(), SR)
    y_p = render(mel, f0s_p, vnet, W, n, dev)
    soundfile.write(OUT / "cs17_pyinf0.wav", y_p.clamp(-1, 1).numpy(), SR)

    print("\n=== 分解計測（aperiodicity: 0=周期/澄んだ, 1=非周期/かすれ）===")
    rows = [
        ("GT女声0", str(OUT / "gt_0.wav")),
        ("cs_dom0(V天井)", str(OUT / "cs_dom_0.wav")),
        ("GT女声1", str(OUT / "gt_1.wav")),
        ("cs_dom1(V天井)", str(OUT / "cs_dom_1.wav")),
        ("入力男声", str(ROOT / "namikawa.mp3")),
        ("cs17_因果f0", str(OUT / "cs17_causalf0.wav")),
        ("cs17_pyin_f0", str(OUT / "cs17_pyinf0.wav")),
        ("pyv_w2_s2000(現行)", str(ROOT / "results/diag_ear_v23/pyv_w2_s2000.wav")),
        ("pyv_w5_20k", str(ROOT / "results/diag_ear_v23/pyv.wav")),
    ]
    for name, p in rows:
        if not Path(p).exists():
            continue
        m = measure(p)
        print(f"  {name:22s} aper {m['aperiodicity']:.3f} hi {m['ap_hi']:.3f}"
              f"  jitter {m['jitter_pct']:.2f}%  f0 {m['f0med']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
