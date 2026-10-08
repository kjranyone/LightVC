"""rvoc1 の中止後の切り分け(renderer.md §5・prereg の fork): フレーム周期(200Hz)の変調の線が GAN の学習とともに増えた原因。学習なし。

held21 の写し合成で、重みごとに(回帰 A 20k・最初から GAN B 20k・rvoc1 40k / 70k / 100k):
  帯域ごとの変調の線(2–4・4–8・8–12kHz・eval_nvoc.amline と同じ定義)と元音声の値(本物 ≈ 0 の確認)、
  条件をフレーム方向になめらかにした(0.25·0.5·0.25・診断専用の非因果)ときの変調の線 = 条件のフレーム間の折れ目(直線補間の角)が原因かの探針。

    uv run python rvoc1_am_probe.py --out ../results/rvoc1/am_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import rvoc as R
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent


def amline_bands(w: np.ndarray) -> list:
    import scipy.signal as ss
    out = []
    for lo, hi in ((2000, 4000), (4000, 8000), (8000, 12000)):
        sos = ss.butter(6, [lo, hi], btype="band", fs=N.SR, output="sos")
        env = np.abs(ss.hilbert(ss.sosfiltfilt(sos, w.astype(np.float64))))
        env = ss.resample_poly(env, 1, 12)
        f, P = ss.welch(env - env.mean(), fs=N.SR / 12, nperseg=4096)
        pk, bg = 0.0, 0.0
        for h in (200, 400, 600):
            on = (f >= h - 3) & (f <= h + 3)
            nb = (f >= h - 40) & (f <= h + 40) & ~on
            pk += P[on].sum()
            bg += P[nb].mean() * on.sum()
        out.append(10 * np.log10(pk / max(bg, 1e-20)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env_smooth", type=float, default=None, help="全 ckpt に共通(旧 ckpt は明示が要る)")
    ap.add_argument("--ckpts", nargs="*", default=None, help="name=path ...(既定 = rvoc1 の探針の組)")
    a = ap.parse_args()
    dev = "cuda"
    ckpts = {"A_reg_20k": ROOT / "results/diag_rvoc_r/snap/ema_20k.pt", "B_gan_20k": ROOT / "results/diag_rvoc_g/snap/ema_20k.pt",
             "rvoc1_70k": ROOT / "results/rvoc1/snap/ema_70k.pt",
             "rvoc1_100k": ROOT / "results/rvoc1/snap/ema_100k.pt"}
    if a.ckpts:
        ckpts = {kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) for kv in a.ckpts}
    items = TR.held(dev)
    k = torch.tensor([0.25, 0.5, 0.25], device=dev)[None, None]
    kc = torch.tensor([0.25, 0.5, 0.25], device=dev)[None, None]
    rep: dict = {"source": np.round(np.mean([amline_bands(it["x"]) for it in items], 0), 2).tolist()}
    for name, path in ckpts.items():
        st = torch.load(path, map_location="cpu", weights_only=False)
        front = TR.Front("pae", TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
        gen = R.RVoc(ch=st["cfg"]["ch"], d_cond=st["cfg"]["d_cond"]).to(dev)
        gen.load_state_dict(st["ema"])
        gen.eval()
        b, bs, gs = [], [], {}
        with torch.no_grad():
            for it in items:
                xa = torch.from_numpy(it["xa"])[None].to(dev)
                fa = torch.from_numpy(it["f0_h"])[None].to(dev)
                cond = front(xa, fa, torch.from_numpy(it["env"])[None].to(dev))
                exc = TR.excitation(fa, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                y = gen(cond, exc)[0].cpu().numpy()[N.DELAY:]
                def sm(kk, rows, pad):
                    c2 = cond.clone()
                    z = F.conv1d(F.pad(cond[0, rows].reshape(-1, 1, cond.shape[-1]), pad, mode="replicate"), kk)
                    c2[0, rows] = z.reshape(len(rows), -1)
                    return c2
                allr = list(range(cond.shape[1]))
                groups = {"all_sym": (k, allr, (1, 1)), "all_causal": (kc, allr, (2, 0)), "env": (k, list(range(25)), (1, 1)),
                          "f0_vuv": (k, [25, 26], (1, 1)), "ap": (k, list(range(27, 31)), (1, 1))}
                if name == "A_reg_20k":
                    pass
                ys = gen(sm(*groups["all_sym"]), exc)[0].cpu().numpy()[N.DELAY:]
                b.append(amline_bands(y))
                bs.append(amline_bands(ys))
                for gname in ("all_causal", "env", "f0_vuv", "ap"):
                    yg = gen(sm(*groups[gname]), exc)[0].cpu().numpy()[N.DELAY:]
                    gs.setdefault(gname, []).append(amline_bands(yg))
        rep[name] = {"bands_2-4_4-8_8-12k": np.round(np.mean(b, 0), 2).tolist(), "mean": round(float(np.mean(b)), 2),
                     "cond_smoothed_bands": np.round(np.mean(bs, 0), 2).tolist(), "cond_smoothed_mean": round(float(np.mean(bs)), 2),
                     "groups_mean": {g: round(float(np.mean(v)), 2) for g, v in gs.items()}}
        print(name, rep[name], flush=True)
    print("source", rep["source"])
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
