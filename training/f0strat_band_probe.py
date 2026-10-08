"""高い声の層別の差がどの帯域・どの区間にあるか(学習なし): 有声フレームの帯域ごとの log-mel 誤差(128 帯・1024 点・低域から 0–1k / 1–2.5k / 2.5–5k / 5–10k / 10k–)を群ごとに。
    uv run python f0strat_band_probe.py --ckpt <ema.pt> --env_smooth 0.25 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import rvoc as R
import train_rvoc as TR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env_smooth", type=float, default=None, help="省略時は ckpt の env_smooth(無ければ 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    g = R.RVoc(ch=st["cfg"]["ch"], kernels=tuple(st["cfg"]["kernels"]), dils=tuple(st["cfg"]["dils"]), d_cond=st["cfg"]["d_cond"]).to(dev)
    g.load_state_dict(st["ema"])
    g.eval()
    fr = TR.Front("pae", TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
    mel = torchaudio.transforms.MelSpectrogram(N.SR, 1024, 1024, N.HOP, n_mels=128, f_min=0, f_max=N.SR // 2, power=1.0, center=True).to(dev)
    edges = [0, 1000, 2500, 5000, 10000, 24000]
    centers = torchaudio.functional.melscale_fbanks(513, 0, N.SR // 2, 128, N.SR, None, "htk")
    hz = (torch.arange(513) * N.SR / 1024)
    bandc = (centers * hz[:, None]).sum(0) / centers.sum(0).clamp(min=1e-9)
    groups = {"low_lt450": [], "high_ge490": []}
    with torch.no_grad():
        for it in TR.held(dev):
            f0 = it["f0_h"]
            med = float(np.median(f0[f0 > 0]))
            y = torch.from_numpy(TR.render(g, fr, it, f0, dev)[N.DELAY:]).to(dev)
            x = torch.from_numpy(it["x"]).to(dev)
            n = min(len(x), len(y))
            ly = (mel(y[:n]) + 1e-5).log()
            lx = (mel(x[:n]) + 1e-5).log()
            T = min(ly.shape[-1], len(f0))
            err = (ly[:, :T] - lx[:, :T]).abs().cpu()
            bias = (ly[:, :T] - lx[:, :T]).cpu()
            vo = torch.from_numpy(f0[:T] > 0)
            row = {}
            for kind, mk in (("voiced", vo), ("unvoiced", ~vo)):
                for b in range(5):
                    sel = (bandc >= edges[b]) & (bandc < edges[b + 1])
                    row[f"{kind}_{edges[b]}-{edges[b + 1]}"] = float(err[sel][:, mk].mean()) if mk.any() else float("nan")
                    row[f"bias_{kind}_{edges[b]}-{edges[b + 1]}"] = float(bias[sel][:, mk].mean()) if mk.any() else float("nan")
            row["voiced_frac"] = float(vo.float().mean())
            if med < 450:
                groups["low_lt450"].append(row)
            elif med >= 490:
                groups["high_ge490"].append(row)
    rep = {g_: {k: round(float(np.nanmean([r[k] for r in rs])), 3) for k in rs[0]} for g_, rs in groups.items()}
    for k in rep["low_lt450"]:
        print(f"{k:28s} low {rep['low_lt450'][k]:7.3f} high {rep['high_ge490'][k]:7.3f}")
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
