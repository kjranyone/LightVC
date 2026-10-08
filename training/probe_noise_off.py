"""GAN 段の悪化の帰属(学習なし): 生成器の雑音チャネル(励起 exc[1])を推論時に 0 / 0.5 倍にしたとき、倍音間の谷の深さと PESQ が戻るか。

    CUDA_VISIBLE_DEVICES=0 uv run python probe_noise_off.py name=path[:nosrc] ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
import probe_contrast_lib as C


def main() -> int:
    dev = "cuda"
    items = sorted(E.held_items(), key=lambda it: -float((it["f0"] > 0).mean()))[:8]
    for arg in sys.argv[1:]:
        name, spec = arg.split("=", 1)
        path, _, flag = spec.partition(":")
        st = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("ema", "model"):
            if key not in st:
                continue
            m = N.NVoc().to(dev)
            m.load_state_dict(st[key])
            m.eval()
            for scale in (1.0, 0.5, 0.0):
                cs, ps = [], []
                for it in items:
                    f0 = it["f0"] * (0 if flag == "nosrc" else 1)
                    g = torch.Generator(device="cpu").manual_seed(0)
                    noise = torch.randn(1, len(it["x"]), generator=g).to(dev) * scale
                    xin = torch.cat([torch.zeros(1, N.WIN - N.HOP), torch.from_numpy(it["x"])[None]], -1).to(dev)
                    with torch.no_grad():
                        y = m(xin, torch.from_numpy(f0)[None].to(dev), noise)[0].cpu().numpy()[N.DELAY:]
                    cs.append(C.contrast(y.astype(np.float64), it["f0"]))
                    ps.append(E.metrics(y, it["x"], 0, dev)["pesq"])
                print(f"{name} step {st.get('step')} {key} noise×{scale}: contrast {np.mean(cs):.3f} pesq {np.mean(ps):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
