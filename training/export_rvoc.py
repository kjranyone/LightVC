"""出力部 rvoc の重みを Rust(crates/lightvc-core/src/rvoc.rs)へ書き出す + 一致検査用の固定の入力と期待出力。
出力 <out>/: weights.bin(f32 LE・名前順)・manifest.json({tensors: {name: [offset, shape]}, cfg, const})・fixture/(cond.f32 [T][D_COND]・exc.f32 [T][2][HOP]・y.f32 [T·HOP])。
名前 = PyTorch のモジュールの経路 + .w / .b(weight_norm は実効の重みに畳む): pre / up.i / src.i / res.i.j.c1.m / res.i.j.c2.m / post。
    CUDA_VISIBLE_DEVICES= uv run python export_rvoc.py --ckpt ../results/rvoc3hi/snap/ema_330k.pt --out ../results/rvoc3hi/export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import rvoc as R
import train_rvoc as TR


def tensors(m: R.RVoc) -> dict:
    out = {}
    for name, mod in m.named_modules():
        if isinstance(mod, R.CConv):
            out[f"{name}.w"] = mod.conv.weight.detach().float().contiguous()
            out[f"{name}.b"] = mod.conv.bias.detach().float().contiguous()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = st["cfg"]
    m = R.RVoc(ch=c["ch"], kernels=tuple(c["kernels"]), dils=tuple(c["dils"]), d_cond=c["d_cond"])
    m.load_state_dict(st["ema"])
    m.eval()
    ten = tensors(m)
    man = {"tensors": {}, "cfg": m.cfg, "const": {"SR": N.SR, "HOP": N.HOP, "DELAY": N.DELAY, "D_COND": int(c["d_cond"]), "UPS": list(R.UPS),
                                                  "step": int(st.get("step", 0)), "H_MAX": N.H_MAX, "H_ROLL": N.H_ROLL, "K_MAX": N.K_MAX, "env_smooth": float(TR.ckpt_env_smooth(st, None)), "macs_per_second": R.macs_per_second(m)}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    T = a.frames
    rng = np.random.default_rng(0)
    t = np.arange(T)
    cond = np.zeros((c["d_cond"], T), np.float32)
    cond[0] = 4.0 + 0.8 * np.sin(t / 20.0)
    for i in range(1, 25):
        cond[i] = 0.3 * np.sin(t / (7.0 + i) + i) / i
    f0 = np.where((t // 60) % 5 == 4, 0.0, 260.0 + 60 * np.sin(t / 37.0)).astype(np.float32)
    vo = (f0 > 0).astype(np.float32)
    cond[25] = np.where(vo > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)
    cond[26] = vo
    cond[27:31] = (0.6 + 0.2 * rng.random((4, T))) * vo
    with torch.no_grad():
        exc = TR.excitation(torch.from_numpy(f0)[None], T * N.HOP, torch.Generator().manual_seed(0))
        y = m(torch.from_numpy(cond)[None], exc)[0].numpy()
    exc_np = exc[0].numpy()
    exc_blocks = exc_np.reshape(2, T, N.HOP).transpose(1, 0, 2).copy()
    cond.T.copy().astype("<f4").tofile(out / "fixture" / "cond.f32")
    exc_blocks.astype("<f4").tofile(out / "fixture" / "exc.f32")
    y.astype("<f4").tofile(out / "fixture" / "y.f32")
    f0.astype("<f4").tofile(out / "fixture" / "f0.f32")
    print("exported", len(ten), "tensors", off, "floats ->", out, "| fixture T", T, "| y rms", float(np.sqrt((y ** 2).mean())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
