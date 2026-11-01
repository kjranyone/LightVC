"""構音の描画(current/artic_render.md・基準 PR4)を Rust(crates/lightvc-core/src/artic_render.rs)へ書き出す。
出力 <out>/: weights.bin(f32 LE・名前順)・manifest.json({tensors: {name: [offset, shape]}, cfg})。
cfg.level_gamma = 大きさの音節の動きのつまみ γ の既定(c0 = s + γ(c0₃₀ − s)・s = 発声(c0n > level_thr)だけで更新する level_hz の 1 極・D669・オーナー判断 10-29 で 0.5)。
名前: env.inp / env.blocks.i / env.out(.w・.b・weight_norm は実効の重みに畳む)・curve [1025](描画の較正 1 × 表の引きの較正 2・パワーの比)・dct [25, 128]・fc [128](mel の帯の中心 Hz)。
    uv run python export_artic_render.py --net ../results/diag_envnet3/last.pt --out ../results/artic_render2/export --level_gamma 0.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import pae as PA
from d634_artic_render import DCT, FC
from d652_render_cal import curve as curve1
from d659_table_cal import curve2
from train_envnet import EnvNet

CFG = {"gain_db": 29.6, "hp_tau_s": 2.0, "c0_hz": 30.0, "lf_hz": 30.0, "shape_hz": 25.0, "hop": 240, "sr": 48000, "nfft": 2048, "c0_sil": float(PA.C0_SIL),
       "per_bands": [[300, 1000], [1000, 2500], [2500, 5000], [5000, 10000]], "level_hz": 2.0, "level_thr": 0.05}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level_gamma", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = torch.load(a.net, map_location="cpu", weights_only=False)
    net = EnvNet(st["cfg"]["ch"], tuple(st["cfg"]["dils"]), st["cfg"]["d_c2"]).eval()
    net.load_state_dict(st["net"])
    ten = {"env.inp.w": net.inp.conv.weight, "env.inp.b": net.inp.conv.bias, "env.out.w": net.out.weight, "env.out.b": net.out.bias}
    for i, b in enumerate(net.blocks):
        ten[f"env.blocks.{i}.w"] = b.conv.weight
        ten[f"env.blocks.{i}.b"] = b.conv.bias
    ten = {k: v.detach().float().numpy() for k, v in ten.items()}
    ten["curve"] = (curve1() * curve2()).astype(np.float32)
    ten["curve1"] = curve1().astype(np.float32)
    ten["dct"] = DCT[:25].astype(np.float32)
    ten["fc"] = FC.astype(np.float32)
    cfg = CFG | {"level_gamma": a.level_gamma, "k": 200, "cv": int(st["cfg"]["d_c2"]), "ch": int(st["cfg"]["ch"]), "dils": list(st["cfg"]["dils"]), "net": a.net, "net_step": int(st.get("step", 0))}
    man = {"tensors": {}, "cfg": cfg}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = np.ascontiguousarray(ten[k], np.float32)
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("exported", out, {k: v.shape for k, v in ten.items() if not k.startswith("env.blocks")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
