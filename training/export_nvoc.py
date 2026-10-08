"""nvoc の重みを Rust(crates/lightvc-core/src/nvoc.rs)へ書き出す+parity 用の固定入力と期待出力。

出力 <out>/: weights.bin(f32 LE・名前順)・manifest.json({tensors: {name: [offset, shape]}, const, cfg})・fixture/(x・f0・noise・y・mel)
--random で未学習(乱数重み)の構成を書き出す(学習前の RTF 実測用)。--nosrc は調波源なしで学習した重み(manifest の NOSRC=true・Rust は f0 を使わない)。

    CUDA_VISIBLE_DEVICES= uv run python export_nvoc.py --ckpt ../results/nvoc1/last.pt --out ../results/nvoc1/export
    CUDA_VISIBLE_DEVICES= uv run python export_nvoc.py --random --ch 256 --kernels 3,7,11/3,7,11/3,7,11/3,7,11 --out /tmp/nv
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

ROOT = Path(__file__).resolve().parent.parent


def build(ch: int, kernels: str) -> N.NVoc:
    ks = tuple(tuple(int(v) for v in s.split(",")) for s in kernels.split("/"))
    return N.NVoc(ch=ch, kernels=ks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--kernels", default="3,7,11/3,7,11/3,7,11/3,7,11")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nosrc", action="store_true", help="調波源なしで学習した重み(推論で f0 を使わない)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    if a.random:
        torch.manual_seed(0)
        m = build(a.ch, a.kernels)
        step = 0
    else:
        st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        c = st["cfg"]
        m = N.NVoc(ch=c["ch"], kernels=tuple(tuple(k) for k in c["kernels"]), dils=tuple(c["dils"]))
        m.load_state_dict(st["ema"])
        step = int(st["step"])
    m.eval()
    ten = N.export_tensors(m)
    man = {"tensors": {}, "cfg": m.cfg,
           "const": {"SR": N.SR, "HOP": N.HOP, "DELAY": N.DELAY, "N_MEL": N.N_MEL, "NFFT": N.NFFT, "WIN": N.WIN,
                     "H_MAX": N.H_MAX, "H_ROLL": N.H_ROLL, "K_MAX": N.K_MAX, "step": step,
                     "macs_per_second": N.macs_per_second(m), "NOSRC": bool(a.nosrc)}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    if a.random:
        rng = np.random.default_rng(0)
        T = 400
        t = np.arange(T * N.HOP) / N.SR
        x = (0.1 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
        f0 = np.where((np.arange(T) // 50) % 4 == 3, 0.0, 180.0 + 20 * np.sin(np.arange(T) / 30)).astype(np.float32)
    else:
        import s0_artic as S
        it = S.held24()[0]
        x = it["x"].astype(np.float32)[:2 * N.SR]
        import artic_dsp as AD
        f0 = AD.causal_yin(x.astype(np.float64), voi_max=0.45)[0].astype(np.float32)
        T = len(x) // N.HOP
        x = x[:T * N.HOP]
        f0 = f0[:T]
    if a.nosrc:
        f0 = np.zeros_like(f0)
    noise = np.random.default_rng(1).standard_normal(T * N.HOP).astype(np.float32)
    with torch.no_grad():
        xin = torch.cat([torch.zeros(1, N.WIN - N.HOP), torch.from_numpy(x)[None]], -1)
        mel = m.mel_ctx(xin)
        y = m(xin, torch.from_numpy(f0)[None], torch.from_numpy(noise)[None])[0].numpy()
    for nm, arr in (("x", x), ("f0", f0), ("noise", noise), ("y", y), ("mel", mel[0].numpy())):
        arr.astype("<f4").tofile(out / "fixture" / f"{nm}.f32")
    print("exported", off, "floats", round(N.macs_per_second(m) / 1e9, 2), "GMAC/s ->", out, "| y rms", float(np.sqrt((y ** 2).mean())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
