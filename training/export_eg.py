"""E1/G1 の重み受け渡し（フラット f32、v2f と同じ流儀）。

順序（この順で連結・キー名は Rust `eg_infer.rs` と完全一致）:
    inp.weight [dim,cin,3] / inp.bias [dim]
    blocks.{i}.dw.weight [dim,dim,3] / .dw.bias / .norm.weight / .norm.bias
    blocks.{i}.pw1.weight [3dim,dim] / .pw1.bias / .pw2.weight [dim,3dim] / .pw2.bias
    out.weight [cout,dim] / out.bias [cout]

    uv run python export_eg.py --ckpt ../results/diag_e1/diag_e1_best.pt --kind e
    uv run python export_eg.py --ckpt ../results/diag_cartB/diag_cartB_best.pt --kind g
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=["e", "g"], required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["net"]
    ar = ck["args"]
    L = ar["L"]
    dim = ar["dim"]
    cin = {"e": 80, "g": 770}[a.kind]
    cout = {"e": 768, "g": 80}[a.kind]
    if a.kind == "e" and ar.get("look", 0):
        raise SystemExit("look>0 の E は未対応（採用 E は look=0）")

    order = ["inp.weight", "inp.bias"]
    for i in range(L):
        order += [f"blocks.{i}.dw.weight", f"blocks.{i}.dw.bias",
                  f"blocks.{i}.norm.weight", f"blocks.{i}.norm.bias",
                  f"blocks.{i}.pw1.weight", f"blocks.{i}.pw1.bias",
                  f"blocks.{i}.pw2.weight", f"blocks.{i}.pw2.bias"]
    order += ["out.weight", "out.bias"]
    missing = [k for k in order if k not in sd]
    extra = [k for k in sd if k not in order]
    if missing or extra:
        raise SystemExit(f"キー不一致 missing={missing[:4]} extra={extra[:4]}")

    flat = np.concatenate([sd[k].numpy().astype(np.float32).ravel() for k in order])
    tag = pathlib.Path(a.ckpt).stem
    out = pathlib.Path(a.out) if a.out else ROOT / "models" / f"{a.kind}1_{tag}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    flat.tofile(out)
    meta = {"kind": a.kind, "dim": dim, "L": L, "cin": cin, "cout": cout,
            "k": 3, "dil": [2 ** (i // 2) for i in range(L)],
            "ctx": ar["ctx"], "n_f32": int(flat.size), "src": str(a.ckpt),
            "step": ck.get("step"),
            "note": "順序は export_eg.py 冒頭・eg_infer.rs と一致（CLAUDE.md キー規約）"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"  {out}  ({flat.size * 4 / 1e6:.1f}MB)")
    print(f"  {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
