"""C.2 の重み受け渡し。**`export_weights.py` は使わない**（旧 B1/B3 系）。

キー名は Rust (`crates/lightvc-core/src/v1d.rs`) と完全一致させる（`CLAUDE.md`）。
    inp.{weight,bias} / blocks.{i}.{dw,norm,pw1,pw2}.{weight,bias}
    norm.{weight,bias} / out.{weight,bias}

    uv run python export_v1d.py --ckpt ../results/<TAG>/<TAG>_best.pt
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parent.parent

EXPECT = ("inp", "norm", "out")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["net"]
    ar = ck.get("args", {})
    tag = ar.get("tag", pathlib.Path(a.ckpt).stem)

    # ⚠ 名前を書き換えない。書き換えた瞬間に Rust 側と食い違い、
    #   しかもロード時に「無い」ではなく「形が違う」で落ちるので原因が遠くなる。
    bad = [k for k in sd if not (k.startswith("blocks.")
                                 or k.split(".")[0] in EXPECT)]
    if bad:
        print(f"  ⚠ 想定外のキー {bad[:6]}（Rust 側に無い＝両方直す規約）")
        return 1

    out = pathlib.Path(a.out) if a.out else ROOT / "models" / f"v1d_{tag}.safetensors"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in sd.items()}, str(out))

    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({
        "arch": ar, "step": ck.get("step"), "test": ck.get("test"),
        "keys": len(sd), "src": str(a.ckpt),
        "note": "キー名は crates/lightvc-core/src/v1d.rs と完全一致（CLAUDE.md）",
    }, ensure_ascii=False, indent=1))
    try:
        rel = out.relative_to(ROOT)
    except ValueError:
        rel = out
    print(f"  {len(sd)} キー -> {rel}")
    print(f"  cin {ar.get('cin')} dim {ar.get('dim')} L {ar.get('L')} "
          f"k_in {ar.get('k_in')} k {ar.get('k')} nbin {ar.get('nbin')} "
          f"ctx {ar.get('ctx')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
