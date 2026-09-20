"""V2F の重みを SIMD 実装が読む平坦 f32 で書き出す（C.2）。

並びは `crates/lightvc-core/src/simd.rs` の `V2fWeights::from_flat` と**完全一致**:
    inp.w, inp.b,
    [conv.w, conv.b, norm.g, norm.b, pw1.w, pw1.b, pw2.w, pw2.b] × L,
    out.w, out.b

⚠ safetensors ではなく生の f32 にする。SIMD 側は `[co][ci][kf][kt]` の
連続配置をそのまま舐めるので、キー名ではなく**並び順が契約**になる。
∴ 並びを変えたら Rust 側の `from_flat` も同時に直す（`CLAUDE.md`）。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
ORDER = ("0.weight", "0.bias", "1.g", "1.b", "2.weight", "2.bias", "3.weight", "3.bias")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu")
    sd = ck["net"]
    ar = ck.get("args", {})
    L = ar.get("L") or ar.get("layers")
    if L is None:
        L = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))

    def w(t: torch.Tensor) -> bytes:
        return t.detach().contiguous().numpy().astype("<f4").tobytes()

    buf = w(sd["inp.weight"]) + w(sd["inp.bias"])
    for i in range(L):
        for k in ORDER:
            buf += w(sd[f"blocks.{i}.{k}"])
    buf += w(sd["out.weight"]) + w(sd["out.bias"])

    tag = ar.get("tag", pathlib.Path(a.ckpt).stem)
    out = pathlib.Path(a.out) if a.out else ROOT / "models" / f"v2f_{tag}.bin"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(buf)
    meta = {"arch": "v2f", "ch": ar.get("ch"), "L": L, "kf": ar.get("kf", 7),
            "kt": ar.get("kt", 3), "cin": ar.get("cin", 4),
            "floats": len(buf) // 4, "step": ck.get("step"), "test": ck.get("test"),
            "src": str(a.ckpt),
            "note": "並び順が契約。simd.rs の V2fWeights::from_flat と同時に直す"}
    out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"  {len(buf)//4} float -> {out}")
    print(f"  ch {meta['ch']} L {L} kf {meta['kf']} kt {meta['kt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
