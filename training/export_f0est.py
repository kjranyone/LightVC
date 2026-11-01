"""F1(f0est.py の因果な f0・有声推定器)を Rust(crates/lightvc-core/src/f0est.rs)へ書き出す + 一致検査用の固定の入力と期待出力。
出力 <out>/: weights.bin・manifest.json({tensors, cfg, const})・fixture/(x.f32・feat.f32 [T][D_IN]・v.f32 [T](有声のロジット)・f0.f32 [T](Hz・無声 0))。
名前: mel.fb・mel.win・inp・blocks.i.c1・blocks.i.c2・out(+ .w / .b)。
    CUDA_VISIBLE_DEVICES= uv run python export_f0est.py --ckpt ../results/f0est3/last.pt --out ../results/f0est3/export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f0est as FE
import nvoc as N


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).eval()
    m.load_state_dict(st["net"])
    front = FE.Front().eval()
    ten = {"mel.fb": front.mel.fb, "mel.win": front.mel.win, "inp.w": m.inp.weight, "inp.b": m.inp.bias, "out.w": m.out.weight, "out.b": m.out.bias}
    for i, b in enumerate(m.blocks):
        ten[f"blocks.{i}.c1.w"] = b.c1.weight; ten[f"blocks.{i}.c1.b"] = b.c1.bias
        ten[f"blocks.{i}.c2.w"] = b.c2.weight; ten[f"blocks.{i}.c2.b"] = b.c2.bias
    ten = {k: v.detach().float().contiguous() for k, v in ten.items()}
    man = {"tensors": {}, "cfg": m.cfg, "const": {"WS": list(FE.WS), "TMAX": FE.TMAX, "CTX48": FE.CTX48, "N_LAG": FE.N_LAG, "LAG_MIN": FE.LAG_MIN,
                                                  "F_LO": FE.F_LO, "F_HI": FE.F_HI, "CENTS": FE.CENTS, "N_BIN": FE.N_BIN, "D_IN": FE.D_IN, "E_FLOOR": FE.E_FLOOR}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import s0_artic as S
    x = S.held24()[1]["x"].astype(np.float32)[: 3 * N.SR]
    T = len(x) // N.HOP
    x = x[: T * N.HOP]
    with torch.no_grad():
        xx = torch.from_numpy(np.concatenate([np.zeros(FE.CTX48, np.float32), x]))[None]
        feat = front(xx, T)
        v, p = m(feat)
        f0 = FE.decode(v, p, 0.5)[0]
    x.astype("<f4").tofile(out / "fixture" / "x.f32")
    feat[0].T.contiguous().numpy().astype("<f4").tofile(out / "fixture" / "feat.f32")
    v[0].numpy().astype("<f4").tofile(out / "fixture" / "v.f32")
    f0.numpy().astype("<f4").tofile(out / "fixture" / "f0.f32")
    print("exported", len(ten), "tensors ->", out, "| T", T, "| voiced", float((f0 > 0).float().mean()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
