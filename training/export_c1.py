"""C1 / C2(c1_content.py の因果な内容符号器・C2 = 連続の内容の特徴でコードブックなし)を Rust(crates/lightvc-core/src/c1.rs)へ書き出す + 一致検査用の固定の入力と期待出力。
出力 <out>/: weights.bin・manifest.json({tensors, cfg, const: PRE ほか})・fixture/(x.f32・logit.f32 [T][K]・unit.f32 [T])。期待出力は製品と同じ規約
(train_c1.prime = 無音 PRE サンプルを先に流し、最後の T フレーム)。名前: mel.fb・mel.win・inp・blocks.i.c1・blocks.i.c2・out(+ .w / .b)。
    CUDA_VISIBLE_DEVICES= uv run python export_c1.py --dir ../results/c1_1 --out ../results/c1_1/export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import nvoc as N
import train_c1 as T1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(Path(a.dir) / "last.pt", map_location="cpu", weights_only=False)
    m = CC.C1(st["cfg"]["k"], st["cfg"]["ch"], tuple(st["cfg"]["dils"])).eval()
    m.load_state_dict(st["net"])
    front = CC.MelFront().eval()
    ten = {"mel.fb": front.mel.fb, "mel.win": front.mel.win, "inp.w": m.inp.weight, "inp.b": m.inp.bias, "out.w": m.out.weight, "out.b": m.out.bias,
           **({"codebook": torch.load(Path(a.dir) / "codebook.pt").float()} if (Path(a.dir) / "codebook.pt").exists() else {})}
    for i, b in enumerate(m.blocks):
        ten[f"blocks.{i}.c1.w"] = b.c1.weight; ten[f"blocks.{i}.c1.b"] = b.c1.bias
        ten[f"blocks.{i}.c2.w"] = b.c2.weight; ten[f"blocks.{i}.c2.b"] = b.c2.bias
    ten = {k: v.detach().float().contiguous() for k, v in ten.items()}
    man = {"tensors": {}, "cfg": m.cfg, "const": {"PRE": int(T1.PRE), "HOP": N.HOP, "WIN": N.WIN, "N_MEL": N.N_MEL, "step": int(st.get("step", 0))}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import s0_artic as S
    x = S.held24()[2]["x"].astype(np.float32)[: 3 * N.SR]
    T = len(x) // N.HOP
    x = x[: T * N.HOP]
    with torch.no_grad():
        lg = m(front(torch.from_numpy(T1.prime(x))[None]))[..., -T:][0]
    x.astype("<f4").tofile(out / "fixture" / "x.f32")
    lg.T.contiguous().numpy().astype("<f4").tofile(out / "fixture" / "logit.f32")
    lg.argmax(0).float().numpy().astype("<f4").tofile(out / "fixture" / "unit.f32")
    print("exported", len(ten), "tensors ->", out, "| T", T, "| K", st["cfg"]["k"], "| PRE", T1.PRE, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
