"""F2(lvl.py の因果な c0 推定器)を Rust(crates/lightvc-core/src/lvl.rs)へ書き出す + 一致検査用の固定の入力と期待出力。
出力 <out>/: weights.bin・manifest.json({tensors, cfg, const})・fixture/(x.f32 = 48kHz の入力・c0n.f32 = F2 の出力・c0n_avg3.f32 = 製品の処方 = 因果 3 平均)。
名前: mel.fb・mel.win・inp・blks.i.c1・blks.i.c2・out(+ .w / .b)。
    CUDA_VISIBLE_DEVICES= uv run python export_lvl.py --ckpt ../results/f2_2/last.pt --out ../results/f2_2/export
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
import lvl as LV
import nvoc as N
import train_c3 as C3

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = LV.Lvl(st["cfg"]["ch"], tuple(st["cfg"]["dils"])).eval()
    m.load_state_dict(st["net"])
    front = CC.MelFront().eval()
    ten = {"mel.fb": front.mel.fb, "mel.win": front.mel.win, "inp.w": m.inp.weight, "inp.b": m.inp.bias, "out.w": m.out.weight, "out.b": m.out.bias}
    for i, b in enumerate(m.blks):
        ten[f"blks.{i}.c1.w"] = b.c1.weight; ten[f"blks.{i}.c1.b"] = b.c1.bias
        ten[f"blks.{i}.c2.w"] = b.c2.weight; ten[f"blks.{i}.c2.b"] = b.c2.bias
    ten = {k: v.detach().float().contiguous() for k, v in ten.items()}
    man = {"tensors": {}, "cfg": m.cfg, "const": {"HOP": N.HOP, "WIN": N.WIN, "NFFT": N.NFFT, "N_MEL": N.N_MEL, "C0_AVG": 3, "step": int(st.get("step", 0))}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import s0_artic as S
    x = S.held24()[0]["x"].astype(np.float32)[: 3 * N.SR]
    T = len(x) // N.HOP
    x = x[: T * N.HOP]
    with torch.no_grad():
        xt = torch.from_numpy(x)[None]
        c0n = m(front(torch.cat([torch.zeros(1, N.WIN - N.HOP), xt], 1)))[:, None]
        avg = C3.causal_avg(c0n, 3)
    x.astype("<f4").tofile(out / "fixture" / "x.f32")
    c0n[0, 0].numpy().astype("<f4").tofile(out / "fixture" / "c0n.f32")
    avg[0, 0].numpy().astype("<f4").tofile(out / "fixture" / "c0n_avg3.f32")
    print("exported", len(ten), "tensors ->", out, "| T", T, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
