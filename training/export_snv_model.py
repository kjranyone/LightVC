"""演奏の取り出し E(train_snv_p11.EncS)と回路 C(train_snv_p1.Circ)を Rust(crates/lightvc-core/src/p_model.rs)へ移すための重みと一致検査の固定の入出力(paper_snv §6 段 P)。
重み = state_dict の名前のまま(E. / C. の接頭)。固定の入出力 = 本物の男声の操作者(real/uirou_0)・その登録(real/uirou_enroll)・A0F の目標の参照(j 0〜2)・k = 7 で、
Python の EV.System と同じ計算: 構造(E の z・鍵・値 / C の z・鍵・値)・演奏 a [T][64]・回路の出力 Y [T][29]。
    cd training && uv run --no-sync python export_snv_model.py --ckpt results/snv_p11_r/last.pt --out ../results/p_model_export
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import snv_p1_eval as EV
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parent.parent
K_SEMI = 7


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    E, C = P11.EncS().eval(), T.Circ().eval()
    E.load_state_dict(st["Es"]), C.load_state_dict(st["C"])
    ten = {f"E.{k}": v.float() for k, v in E.state_dict().items()} | {f"C.{k}": v.float() for k, v in C.state_dict().items()}
    man = {"tensors": {}, "ckpt": a.ckpt, "step": st.get("step"), "cfg": {"D_Y": T.D_Y, "D_A": T.D_A, "ch": 256, "zdim": 256, "heads": 4, "dk": 64, "dv": 16, "dils": [1, 2, 4, 8, 16, 32] * 2, "attn_after": 3, "K_SEMI": K_SEMI}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].contiguous().numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    src = EV.utt("real/uirou_0")
    enr = [EV.utt("real/uirou_enroll")]
    refs = [EV.utt(f"a0/{meta['targets'][0]}_{j}") for j in (0, 1, 2)]
    t = lambda x: torch.from_numpy(x)[None]
    with torch.no_grad():
        ry_o, rv_o = t(np.concatenate([u["y"] for u in enr], 1)), t(np.concatenate([u["v"] for u in enr]))
        sE_op = E.structure(P11.xin(ry_o, rv_o), torch.ones(rv_o.shape, dtype=torch.bool))
        ry_t, rv_t = t(np.concatenate([u["y"] for u in refs], 1)), t(np.concatenate([u["v"] for u in refs]))
        rm_t = torch.ones(rv_t.shape, dtype=torch.bool)
        sE_t = E.structure(P11.xin(ry_t, rv_t), rm_t)
        a_ref = E(P11.xin(ry_t, rv_t), sE_t)[0]
        sC = C.structure(ry_t, a_ref, rm_t)
        y, v = t(src["y"]), t(src["v"])
        lf = t(src["lf"]) + K_SEMI * T.LN2_12 * v
        aa = E(P11.xin(y, v), sE_op)[0]
        yo = C(aa, lf, v, sC)
    fx = out / "fixture"
    w = lambda name, x: np.ascontiguousarray(x.detach().numpy().astype("<f4")).tofile(fx / f"{name}.f32")
    w("op_y", ry_o[0].T), w("op_v", rv_o[0]), w("ref_y", ry_t[0].T), w("ref_v", rv_t[0])
    w("src_y", y[0].T), w("src_v", v[0]), w("src_lf_k", lf[0])
    w("sE_op_z", sE_op["z"][0]), w("sE_op_k", sE_op["k"][0]), w("sE_op_v", sE_op["v"][0])
    w("sE_t_z", sE_t["z"][0]), w("a_ref", a_ref[0].T), w("sC_z", sC["z"][0]), w("sC_k", sC["k"][0]), w("sC_v", sC["v"][0])
    w("a", aa[0].T), w("yo", yo[0].T)
    (fx / "shapes.json").write_text(json.dumps({"T": int(y.shape[-1]), "Tr_op": int(ry_o.shape[-1]), "Tr_t": int(ry_t.shape[-1])}))
    print("exported", len(ten), "tensors", off, "floats ->", out, "| T", y.shape[-1], "Tr_op", ry_o.shape[-1], "Tr_t", ry_t.shape[-1], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
