"""段 P の製品経路(crates/lightvc-core/src/snv_stream.rs)の端から端の一致検査の固定の入出力(paper_snv §6 段 P)。
48 kHz の波形 → 前段 v3(snv_p0_front.analyze)→ E / C(--ckpt・train_snv_p11.EncS + train_snv_p1.Circ)→ rvoc3hi → 波形(先頭の DELAY を落とす前)。
入力 = 本物の男声の操作者(real/uirou_0)・その登録(real/uirou_enroll)・目標の参照 3 本(a0/<targets[0]>_j を 48 kHz へ)・k = 7・未来の書き換えの差し替え(real/namikawa_0)。
励起の雑音は書き出して Rust で差し込む(パルス列は Rust の Excite が作る)。特徴は data/snv_p_feat の値と一致することを確かめてから使う。
    cd training && uv run --no-sync python export_snv_stream.py --ckpt results/snv_p11_r/last.pt --out ../results/snv_stream_export
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rvoc as R
import snv_p0_front as P
import snv_p1_eval as EV
import snv_p_feat as F
import train_rvoc as TR
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parent.parent
K_SEMI = 7


def feat(x: np.ndarray, npz: Path) -> tuple[np.ndarray, dict]:
    x = x[: P.n_frames(len(x)) * P.H]
    d = P.analyze(x)
    ref = np.load(npz)
    for k in ("f0", "env", "per"):
        assert np.array_equal(d[k].astype(np.float32), ref[k]), f"{npz.name}: {k} が data/snv_p_feat と違う"
    f0, env, per = d["f0"].astype(np.float32), d["env"].astype(np.float32), d["per"].astype(np.float32)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return x, {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    fx = out / "fixture"
    fx.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    st = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    E, C = P11.EncS().eval(), T.Circ().eval()
    E.load_state_dict(st["Es"]), C.load_state_dict(st["C"])
    sr_ = torch.load(EV.CK_R, map_location="cpu", weights_only=False)
    cfg = sr_["cfg"]
    gen = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).eval()
    gen.load_state_dict(sr_["ema"])
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    pf = ROOT / "data/snv_p_feat"
    src_x, src = feat(F.load48(pf / "real/uirou_0.wav"), pf / "real/uirou_0.npz")
    op_x, op = feat(F.load48(pf / "real/uirou_enroll.wav"), pf / "real/uirou_enroll.npz")
    alt_x, _ = feat(F.load48(pf / "real/namikawa_0.wav"), pf / "real/namikawa_0.npz")
    refs = [feat(F.load48(ROOT / f"results/snv_a0/wav/{meta['targets'][0]}_{j}.wav"), pf / f"a0/{meta['targets'][0]}_{j}.npz") for j in (0, 1, 2)]
    t = lambda x: torch.from_numpy(x)[None]
    with torch.no_grad():
        ry_o, rv_o = t(op["y"]), t(op["v"])
        sE_op = E.structure(P11.xin(ry_o, rv_o), torch.ones(rv_o.shape, dtype=torch.bool))
        ry_t, rv_t = t(np.concatenate([r["y"] for _, r in refs], 1)), t(np.concatenate([r["v"] for _, r in refs]))
        sC = P11.structures(E, C, ry_t, rv_t, torch.ones(rv_t.shape, dtype=torch.bool))[1]
        y, v = t(src["y"]), t(src["v"])
        lf = t(src["lf"]) + K_SEMI * T.LN2_12 * v
        yo = C(E(P11.xin(y, v), sE_op)[0], lf, v, sC)
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = t(src["f0"] * 2 ** (K_SEMI / 12))
        exc = TR.excitation(f0, cond.shape[-1] * P.H, torch.Generator().manual_seed(0))
        w = gen(cond, exc)[0]
    wr = lambda name, x: np.ascontiguousarray(np.asarray(x, dtype="<f4")).tofile(fx / f"{name}.f32")
    wr("src_x", src_x), wr("op_x", op_x), wr("alt_x", alt_x)
    for j, (rx, _) in enumerate(refs):
        wr(f"ref{j}_x", rx)
    wr("noise", exc[0, 1].numpy()), wr("yo", yo[0].T.numpy()), wr("w", w.numpy().reshape(-1))
    shapes = {"T": int(y.shape[-1]), "k": K_SEMI, "refs": len(refs), "ckpt": a.ckpt, "front": "results/pfront_export", "model": "results/p_model_export", "voc": "results/rvoc3hi/export_330k"}
    (fx / "shapes.json").write_text(json.dumps(shapes, indent=1))
    print("exported", shapes, "| w", w.shape, "rms", float(w.pow(2).mean().sqrt()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
