"""研究の基準の型 snv_p30_c(符号 = 凍結した CPC の文脈・train_snv_p3.PlayW = 登録の平均を引く → PCA 32 → 縮みつきの共分散で白くする → 級の正規化 128 → proj 64・回路 = train_snv_p1.Circ)を
Rust(crates/lightvc-core/src/p30_model.rs)へ移すための重みと一致検査の固定の入出力(paper_snv §6 段 P・製品の既定は変えない別の型)。
重み = state_dict の名前のまま(Pl. = PlayW・C. = Circ)。CPC の学習だけの予測の頭 cpc.W.* は書き出さない(manifest の名前 = Rust が読む名前)。級の正規化の s2・縮み・遅れ・TAU・層の正規化の eps は manifest の cfg(kind = p30)。
fixture/(模型の段)= 本物の男声の操作者(real/uirou_0)・その登録(real/uirou_enroll)・A0F の目標の参照(j 0〜2)・k = 7 で、EV.System の p3 の枝と同じ計算:
  操作者の統計(mu・白くする行列 Wm・級の正規化の delta)・目標の参照の演奏 a_ref と C の構造(z・鍵・値)・源の符号の文脈 c [T][256]・演奏 a [T][64]・回路の出力 Y [T][29]。
fixture_stream/(端から端)= export_snv_stream.py と同じ入力・描画器・励起の雑音で、波形 w と Y。
    cd training && uv run --no-sync python export_snv_p30.py --ckpt results/snv_p30_c/last.pt --out ../results/snv_p30_export
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rvoc as R
import snv_cmn as CM
import snv_p0_front as P
import snv_p1_eval as EV
import snv_p_feat as F
import train_rvoc as TR
import train_snv_p1 as T
import train_snv_p3 as P3

ROOT = Path(__file__).resolve().parent.parent
K_SEMI = 7


def load_model(ckpt: str) -> tuple[P3.PlayW, T.Circ, dict]:
    st = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
    assert st.get("play") == "wst" and st.get("circ", "x") == "x" and not st.get("xc0"), "PlayW + Circ の型だけ"
    Pl = P3.make_play(ROOT / st["cpc"], st["play"], ROOT / st["pca"], st.get("lags"), st.get("shrink"), ROOT / st["cb"], False).eval()
    Pl.proj.load_state_dict(st["proj"])
    C = T.Circ().eval()
    C.load_state_dict(st["C"])
    return Pl, C, st


def feat(x: np.ndarray, npz: Path) -> tuple[np.ndarray, dict]:
    x = x[: P.n_frames(len(x)) * P.H]
    d = P.analyze(x)
    ref = np.load(npz)
    v3 = P.W_YIN == 1200 and P.WIN_CAP_MS >= 1000.0 and P.TAU_UV_HZ == 200.0 and P.WIN_K == 3.0 and not P.CT_MODE
    for k in ("f0", "env", "per") if v3 else ():
        assert np.array_equal(d[k].astype(np.float32), ref[k]), f"{npz.name}: {k} が data/snv_p_feat と違う"
    f0, env, per = d["f0"].astype(np.float32), d["env"].astype(np.float32), d["per"].astype(np.float32)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return x, {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/snv_p30_c/last.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    fx, fs = out / "fixture", out / "fixture_stream"
    fx.mkdir(parents=True, exist_ok=True)
    fs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    Pl, C, st = load_model(a.ckpt)
    ten = {f"Pl.{k}": v.float() for k, v in Pl.state_dict().items() if not k.startswith("cpc.W.")} | {f"C.{k}": v.float() for k, v in C.state_dict().items()}
    cfg = {"kind": "p30", "D_Y": T.D_Y, "D_A": T.D_A, "ch": 256, "zdim": 256, "heads": 4, "dk": 64, "dv": 16, "dils": [1, 2, 4, 8, 16, 32] * 2, "attn_after": 3, "K_SEMI": K_SEMI,
           "cpc_dils": [1, 2, 4, 8, 16, 32], "pca_k": int(Pl.P.shape[1]), "lags": list(Pl.lags), "shrink": float(Pl.shrink), "cb_s2": float(Pl.cb_s2), "tau": float(CM.TAU), "ln_eps": 1e-5,
           "cpc": st["cpc"], "pca": st["pca"], "cb": st["cb"]}
    assert cfg["lags"] == [0], cfg["lags"]
    man = {"tensors": {}, "ckpt": a.ckpt, "step": st.get("step"), "cfg": cfg}
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
        st_op = Pl.mean_of(ry_o, rv_o, torch.ones(rv_o.shape, dtype=torch.bool))
        ry_t, rv_t = t(np.concatenate([u["y"] for u in refs], 1)), t(np.concatenate([u["v"] for u in refs]))
        rm_t = torch.ones(rv_t.shape, dtype=torch.bool)
        a_ref = Pl(ry_t, rv_t, Pl.mean_of(ry_t, rv_t, rm_t))
        sC = C.structure(ry_t, a_ref, rm_t)
        y, v = t(src["y"]), t(src["v"])
        lf = t(src["lf"]) + K_SEMI * T.LN2_12 * v
        c = Pl.ctx(y, v)
        aa = Pl(y, v, st_op)
        yo = C(aa, lf, v, sC)
    w = lambda d, name, x: np.ascontiguousarray(np.asarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype="<f4")).tofile(d / f"{name}.f32")
    w(fx, "op_y", ry_o[0].T), w(fx, "op_v", rv_o[0]), w(fx, "ref_y", ry_t[0].T), w(fx, "ref_v", rv_t[0])
    w(fx, "src_y", y[0].T), w(fx, "src_v", v[0]), w(fx, "src_lf_k", lf[0])
    w(fx, "op_mu", st_op[0][0]), w(fx, "op_wm", st_op[1][0]), w(fx, "op_delta", st_op[2][0])
    w(fx, "a_ref", a_ref[0].T), w(fx, "sC_z", sC["z"][0]), w(fx, "sC_k", sC["k"][0]), w(fx, "sC_v", sC["v"][0])
    w(fx, "c", c[0].T), w(fx, "a", aa[0].T), w(fx, "yo", yo[0].T)
    (fx / "shapes.json").write_text(json.dumps({"T": int(y.shape[-1]), "Tr_op": int(ry_o.shape[-1]), "Tr_t": int(ry_t.shape[-1])}))
    sr_ = torch.load(EV.CK_R, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).eval()
    gen.load_state_dict(sr_["ema"])
    pf = ROOT / "data/snv_p_feat"
    src_x, s2 = feat(F.load48(pf / "real/uirou_0.wav"), pf / "real/uirou_0.npz")
    op_x, o2 = feat(F.load48(pf / "real/uirou_enroll.wav"), pf / "real/uirou_enroll.npz")
    alt_x, _ = feat(F.load48(pf / "real/namikawa_0.wav"), pf / "real/namikawa_0.npz")
    rf = [feat(F.load48(ROOT / f"results/snv_a0/wav/{meta['targets'][0]}_{j}.wav"), pf / f"a0/{meta['targets'][0]}_{j}.npz") for j in (0, 1, 2)]
    with torch.no_grad():
        so = Pl.mean_of(t(o2["y"]), t(o2["v"]), torch.ones((1, len(o2["v"])), dtype=torch.bool))
        ry2, rv2 = t(np.concatenate([r["y"] for _, r in rf], 1)), t(np.concatenate([r["v"] for _, r in rf]))
        rm2 = torch.ones(rv2.shape, dtype=torch.bool)
        sC2 = C.structure(ry2, Pl(ry2, rv2, Pl.mean_of(ry2, rv2, rm2)), rm2)
        y2, v2 = t(s2["y"]), t(s2["v"])
        lf2 = t(s2["lf"]) + K_SEMI * T.LN2_12 * v2
        yo2 = C(Pl(y2, v2, so), lf2, v2, sC2)
        cond = torch.cat([yo2[:, :25], lf2[:, None], v2[:, None], yo2[:, 25:29].clamp(0, 1.5) * v2[:, None]], 1)
        exc = TR.excitation(t(s2["f0"] * 2 ** (K_SEMI / 12)), cond.shape[-1] * P.H, torch.Generator().manual_seed(0))
        wav = gen(cond, exc)[0]
    w(fs, "src_x", src_x), w(fs, "op_x", op_x), w(fs, "alt_x", alt_x)
    for j, (rx, _) in enumerate(rf):
        w(fs, f"ref{j}_x", rx)
    w(fs, "noise", exc[0, 1]), w(fs, "yo", yo2[0].T), w(fs, "w", wav.reshape(-1))
    shapes = {"T": int(y2.shape[-1]), "k": K_SEMI, "refs": len(rf), "ckpt": a.ckpt, "front": "results/pfront_export", "model": "results/snv_p30_export", "voc": "results/rvoc3hi/export_330k"}
    (fs / "shapes.json").write_text(json.dumps(shapes, indent=1))
    print("exported", len(ten), "tensors", off, "floats ->", out, "| T", y.shape[-1], "Tr_op", ry_o.shape[-1], "Tr_t", ry_t.shape[-1], "| stream", shapes, "rms", float(wav.pow(2).mean().sqrt()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
