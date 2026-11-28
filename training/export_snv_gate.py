"""研究の形 snv_p30_c+mask:<門>(paper_snv §6 段 P)の前段の門を Rust(crates/lightvc-core/src/p_gate.rs と p_front.rs の重みの経路)へ移すための重みと一致検査の固定の入出力(製品の既定は変えない別の前段)。
門 = train_snv_mask.load の Gated(門の網 gate・重みの網 mask)・features = 門の確率 → hysteresis(on, off, thr)→ 開いたフレームだけ √sigmoid(重み)・dec = myin で有声と高さ(snv_p0_front.f0_track_myin)。
出力 <out>/(既定 = results/<門のタグ>_export)= 前段 v3 の定数と mel.fb・dct(export_pfront.py と同じ値・p_front.rs がそのまま読む)+ Gated の名前のままの重み(gate.* / mask.*・網の前段の窓と帯の行列の buffer を含む)・cfg(kind = gate)。
fixture/ = 操作者の発話(real/uirou_0)の途中に開発の音楽(train_snv_sep.bg_split の dev を read_bg で読む)を発話の RMS に対して 0 dB で足した入力の段ごとの値(門のロジット・開閉・重みのロジット・myin の高さ・y / lf / v)と、足さない入力の門の値と y。
fixture_stream/ = 端から端(前段 + 門と重み → p30 → rvoc3hi)= export_snv_p30.py の fixture_stream と同じ登録・参照・k・励起の雑音の作りで、源の入力だけ音楽を足したもの(操作者の登録も門を通す = snv_p1_eval.System.prep)。
  門が開いた後の波形は標本ごとには合わない: 重みの網の f32 の丸めで myin の高さが相対で最大 5.5e-6 ずれ、励起の位相に積もる(波形の差 RMS の 1.1 %・振幅のスペクトログラム STFT 1024 / hop 240 の相対の差 2.0e-4)。
  Rust の試験は開くまでを標本ごとの線で、開いた後を Y の線と振幅のスペクトログラムの相対の差 < 1e-3 で比べる。
    cd training && nice -n 19 uv run --no-sync python export_snv_gate.py --gate results/snv_p45_gate_t80/last.pt [--out ../results/snv_p45_gate_t80_export]
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter

import export_snv_p30 as X30
import pae as PA
import rvoc as R
import snv_p0_front as P
import snv_p1_eval as EV
import snv_p_feat as F
import train_rvoc as TR
import train_snv_mask as MK
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = Path(__file__).resolve().parent.parent
K_SEMI = 7
MUS = (2.0, 6.0)
FADE = 0.02
SNR = 0.0
TRIES = 24
FRONT_KEYS = ("SR", "H", "W_YIN", "LP_HZ", "LP_ORDER", "K_CAND", "C_MAX", "BETA", "LAMBDA", "GAMMA", "U_COST", "HOLD_BASE", "HOLD_STEP",
              "LB_HZ", "LB_MIN", "TMIN", "TMAX", "E_FLOOR", "NFFT_P", "NFFT_X", "NFFT_S", "ETA", "PAD_L")


def front_consts() -> tuple[dict, dict]:
    fb, dct = PA._np_consts()
    b, aa = butter(P.LP_ORDER, P.LP_HZ / (P.SR / 2))
    ten = {"mel.fb": torch.from_numpy(fb).float(), "dct": torch.from_numpy(dct[:25]).float()}
    const = {k: getattr(P, k) for k in FRONT_KEYS}
    const |= {"butter_b": b.tolist(), "butter_a": aa.tolist(), "NFFT": PA.NFFT, "HB": [list(x) for x in PA.HB], "C0_SIL": PA.C0_SIL}
    if P.WIN_CAP_MS < 1000.0:
        const["ENV_CAP_MS"] = P.WIN_CAP_MS
    if P.TAU_UV_HZ != 200.0:
        const["TAU_UV_HZ"] = P.TAU_UV_HZ
    if os.environ.get("SNV_F0V4") == "1":
        const["F0_MODE"] = "v4_dual"
    assert P.WIN_K == 3.0 and not P.CT_MODE, "窓の倍率と窓の形の診断のスイッチは移植に含めていない"
    assert const["ETA"] == 0.0, "倍音の検査(ETA > 0)は移植に含めていない"
    ref = ROOT / "results/pfront_export/manifest.json"
    if ref.exists() and P.W_YIN == 1200 and "ENV_CAP_MS" not in const and "TAU_UV_HZ" not in const and "F0_MODE" not in const:
        man = json.loads(ref.read_text())
        assert json.loads(json.dumps(const)) == man["const"], "前段の定数が results/pfront_export と違う"
        wb = np.fromfile(ref.parent / "weights.bin", "<f4")
        for k, v in ten.items():
            o, sh = man["tensors"][k]
            assert np.array_equal(wb[o:o + int(np.prod(sh))].reshape(sh), v.numpy()), f"{k} が results/pfront_export と違う"
    return ten, const


def gate_states(G: torch.nn.Module, x: np.ndarray) -> dict:
    Tn = P.n_frames(len(x))
    xin = torch.from_numpy(np.asarray(x[: Tn * P.H], np.float32))[None]
    with torch.no_grad():
        zg = G.gate(xin)[0, 0][:Tn]
        zm = G.mask(xin)[0][:, :Tn]
    op = MK.hysteresis(torch.sigmoid(zg.double()), *G.hyst, G.thr)
    return {"zg": zg.numpy(), "zm": zm.T.contiguous().numpy(), "op": op.numpy()}


def mixed(x: np.ndarray, src: str) -> np.ndarray:
    a, b = int(MUS[0] * P.SR), int(MUS[1] * P.SR)
    g = np.random.default_rng(zlib.crc32(f"r2|{Path(src).stem}".encode()))
    m = SEP.read_bg(src, b - a, g) * np.sqrt((x.astype(np.float64) ** 2).mean()) * 10 ** (-SNR / 20)
    nf = int(FADE * P.SR)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
    m[:nf] *= ramp
    m[-nf:] *= ramp[::-1]
    y = x.astype(np.float64).copy()
    y[a:b] += m
    return y.astype(np.float32)


def transitions(op: np.ndarray) -> list[list[int]]:
    d = np.diff(np.r_[0, op.astype(np.int64)])
    return [[int(t), int(d[t] > 0)] for t in np.nonzero(d)[0]]


def pick_music(G: torch.nn.Module, x: np.ndarray) -> tuple[str, np.ndarray, dict, list]:
    pool = SEP.bg_split()["dev"]["music"]
    i0 = zlib.crc32(b"r2|fixture") % len(pool)
    a, b = int(MUS[0] * P.SR) // P.H, int(MUS[1] * P.SR) // P.H
    for j in range(TRIES):
        src = pool[(i0 + j) % len(pool)]
        y = mixed(x, src)
        s = gate_states(G, y)
        tr = transitions(s["op"])
        ons, offs = [t for t, u in tr if u], [t for t, u in tr if not u]
        if ons and offs and a <= ons[0] < b and offs[-1] >= b + G.hyst[1] and not s["op"][-1] and not s["op"][: a].any():
            return src, y, s, tr
    raise SystemExit("開いて閉じる遷移を含む開発の音楽が見つからない")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--p30", default="results/snv_p30_c/last.pt")
    ap.add_argument("--voc-export", default="results/rvoc3hi/export_330k")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = Path(a.gate).parent.name
    out = Path(a.out) if a.out else ROOT / "results" / f"{tag}_export"
    keep = {ROOT / "results" / n for n in ("pfront_export", "pfront_export_p100", "pfront_export_p124", "p_model_export", "rvoc3hi/export_330k", "snv_p123_s/export_20k", "snv_stream_export", "snv_p30_export", "snv_p45_gate_t80_export", "snv_p45_gate_t80_export_v3", "snv_p126_gate_export")}
    assert out.resolve() not in {k.resolve() for k in keep}, f"{out} は製品か既存の書き出し"
    fx, fs = out / "fixture", out / "fixture_stream"
    fx.mkdir(parents=True, exist_ok=True)
    fs.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    st = torch.load(ROOT / a.gate, map_location="cpu", weights_only=False)
    G = MK.load(ROOT / a.gate, "cpu").eval()
    assert G.kind == "gate" and G.dec == "myin" and G.hyst, (G.kind, G.dec, G.hyst)
    gd, md = [b.d for b in G.gate.blocks], [b.d for b in G.mask.blocks]
    assert st["cfg"].get("dil", gd) == gd, ("門の網は train_snv_mask.load で既定の DIL", st["cfg"].get("dil"), gd)
    ten, const = front_consts()
    ten |= {k: v.detach().float() for k, v in G.named_parameters()} | {k: v.detach().float() for k, v in G.named_buffers()}
    ms = torch.load(ROOT / st["cfg"]["mask"], map_location="cpu", weights_only=False)
    cfg = {"kind": "gate", "gate": a.gate, "mask": st["cfg"]["mask"], "step_gate": st.get("step"), "step_mask": ms.get("step"), "hyst": list(G.hyst), "thr": float(G.thr), "gamma": float(G.gamma),
           "inst_mask": bool(getattr(G, "inst_mask", False)),
           "dec": G.dec, "ch": int(G.gate.inp.out_channels), "gate_dil": gd, "mask_dil": md, "nb_gate": int(G.gate.head.out_channels), "nb_mask": int(G.mask.head.out_channels),
           "nfft": SEP.NFFT, "nfft2": SEP.NFFT2, "d_in": SEP.D_IN, "ln_eps": float(G.gate.ln.eps), "log_eps": 1e-6}
    if os.environ.get("SNV_F0V4") == "1":
        cfg["f0_mode"] = "v4_dual"
    man = {"tensors": {}, "const": const, "cfg": cfg}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].contiguous().numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    pf = ROOT / "data/snv_p_feat"
    clean, _ = X30.feat(F.load48(pf / "real/uirou_0.wav"), pf / "real/uirou_0.npz")
    src, x, s, tr = pick_music(G, clean)
    u = MK.features(G, x, "cpu")
    assert np.array_equal(s["op"].astype(np.float64).mean(), u["gate_open"]), "門の開閉の再計算が features と違う"
    sc = gate_states(G, clean)
    uc = MK.features(G, clean, "cpu")
    if os.environ.get("SNV_F0V4") == "1":
        import snv_f0v4 as V4
        fn_c, fv_c, bn_c = V4.f0_dual_track(torch.from_numpy(clean))
        rc = MK.features_with_mask(clean, torch.ones((P.n_frames(len(clean)), MK.NB), dtype=torch.float64), (fn_c, bn_c), f0_env=torch.where((fv_c <= 0) & (fn_c > 0), fn_c, fv_c))
    else:
        _, rc = X30.feat(clean, pf / "real/uirou_0.npz")
    clean_same = bool(np.array_equal(uc["y"], rc["y"]) and np.array_equal(uc["f0"], rc["f0"]))
    w = lambda d, name, v: np.ascontiguousarray(np.asarray(v, dtype="<f4")).tofile(d / f"{name}.f32")
    w(fx, "x", x), w(fx, "gate_logit", s["zg"]), w(fx, "state", s["op"]), w(fx, "mask_logit", s["zm"]), w(fx, "f0", u["f0"]), w(fx, "y", u["y"].T), w(fx, "lf", u["lf"]), w(fx, "v", u["v"])
    w(fx, "clean_x", clean), w(fx, "clean_gate_logit", sc["zg"]), w(fx, "clean_state", sc["op"]), w(fx, "clean_y", uc["y"].T)
    mon_arr = (s["op"] > 0.5) | ((1.0 / (1.0 + np.exp(-s["zg"].astype(np.float64))) >= G.thr) if getattr(G, "inst_mask", False) else False)
    fmask = int(np.nonzero(mon_arr)[0][0]) if mon_arr.any() else int(tr[0][0])
    (fx / "shapes.json").write_text(json.dumps({"T": int(len(u["v"])), "nb": int(s["zm"].shape[1]), "music": src, "mus_s": list(MUS), "snr_db": SNR, "fade_s": FADE, "transitions": tr,
                                                "first_mask": fmask, "gate_open": float(u["gate_open"]), "clean_gate_open": float(uc["gate_open"]), "clean_same_as_v3": clean_same}, indent=1))
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    op_x, _ = X30.feat(F.load48(pf / "real/uirou_enroll.wav"), pf / "real/uirou_enroll.npz")
    alt_x, _ = X30.feat(F.load48(pf / "real/namikawa_0.wav"), pf / "real/namikawa_0.npz")
    rf = [X30.feat(F.load48(ROOT / f"results/snv_a0/wav/{meta['targets'][0]}_{j}.wav"), pf / f"a0/{meta['targets'][0]}_{j}.npz") for j in (0, 1, 2)]
    uo = MK.features(G, op_x, "cpu")
    if os.environ.get("SNV_F0V4") == "1":
        import snv_f0v4 as V4
        fn_o, fv_o, bn_o = V4.f0_dual_track(torch.from_numpy(op_x))
        ro = MK.features_with_mask(op_x, torch.ones((P.n_frames(len(op_x)), MK.NB), dtype=torch.float64), (fn_o, bn_o), f0_env=torch.where((fv_o <= 0) & (fn_o > 0), fn_o, fv_o))
    else:
        _, ro = X30.feat(op_x, pf / "real/uirou_enroll.npz")
    op_same = bool(np.array_equal(uo["y"], ro["y"]))
    Pl, C, _ = X30.load_model(a.p30)
    ck_r = ROOT / os.environ["SNV_RCK"] if os.environ.get("SNV_RCK") else EV.CK_R
    sr_ = torch.load(ck_r, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).eval()
    gen.load_state_dict(sr_["ema"])
    gen.env_smooth = float(sr_.get("env_smooth", 0.0))
    esm_val = 0.0 if os.environ.get("SNV_ESM") == "0" else (gen.env_smooth if os.environ.get("SNV_ESM") else 0.0)
    t = lambda v: torch.from_numpy(v)[None]
    with torch.no_grad():
        so = Pl.mean_of(t(uo["y"]), t(uo["v"]), torch.ones((1, len(uo["v"])), dtype=torch.bool))
        ry2, rv2 = t(np.concatenate([r["y"] for _, r in rf], 1)), t(np.concatenate([r["v"] for _, r in rf]))
        rm2 = torch.ones(rv2.shape, dtype=torch.bool)
        sC2 = C.structure(ry2, Pl(ry2, rv2, Pl.mean_of(ry2, rv2, rm2)), rm2)
        y2, v2 = t(u["y"]), t(u["v"])
        lf2 = t(u["lf"]) + K_SEMI * T.LN2_12 * v2
        yo2 = C(Pl(y2, v2, so), lf2, v2, sC2)
        env2 = EV.smooth_env(gen, yo2[:, :25]) if esm_val > 0 else yo2[:, :25]
        cond = torch.cat([env2, lf2[:, None], v2[:, None], yo2[:, 25:29].clamp(0, 1.5) * v2[:, None]], 1)
        exc = TR.excitation(t(u["f0"] * 2 ** (K_SEMI / 12)), cond.shape[-1] * P.H, torch.Generator().manual_seed(0))
        wav = gen(cond, exc)[0]
    w(fs, "src_x", x), w(fs, "op_x", op_x), w(fs, "alt_x", alt_x)
    for j, (rx, _) in enumerate(rf):
        w(fs, f"ref{j}_x", rx)
    w(fs, "noise", exc[0, 1].numpy()), w(fs, "yo", yo2[0].T.numpy()), w(fs, "w", wav.reshape(-1).numpy())
    shapes = {"T": int(y2.shape[-1]), "k": K_SEMI, "refs": len(rf), "gate": a.gate, "ckpt": a.p30, "front": str(out.resolve().relative_to(ROOT.resolve())), "model": "results/snv_p30_export",
              "voc": a.voc_export, "env_smooth": esm_val, "op_gate_open": float(uo["gate_open"]), "op_same_as_v3": op_same}
    (fs / "shapes.json").write_text(json.dumps(shapes, indent=1))
    print("exported", len(ten), "tensors", off, "floats ->", out, flush=True)
    print("音楽の区間", MUS, "s・遷移(フレーム, 1 = 開く)", tr, "・開いた割合", round(float(u["gate_open"]), 4), "・myin の有声", round(float(u["v"].mean()), 4), flush=True)
    print("音楽なし: 門の開いた割合", float(uc["gate_open"]), "・前段 v3 と 1 bit 同じ", clean_same, "/ 操作者の登録: 開いた割合", float(uo["gate_open"]), "・同じ", op_same, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
