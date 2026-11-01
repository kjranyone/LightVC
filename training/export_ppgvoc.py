"""内容 → 波形の生成器(train_ppgvoc.py の TabVoc・文脈の符号器 CtxEnc つき)の重みを Rust(crates/lightvc-core/src/rvoc.rs)へ書き出す + 一致検査用の固定の入力と期待出力。
export_rvoc.py と同じ形式(weights.bin・manifest.json・fixture/)。条件 = 生の 231 行 [c0n・Σp T / 10 (24)・log f0・有声・周期性 × 有声 (4)・C1 の事後 (200)]
(文脈の符号器を通す前)。Rust は生の条件を受け取り、文脈の符号器のずれを包絡の行に足してから pre に入れる(TabVoc.make と同じ順)。
名前 = PyTorch のモジュールの経路 + .w / .b(weight_norm は実効の重みに畳む): pre / up.i / src.i / res.i.j.c1.m / res.i.j.c2.m / post / ctx.inp / ctx.blocks.i / ctx.out。
遅い音色の揺らぎ(ckpt の slow・diag_ppgvoc21): slow.B [24, k]・slow.fir [k, L](最小位相)・slow.gain [k] と cfg.slow {k, L, sigma, seed}(共通の乱数・train_ppgvoc.slow_noise_prng と同じ)。
    CUDA_VISIBLE_DEVICES= uv run python export_ppgvoc.py --ckpt ../results/diag_ppgvoc14/last_ema.pt --out ../results/diag_ppgvoc14/export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import rvoc as R
import train_ppgvoc as P
import train_rvoc as TR


def tensors(m: torch.nn.Module) -> dict:
    out = {}
    for name, mod in m.named_modules():
        if isinstance(mod, R.CConv):
            out[f"{name}.w"] = mod.conv.weight.detach().float().contiguous()
            out[f"{name}.b"] = mod.conv.bias.detach().float().contiguous()
        elif name == "ctx.out":
            out[f"{name}.w"] = mod.weight.detach().float().contiguous()
            out[f"{name}.b"] = mod.bias.detach().float().contiguous()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--slow_sigma", type=float, default=0.0, help="遅い音色の揺らぎの σ(0 = ckpt の値)")
    ap.add_argument("--slow_seed", type=int, default=20261019, help="遅い音色の揺らぎの共通の乱数の種(一致検査の参照と Rust の既定)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert st.get("spk", "table") == "table" and st.get("ppg", "soft") in ("soft", "hard"), "表の経路・柔らかい事後(hard は診断 diag_ppgvoc65 だけ)"
    if st.get("ppg", "soft") == "hard" or st.get("c0_tab") or st.get("c2_const"):
        print("診断の生成器: Rust では LIGHTVC_DIAG_Q=onehot・LIGHTVC_DIAG_C2 =", st.get("c2_const"), "・c0 の流れ LIGHTVC_DIAG_STREAM(表", st.get("c0_tab"), ")が要る", flush=True)
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    P.slow_from_ckpt(st)
    if st.get("cvin"):
        cp = st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"])
        assert cp.endswith(".pt"), "Rust へは因果な生徒 C2(.pt)だけ"
        P.cvin_setup(cp, "cpu")
    if P.SLOW["on"]:
        P.SLOW["causal"] = True
        P.SLOW["sigma"] = a.slow_sigma if a.slow_sigma > 0 else P.SLOW["sigma"]
        P.SLOW["seed_fixed"] = a.slow_seed
    m = P.init_gen("cpu", "table")
    m.load_state_dict(st["ema"])
    m.eval()
    ten = tensors(m)
    cfg = dict(m.cfg)
    cfg["ppg_k"] = P.K
    cfg["cv_d"] = int(P.CVIN["d"]) if P.CVIN["on"] else 0
    cfg["c0low"] = {"v": [float(x) for x in P.C0LOW["v"]], "a": float(P.C0LOW["a"])} if P.C0LOW["on"] else None
    cfg["ctx"] = {"ch": P.CTX["ch"], "dils": [1, 2, 4, 8, 16, 32, 64]} if P.CTX["on"] else None
    cfg["slow"] = None
    if P.SLOW["on"]:
        ten["slow.B"] = P.SLOW["B"].float().contiguous()
        ten["slow.fir"] = P.SLOW["fir"].float().contiguous()
        ten["slow.gain"] = P.SLOW["fir_gain"].float().contiguous()
        cfg["slow"] = {"k": int(P.SLOW["B"].shape[1]), "L": int(P.SLOW["fir"].shape[1]), "sigma": float(P.SLOW["sigma"]), "seed": int(a.slow_seed)}
    man = {"tensors": {}, "cfg": cfg, "const": {"SR": N.SR, "HOP": N.HOP, "DELAY": N.DELAY, "D_COND": int(cfg["d_cond"]), "UPS": list(R.UPS), "step": int(st.get("step", 0)),
                                                "H_MAX": N.H_MAX, "H_ROLL": N.H_ROLL, "K_MAX": N.K_MAX, "macs_per_second": R.macs_per_second(m)}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    T = a.frames
    rng = np.random.default_rng(0)
    t = np.arange(T)
    f0 = np.where((t // 60) % 5 == 4, 0.0, 260.0 + 60 * np.sin(t / 37.0)).astype(np.float32)
    vo = (f0 > 0).astype(np.float32)
    lg = np.cumsum(rng.normal(0, 0.6, (P.K, T)), 1).astype(np.float32)
    p = torch.from_numpy(lg).softmax(0).numpy()
    tab = (rng.normal(0, 2.0, (24, P.K))).astype(np.float32)
    per = (0.6 + 0.2 * rng.random(4)).astype(np.float32)
    c0 = (4.0 + 0.8 * np.sin(t / 20.0)).astype(np.float32)
    lf = np.where(vo > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    cvf = (0.7 * rng.normal(0, 1.0, (cfg["cv_d"], T))).astype(np.float32) if cfg["cv_d"] else None
    with torch.no_grad():
        tt = lambda x: torch.from_numpy(np.asarray(x, np.float32))
        raw = P.cond_ppg(tt(p)[None], tt(c0)[None, None], tt(lf)[None, None], tt(vo)[None, None], tt(tab)[None], tt(per)[None])
        if cvf is not None:
            raw = torch.cat([raw, tt(cvf)[None]], 1)
        cond2 = m.make(tt(p)[None], tt(c0)[None, None], tt(lf)[None, None], tt(vo)[None, None], (tt(tab)[None], tt(per)[None]), cvf=None if cvf is None else tt(cvf)[None])
        exc = TR.excitation(torch.from_numpy(f0)[None], T * N.HOP, torch.Generator().manual_seed(0))
        y = m(cond2, exc)[0].numpy()
    assert raw.shape[1] == cfg["d_cond"], (raw.shape, cfg["d_cond"])
    exc_blocks = exc[0].numpy().reshape(2, T, N.HOP).transpose(1, 0, 2).copy()
    raw[0].numpy().T.copy().astype("<f4").tofile(out / "fixture" / "cond.f32")
    exc_blocks.astype("<f4").tofile(out / "fixture" / "exc.f32")
    y.astype("<f4").tofile(out / "fixture" / "y.f32")
    f0.astype("<f4").tofile(out / "fixture" / "f0.f32")
    dctx = float((cond2 - raw).abs().max())
    print("exported", len(ten), "tensors", off, "floats ->", out, "| d_cond", cfg["d_cond"], "| ctx", P.CTX["on"], "| max|ctx のずれ + 揺らぎ|", round(dctx, 4), "| slow", cfg["slow"], "| y rms", float(np.sqrt((y ** 2).mean())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
