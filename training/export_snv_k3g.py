"""研究の候補 K3G(paper_snv §6 段 P)= ens:snv_p24_c+G|snv_p30_c+G|snv_p91e_c+G(G = +mask:<門>+fb:anch)を Rust(crates/lightvc-core/src/k3g_model.rs・p_fixenv.rs)へ移すための重みと一致検査の固定の入出力(製品の既定は変えない別の型)。
型 = snv_p30_c と同じ組(PlayW + Circ)を 3 つ・回路の出力(29 次元)を平均して 1 回描く(snv_p1_eval.Ens)。門が開いたフレームは、源の包絡の一律の伸縮(錨・snv_p0c_anchor)に差し替える(snv_p1_eval.System.fallback の anch)。
envc の印のある組(snv_p91e_c)の符号だけ、固定の窓の包絡(snv_p0_front.fixed_env・40 ms のハン・左寄せ)を読む: 源・操作者の登録・目標の構造の鍵。回路の値・錨・描画は元の包絡。
重み = state_dict の名前のまま、組ごとに m<i>. を前に付ける(m0.Pl.* = PlayW・m0.C.* = Circ)。CPC の学習だけの予測の頭 cpc.W.* は書き出さない。mel.fb・dct は前段の書き出しと同じ値(固定の窓の包絡と錨が読む)。
cfg(kind = k3g): 組の一覧(prefix・envc・cb_s2・shrink)・fixed_env(窓の長さ・FFT の点数・対数の床)・anchor(α の格子・当てはめの帯の上限・mel の帯の中心の周波数)。
fixture/(模型の段・源 = 操作者の発話に開発の音楽を足したもの = 門が開く)= 入力(y・固定の窓の包絡つきの yc・有声・高さ・門)と、組ごとの操作者の統計・目標の構造・符号の文脈・演奏・回路の出力、錨、平均の出力。x と fenv = 固定の窓の包絡の流しの一致検査。
fixture_stream/(端から端・音楽なし)と fixture_stream_music/(端から端・門が開く)= export_snv_p30.py・export_snv_gate.py と同じ操作者の発話・登録・目標の参照・k = 7・励起の雑音の作り。前段 = 門の書き出し(results/<門>_export)。
基準の計算は snv_p1_eval.System / Ens をそのまま呼ぶ(CPU)。組に分けた再計算と「平均の後に錨を 1 回」の形が Ens の出力と合うことを書き出しの中で確かめる。
    cd training && SNV_ENVC=1 nice -n 19 uv run --no-sync python export_snv_k3g.py [--out ../results/snv_k3g_export]
SNV_WF16=1(段 P94)= 回路と CPC のフレームごとに読む重みを f16 に丸めた系で書き出す(cfg.half = true・Rust は f16 で持つ)。SNV_KV16=1 = 注意の鍵と値も(cfg.kv16)。
    cd training && SNV_ENVC=1 SNV_WF16=1 nice -n 19 uv run --no-sync python export_snv_k3g.py --out ../results/snv_k3g16_export
"""
from __future__ import annotations

import os

assert os.environ.get("SNV_ENVC"), "SNV_ENVC=1 で起動する(目標の参照の固定の窓の包絡 yc を train_snv_p1.load_utt が付ける)"
os.environ["SNV_DEV"] = "cpu"

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import export_snv_gate as XG
import export_snv_p30 as X30
import snv_cmn as CM
import snv_p0_front as P
import snv_p0c_anchor as AN
import snv_p1_eval as EV
import snv_p_feat as F
import train_rvoc as TR
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
K_SEMI = 7
WIN_MS = 40.0
MEMBERS = "snv_p24_c,snv_p30_c,snv_p91e_c"
KEEP = ("pfront_export", "pfront_export_p100", "pfront_export_p124", "p_model_export", "rvoc3hi/export_330k", "snv_p123_s/export_20k", "snv_stream_export", "snv_p30_export", "snv_p45_gate_t80_export", "snv_p45_gate_t80_export_v3", "snv_p126_gate_export", "snv_k3g_export", "snv_k3g16_export", "snv_k3g16kv_export", "snv_p126_k3g16kv_export")


def t(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x)[None]


def w(d: Path, name: str, x) -> None:
    np.ascontiguousarray(np.asarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype="<f4")).tofile(d / f"{name}.f32")


def env_of(y: np.ndarray) -> np.ndarray:
    return np.concatenate([y[:1] * 10 + T.PA.C0_SIL, y[1:25] * 10], 0)


@torch.no_grad()
def member_stage(p: EV.System, src: dict, stat: list[dict], refs: list[dict], S: dict, k: int) -> dict:
    u = p.prep(src)
    us = [p.prep(q) for q in stat]
    y, v = t(u["y"]), t(u["v"])
    lf = t(u["lf"]) + k * T.LN2_12 * v
    yk = t(p.cy(u)["y"])
    st = p.op_stat(stat)
    ry, rv, rm, _ = EV.struct_inputs(refs)
    ryc = EV.struct_inputs([p.cy(r) for r in refs])[0]
    a_ref = p.Pl(ryc, rv, p.Pl.mean_of(ryc, rv, rm))
    S2 = p.C.structure(ry, a_ref, rm)
    if os.environ.get("SNV_KV16"):
        S2 = S2 | {q: S2[q].half().float() for q in ("k", "v")}
    assert all(torch.equal(S[q], S2[q]) for q in ("z", "k", "v")), "構造の再計算が System.structure と違う"
    a = p.Pl(yk, v, st)
    yo = p.C(a, lf, v, S)
    yf = yo
    for fk in p.fb:
        yf = p.fallback(fk, u, y, v, a, yf, stat, S)
    ref = p.convert(src, stat, S, k, with_y=True)[1]
    assert torch.equal(yf, ref), "組に分けた再計算が System.convert と違う"
    mu_t = env_of(p.hmu[id(S)].numpy())
    mu_o = EV.mu_of(us)[0].numpy()
    return {"u": u, "uo": us, "y": y, "v": v, "lf": lf, "yk": yk, "st": st, "ry": ry, "rv": rv, "ryc": ryc, "a_ref": a_ref, "c": p.Pl.ctx(yk, v), "a": a, "yo": yo, "yf": yf,
            "yw": p.anchor(u, stat, S), "mu_o": mu_o, "mu_t": mu_t, "alpha": AN.fit_alpha(mu_o, mu_t)}


@torch.no_grad()
def render_full(sys: EV.Ens, gen: torch.nn.Module, src: dict, yo: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    u = sys.parts[0].prep(src)
    v = t(u["v"])
    lf = t(u["lf"]) + k * T.LN2_12 * v
    cond = torch.cat([EV.smooth_env(gen, yo[:, :25]), lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
    exc = TR.excitation(t(u["f0"] * 2 ** (k / 12)), cond.shape[-1] * P.H, torch.Generator().manual_seed(0))
    return gen(cond, exc)[0], exc


def stream_fixture(fs: Path, sys: EV.Ens, gen: torch.nn.Module, x: np.ndarray, stat: list[dict], S: list, waves: dict, shapes: dict) -> dict:
    fs.mkdir(parents=True, exist_ok=True)
    src = {"wav": x}
    wc, yo = sys.convert(src, stat, S, K_SEMI, with_y=True)
    wav, exc = render_full(sys, gen, src, yo, K_SEMI)
    delay = int(getattr(gen, "delay", EV.N.DELAY))
    assert np.array_equal(wav.numpy()[delay:].astype(np.float32), wc), "描画の再計算が Ens.convert の波形と違う"
    u = sys.parts[0].prep(src)
    w(fs, "src_x", x), w(fs, "op_x", waves["op"]), w(fs, "alt_x", waves["alt"])
    for j, rx in enumerate(waves["refs"]):
        w(fs, f"ref{j}_x", rx)
    w(fs, "noise", exc[0, 1]), w(fs, "yo", yo[0].T), w(fs, "w", wav.reshape(-1))
    tr = XG.transitions(u["gate"] > 0.5)
    sh = shapes | {"T": int(yo.shape[-1]), "transitions": tr, "gate_open": float(u["gate_open"]), "rms": float(wav.pow(2).mean().sqrt()),
                   "env_smooth": 0.0 if os.environ.get("SNV_ESM") == "0" else float(getattr(gen, "env_smooth", 0.0))}
    (fs / "shapes.json").write_text(json.dumps(sh, indent=1))
    return sh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=MEMBERS)
    ap.add_argument("--gate", default="snv_p45_gate_t80")
    ap.add_argument("--front", default="")
    ap.add_argument("--voc-export", default="results/rvoc3hi/export_330k")
    ap.add_argument("--fb", default="anch")
    ap.add_argument("--out", default=str(ROOT / "results/snv_k3g_export"))
    a = ap.parse_args()
    out = Path(a.out)
    assert out.resolve() not in {(ROOT / "results" / n).resolve() for n in KEEP}, f"{out} は製品か既存の書き出し"
    front = Path(a.front) if a.front else ROOT / "results" / f"{a.gate}_export"
    gman = json.loads((front / "manifest.json").read_text())
    assert gman["cfg"]["kind"] == "gate" and gman["cfg"]["gate"] == f"results/{a.gate}/last.pt", "門の書き出し(前段)が --gate と違う"
    fx = out / "fixture"
    fx.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    names = a.members.split(",")
    gen = EV.load_gen()
    sys = EV.System("ens:" + "|".join(f"{n}+mask:{a.gate}+fb:{a.fb}" for n in names), gen)
    fb_parts = [q for q in a.fb.split(",") if q]
    assert isinstance(sys, EV.Ens) and all(p.p3 and p.fb == fb_parts and not (p.tf or p.aw or p.cs or p.hp) for p in sys.parts)
    G = sys.parts[0].sep
    assert G.kind == "gate" and G.dec == "myin" and G.hyst and torch.load(ROOT / f"results/{a.gate}/last.pt", map_location="cpu", weights_only=False).get("step") == gman["cfg"]["step_gate"]
    ten, _ = XG.front_consts()
    mem = []
    for i, (n, p) in enumerate(zip(names, sys.parts)):
        st = torch.load(ROOT / "results" / n / "last.pt", map_location="cpu", weights_only=True)
        assert st.get("play") == "wst" and st.get("circ", "x") == "x" and not st.get("xc0") and list(p.Pl.lags) == [0], f"{n}: PlayW + Circ・遅れ [0] の型だけ"
        ten |= {f"m{i}.Pl.{k}": v.float().cpu() for k, v in p.Pl.state_dict().items() if not k.startswith("cpc.W.")} | {f"m{i}.C.{k}": v.float().cpu() for k, v in p.C.state_dict().items()}
        mem.append({"prefix": f"m{i}", "name": n, "ckpt": f"results/{n}/last.pt", "step": st.get("step"), "envc": bool(p.envc), "cpc": st["cpc"], "pca": st["pca"], "cb": st["cb"],
                    "pca_k": int(p.Pl.P.shape[1]), "lags": list(p.Pl.lags), "shrink": float(p.Pl.shrink), "cb_s2": float(p.Pl.cb_s2)})
    W = int(WIN_MS * 1e-3 * P.SR)
    has_c0bg = len(fb_parts) == 2 and fb_parts[0].startswith("c0bg") and fb_parts[1] == "anch"
    assert fb_parts == ["anch"] or has_c0bg, f"未対応の fb: {a.fb}"
    cfg = {"kind": "k3g", "half": bool(os.environ.get("SNV_WF16")), "kv16": bool(os.environ.get("SNV_KV16")), "target_v4": EV.FEAT.name.endswith("_v4"), "members": mem, "gate": f"results/{a.gate}/last.pt",
           "fb": "c0bg+anch" if has_c0bg else "anch", "D_Y": T.D_Y, "D_A": T.D_A, "ch": 256, "zdim": 256, "heads": 4, "dk": 64, "dv": 16, "dils": [1, 2, 4, 8, 16, 32] * 2,
           "attn_after": 3, "K_SEMI": K_SEMI, "cpc_dils": [1, 2, 4, 8, 16, 32], "tau": float(CM.TAU), "ln_eps": 1e-5,
           "fixed_env": {"win_ms": WIN_MS, "win": W, "nfft": T.PA.NFFT, "hop": P.H, "floor": 1e-5, "window": "hann_sym"},
           "anchor": {"alphas": [float(v) for v in AN.ALPHAS], "fit_hz": AN.FIT_HZ, "mel_hz": [float(v) for v in AN.mel_hz()], "c0_sil": float(T.PA.C0_SIL)}}
    if has_c0bg:
        if fb_parts[0].startswith("c0bgr"):
            pts = fb_parts[0][5:].split("_")
            lam_v = float(pts[0])
            th_lo = float(pts[1]) if len(pts) > 1 and pts[1] else 0.10
            th_hi = float(pts[2]) if len(pts) > 2 and pts[2] else 0.20
            cfg["c0bg"] = {"lam": lam_v, "thr": 0.5 * (th_lo + th_hi), "thr_lo": th_lo, "thr_hi": th_hi, "soft": True,
                           "fixed_env": {"win_ms": 20.0, "win": int(0.020 * P.SR), "nfft": T.PA.NFFT, "hop": P.H, "floor": 1e-5, "window": "hann_sym"}}
        else:
            lam_s, _, thr_s = fb_parts[0][4:].partition("_")
            th_v = float(thr_s or "0.15")
            cfg["c0bg"] = {"lam": float(lam_s), "thr": th_v, "thr_lo": th_v, "thr_hi": th_v, "soft": False,
                           "fixed_env": {"win_ms": 20.0, "win": int(0.020 * P.SR), "nfft": T.PA.NFFT, "hop": P.H, "floor": 1e-5, "window": "hann_sym"}}
    man = {"tensors": {}, "cfg": cfg}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].contiguous().numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tg = meta["targets"][0]
    pf = ROOT / "data/snv_p_feat"
    clean, _ = X30.feat(F.load48(pf / "real/uirou_0.wav"), pf / "real/uirou_0.npz")
    op_x, _ = X30.feat(F.load48(pf / "real/uirou_enroll.wav"), pf / "real/uirou_enroll.npz")
    alt_x, _ = X30.feat(F.load48(pf / "real/namikawa_0.wav"), pf / "real/namikawa_0.npz")
    if cfg["target_v4"]:
        import snv_p125_feat as F125
        rf = []
        for j in (0, 1, 2):
            rx = F.load48(ROOT / f"results/snv_a0/wav/{tg}_{j}.wav")
            rx = rx[: P.n_frames(len(rx)) * P.H]
            d4 = F125.analyze_v4(rx)
            lf4 = np.where(d4["f0"] > 0, np.log(np.maximum(d4["f0"], 1.0) / 200.0), 0.0).astype(np.float32)
            rf.append((rx, {"y": T.to_y(d4["env"], d4["per"], d4["f0"]), "lf": lf4, "v": (d4["f0"] > 0).astype(np.float32), "f0": d4["f0"]}))
    else:
        rf = [X30.feat(F.load48(ROOT / f"results/snv_a0/wav/{tg}_{j}.wav"), pf / f"a0/{tg}_{j}.npz") for j in (0, 1, 2)]
    refs = [EV.utt(f"a0/{tg}_{j}") for j in (0, 1, 2)]
    for (rx, r2), u in zip(rf, refs):
        n = len(u["v"])
        assert np.array_equal(r2["y"], u["y"]) and np.array_equal(T.to_y(P.fixed_env(rx, n, WIN_MS), np.zeros((4, n), np.float32), np.ones(n, np.float32))[:25], u["yc"][:25]) and np.array_equal(u["yc"][25:], u["y"][25:]), "目標の参照の波形と保存の特徴(包絡・固定の窓の包絡)が違う"
    music, mixed, _, _ = XG.pick_music(G, clean)
    stat = [{"wav": op_x}]
    S = sys.structure(f"t|{tg}", refs)
    src = {"wav": mixed}
    yo_ens = sys.convert(src, stat, S, K_SEMI, with_y=True)[1]
    ms = [member_stage(p, src, stat, refs, s, K_SEMI) for p, s in zip(sys.parts, S)]
    m0 = ms[0]
    g = m0["u"]["gate"] > 0.5
    for m in ms[1:]:
        assert all(np.array_equal(m["u"][q], m0["u"][q]) for q in ("y", "v", "lf", "f0", "gate")) and np.array_equal(m["uo"][0]["y"], m0["uo"][0]["y"]), "組ごとの前段の出力が違う"
        assert torch.equal(m["yw"], m0["yw"]) and m["alpha"] == m0["alpha"] and torch.equal(m["ry"], m0["ry"]), "組ごとの錨が違う"
    assert [bool(p.envc) for p in sys.parts].count(True) >= 1 and all(("yc" in m["u"]) == bool(p.envc) for p, m in zip(sys.parts, ms))
    assert torch.equal(sum(m["yf"][:, :29] for m in ms) / len(ms), yo_ens), "組の出力の平均が Ens.convert と違う"
    avg_raw = sum(m["yo"][:, :29] for m in ms) / len(ms)
    if has_c0bg:
        avg_c0bg = sys.parts[0].fallback(fb_parts[0], m0["u"], m0["y"], m0["v"], m0["a"], avg_raw, stat, S[0])
        once = torch.where(torch.from_numpy(g)[None, None], m0["yw"], avg_c0bg)
    else:
        once = torch.where(torch.from_numpy(g)[None, None], m0["yw"], avg_raw)
    d_once = float((once - yo_ens).abs().max())
    assert d_once <= 1e-6, f"平均の後に錨を 1 回の形が Ens と違う({d_once})"
    me = next(m for p, m in zip(sys.parts, ms) if p.envc)
    Tn = int(m0["y"].shape[-1])
    fenv = P.fixed_env(mixed, Tn, WIN_MS)
    assert np.array_equal(T.to_y(fenv, np.zeros((4, Tn), np.float32), np.ones(Tn, np.float32))[:25], me["u"]["yc"][:25]) and np.array_equal(me["u"]["yc"][25:], m0["u"]["y"][25:])
    oyc = np.concatenate([q["yc"] for q in me["uo"]], 1)
    w(fx, "x", mixed), w(fx, "fenv", fenv.T)
    w(fx, "src_y", m0["y"][0].T), w(fx, "src_yc", me["yk"][0].T), w(fx, "src_v", m0["v"][0]), w(fx, "src_lf_k", m0["lf"][0]), w(fx, "src_gate", g.astype(np.float32))
    if has_c0bg:
        w(fx, "src_c0_20", m0["u"]["c0_20"]), w(fx, "src_gate_p", m0["u"]["gate_p"])
        w(fx, "op_c0_20", np.concatenate([q["c0_20"] for q in m0["uo"]]))
    w(fx, "op_y", np.concatenate([q["y"] for q in m0["uo"]], 1).T), w(fx, "op_yc", oyc.T), w(fx, "op_v", np.concatenate([q["v"] for q in m0["uo"]]))
    w(fx, "ref_y", m0["ry"][0].T), w(fx, "ref_yc", me["ryc"][0].T), w(fx, "ref_v", m0["rv"][0])
    w(fx, "mu_o", m0["mu_o"]), w(fx, "mu_t", m0["mu_t"]), w(fx, "yw", m0["yw"][0].T), w(fx, "yo", yo_ens[0].T)
    for i, (m, s) in enumerate(zip(ms, S)):
        w(fx, f"m{i}_op_mu", m["st"][0][0]), w(fx, f"m{i}_op_wm", m["st"][1][0]), w(fx, f"m{i}_op_delta", m["st"][2][0])
        w(fx, f"m{i}_a_ref", m["a_ref"][0].T), w(fx, f"m{i}_sC_z", s["z"][0]), w(fx, f"m{i}_sC_k", s["k"][0]), w(fx, f"m{i}_sC_v", s["v"][0])
        w(fx, f"m{i}_c", m["c"][0].T), w(fx, f"m{i}_a", m["a"][0].T), w(fx, f"m{i}_yo", m["yo"][0].T)
    tr = XG.transitions(g)
    mon_k = g | ((m0["u"]["gate_p"] >= float(G.thr)) if getattr(G, "inst_mask", False) else False)
    fmask_k = int(np.nonzero(mon_k)[0][0]) if mon_k.any() else int(tr[0][0])
    (fx / "shapes.json").write_text(json.dumps({"T": Tn, "Tr_op": int(oyc.shape[1]), "Tr_t": int(m0["ry"].shape[-1]), "k": K_SEMI, "members": len(ms), "envc": [bool(p.envc) for p in sys.parts], "alpha": m0["alpha"],
                                                "music": Path(music).name, "mus_s": list(XG.MUS), "snr_db": XG.SNR, "transitions": tr, "first_mask": fmask_k, "gate_open": float(g.mean()), "anchor_once_vs_ens_max_abs": d_once}, indent=1))
    waves = {"op": op_x, "alt": alt_x, "refs": [rx for rx, _ in rf]}
    shapes = {"k": K_SEMI, "refs": len(rf), "members": names, "gate": f"results/{a.gate}/last.pt", "front": str(front.resolve().relative_to(ROOT.resolve())), "model": str(out.resolve().relative_to(ROOT.resolve())),
              "voc": a.voc_export, "alpha": m0["alpha"], "op_gate_open": float(m0["uo"][0]["gate_open"])}
    sc = stream_fixture(out / "fixture_stream", sys, gen, clean, stat, S, waves, shapes)
    sm = stream_fixture(out / "fixture_stream_music", sys, gen, mixed, stat, S, waves, shapes | {"music": Path(music).name, "mus_s": list(XG.MUS), "snr_db": XG.SNR})
    assert sc["gate_open"] == 0.0 and not sc["transitions"], "音楽なしの源で門が開いた"
    assert sm["transitions"] == tr and any(q[1] for q in tr), "音楽つきの源で門が開かない"
    print("exported", len(ten), "tensors", off, "floats ->", out, "| 組", [(q["name"], q["envc"]) for q in mem], flush=True)
    print("模型の段: T", Tn, "Tr_op", oyc.shape[1], "Tr_t", int(m0["ry"].shape[-1]), "・α", m0["alpha"], "・門の遷移", tr, "・開いた割合", round(float(g.mean()), 4), "・平均の後に錨を 1 回 対 Ens の最大差", d_once, flush=True)
    print("端から端: 音楽なし", {q: sc[q] for q in ("T", "gate_open", "rms")}, "/ 音楽つき", {q: sm[q] for q in ("T", "gate_open", "rms", "transitions")}, "/ 操作者の登録の門", shapes["op_gate_open"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
