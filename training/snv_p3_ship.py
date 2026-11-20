"""P3 の出荷ゲート(AGENTS.md・synth_nam_vc §8.2): 変換の推論の経路 全体(P0a の前段 → 凍結した CPC の演奏 → C → rvoc3hi)の未来不変性を実音声のプローブで測り、静的な遅延の台帳と演算量を書く。
経路 = 操作者の音 x → 高さ・包絡・周期性(因果)→ CPC の文脈 c − 操作者の登録の c の平均 → 線形 → C(目標の参照の構造・高さ + k 半音)→ Y → 描画器の界面 → 48 kHz の出力の塊。
登録の平均と目標の構造は登録の定数。CPC は --cpc の重み・C と線形は未学習(因果性は配線で決まる)か --ckpt。全て CPU・決定的。出力 = results/snv_p3/ship.json。
--e_lift K(paper_snv §5 M19): --enc s の E の入力の包絡の c_K〜c24 を定数に置き換える(フレームごと・定数 = --ckpt の lift_mean か 0)。
--sep <ckpt>|random(paper_snv §5 M20): 前段 v3 の代わりに前段 D(train_snv_sep・左寄せの窓の短時間スペクトル → 因果の畳み込み)の出力で E・C・描画器を駆動する(操作者の登録も D を通す)。random = 未学習の D(配線の因果性の検査)。
--mask <ckpt>|random(paper_snv §5 M23): 前段 v3 に時間周波数の重み(train_snv_mask)を掛けた特徴で駆動する。random = 未学習の重みの網(初期の重み ≈ 0.95・配線の因果性の検査)。random_gate(paper_snv §5 M27)= 未学習の門(出力の偏り 0 = 門が入力で開閉する)+ 段 I21 の重み・--dec myin で重みで濾した YIN も通す。random_det(paper_snv §5 M24)= 未学習の判定の網(出力の偏り 0 = 判定が入力で入れ替わる・2 値で外す経路の因果性の検査)。
--fb c0s,anch(paper_snv §5 M48・段 P67): 出力の c0 を、登録の統計で標準化した源の c0 を目標の参照の平均と標準偏差に戻したものにする(登録の定数だけ)。
--fb c0j<τ>,anch(paper_snv §5 M50・段 P77): c0s を、回路の c0 との差が目標の標準偏差の τ 倍以下のフレームだけに当てる(それより上は回路の c0 のまま・登録の定数だけ)。
--fb c0g<θ>,anch(paper_snv §5 M50・段 P77 step a″): c0s を、門の確率(持続の条件と閾の前の値・因果)が θ 未満のフレームだけに当てる。
--fb swapall<β>,anch(paper_snv §5 M43・段 P58): 全てのフレームで C(a, 目標) + β(y − C(a, 操作者))、門が開いたフレームで一律の伸縮(α と操作者の構造は登録の定数)を通す。
--ens の組の重みに envc の印が在るとき(段 P91・P93)、その組の符号は、操作者の波形から作った固定の窓の包絡(snv_p0_front.fixed_env・左寄せ)を読む。目標の構造の鍵も参照の波形の固定の窓の包絡で作る。
--ens <cpc>;<pca>;<cb>;<ckpt>[,…](paper_snv §5 M49・段 P75): 符号と回路の組を足し、回路の出力(錨の差し替えの後)を全ての組で平均して描く(--fb は c0s・anch だけ)。
--aw <α>(paper_snv 段 P76): 符号に渡す入力の包絡(源と操作者の登録)の周波数軸を固定の α 倍にする(錨の経路は加工しない包絡のまま)。
--hyst on,off(paper_snv rev39 段 I24): 門に開閉の持続(フレーム数)を付けて通す(因果の状態機械の検査)。
--mask random_gate:<重みの保存>(段 P47): 未学習の門(入力で開閉する)と、指定した重み(網の幅を替えた重みなど)で、重みの経路の未来不変性を確かめる。
    cd training && uv run --no-sync python snv_p3_ship.py --cpc results/snv_p2b_cpc/last.pt [--ckpt results/<tag>/last.pt] [--out results/<dir>] [--circ q] [--play wst --pca results/snv_p8b/pca_c_P2b.npy] [--enc w --pca results/snv_p8b/pca_c_P2b.npy(paper_snv §5 M28 の E = 白くした演奏 → 軌跡)] [--enc s(P11 の E = 操作者の回路の逆)| p12(P12 の当てはめる回路・未学習の配線では θ は乱数)]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import nvoc as N
import rvoc as R
import snv_p0_front as P
import train_rvoc as TR
import train_snv_p1 as T
import train_snv_p2_cpc as CP
import train_snv_p3 as P3
from ship_check import Lookahead, future_invariance, ledger, probes

ROOT = Path(__file__).resolve().parent.parent
CK_R = ROOT / "results/rvoc3hi/snap/ema_330k.pt"
OUT = ROOT / "results/snv_p3"
SEMI = 7
DEV = "cpu"


def block_macs(ch: int, n: int, k: int = 3) -> int:
    return n * (ch * 2 * ch * k + ch * ch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--cpc", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--circ", choices=["x", "q"], default="x")
    ap.add_argument("--play", choices=["c", "wst"], default="c")
    ap.add_argument("--enc", choices=["cpc", "s", "p12", "w"], default="cpc")
    ap.add_argument("--e_lift", type=int, default=0)
    ap.add_argument("--pca", default="")
    ap.add_argument("--lags", default="")
    ap.add_argument("--shrink", type=float, default=None)
    ap.add_argument("--cb", default="")
    ap.add_argument("--xc0", action="store_true")
    ap.add_argument("--sep", default="")
    ap.add_argument("--mask", default="")
    ap.add_argument("--dec", default="")
    ap.add_argument("--hyst", default="")
    ap.add_argument("--fb", default="")
    ap.add_argument("--ens", default="")
    ap.add_argument("--aw", type=float, default=0.0)
    a = ap.parse_args()
    out = ROOT / a.out if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    Pl, C = P3.make_play(ROOT / a.cpc, a.play, ROOT / a.pca if a.pca else None, [int(x) for x in a.lags.split(",")] if a.lags else None, a.shrink, ROOT / a.cb if a.cb else None, a.xc0).eval(), (T.CircQ() if a.circ == "q" else T.Circ()).eval()
    if a.enc == "s":
        import train_snv_p11 as P11
        Es = P11.EncS().eval()
        lmean = torch.zeros(25)

        class PlS(torch.nn.Module):
            def mean_of(self, y: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> dict:
                return Es.structure(P11.xin(P11.lift(y, a.e_lift, lmean), v), m)

            def forward(self, y: torch.Tensor, v: torch.Tensor, st: dict) -> torch.Tensor:
                return Es(P11.xin(P11.lift(y, a.e_lift, lmean), v), st)[0]

        Pl = PlS()
    if a.enc == "w":
        import snv_encw as EW
        import train_snv_p11 as P11
        Es = EW.EncW(ROOT / a.cpc, ROOT / a.pca).eval()

        class PlW(torch.nn.Module):
            def mean_of(self, y: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> dict:
                return Es.structure(P11.xin(y, v), m)

            def forward(self, y: torch.Tensor, v: torch.Tensor, st: dict) -> torch.Tensor:
                return Es(P11.xin(y, v), st)[0]

        Pl = PlW()
    if a.enc == "p12":
        import train_snv_p12 as P12
        M12 = P12.Circuit().eval()
        th_op, th_t = torch.randn(1, P12.D_TH), torch.randn(1, P12.D_TH)

        class PlF(torch.nn.Module):
            def mean_of(self, y: torch.Tensor, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                return th_t if y.shape[-1] == ry.shape[-1] else th_op

            def forward(self, y: torch.Tensor, v: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
                return M12.enc(y, v, th)[0]

        class CF(torch.nn.Module):
            def structure(self, ry_: torch.Tensor, ra: torch.Tensor, rm_: torch.Tensor) -> torch.Tensor:
                return th_t

            def forward(self, a_: torch.Tensor, lf_: torch.Tensor, v_: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
                return M12.dec(a_, lf_, v_, th)

        Pl, C = PlF(), CF()
    if a.ckpt:
        st = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=True)
        C.load_state_dict(st["C"])
        if a.enc == "w":
            Es.load_state_dict(st["Es"])
        elif a.enc == "s":
            Es.load_state_dict(st["Es"])
            if st.get("lift_mean"):
                lmean.copy_(torch.tensor(st["lift_mean"]))
        else:
            Pl.proj.load_state_dict(st["proj"])
    extra = []
    for spec in [q for q in a.ens.split(",") if q]:
        cpc_, pca_, cb_, ck_ = spec.split(";")
        Pk = P3.make_play(ROOT / cpc_, a.play, ROOT / pca_ if pca_ else None, [int(x) for x in a.lags.split(",")] if a.lags else None, a.shrink, ROOT / cb_ if cb_ else None, a.xc0).eval()
        Ck = (T.CircQ() if a.circ == "q" else T.Circ()).eval()
        envk = False
        if ck_:
            sk = torch.load(ROOT / ck_, map_location="cpu", weights_only=True)
            Ck.load_state_dict(sk["C"])
            Pk.proj.load_state_dict(sk["proj"])
            envk = bool(sk.get("envc"))
        extra.append((Pk, Ck, envk))
    assert not extra or (a.enc == "cpc" and all(k in ("c0s", "anch") or k.startswith(("c0j", "c0g")) for k in a.fb.split(",") if k)), "--ens は --enc cpc と --fb c0s / c0j<τ> / c0g<θ>,anch だけ"
    D = None
    if a.sep:
        import train_snv_sep as SEP
        D = SEP.Sep().eval() if a.sep == "random" else SEP.load(ROOT / a.sep, DEV)
    if a.mask:
        import train_snv_mask as SEP
        if a.mask.startswith("random_gate"):
            D = SEP.Gated(SEP.MaskNet(b0=0.0, nb=1).eval(), SEP.load(ROOT / (a.mask.partition(":")[2] or "results/snv_i21_mask_nf/last.pt"), DEV)).eval()
        else:
            D = SEP.MaskNet().eval() if a.mask == "random" else SEP.MaskNet(b0=0.0).eval() if a.mask == "random_det" else SEP.load(ROOT / a.mask, DEV)
        if a.mask == "random_det":
            D.kind = "det"
        if a.dec:
            D.dec = a.dec
        if a.hyst:
            D.hyst = [int(v) for v in a.hyst.split(",")]

    last: dict = {}

    def front(x: np.ndarray) -> tuple:
        if D is None:
            f = P.analyze(x)
            y_, f0_ = T.to_y(f["env"], f["per"], f["f0"]), f["f0"]
        else:
            u = SEP.features(D, x, DEV)
            y_, f0_ = u["y"], u["f0"]
            last["gate"], last["gate_p"] = u.get("gate"), u.get("gate_p")
        v_ = (f0_ > 0).astype(np.float32)
        lf_ = np.where(f0_ > 0, np.log(np.maximum(f0_, 1.0) / 200.0), 0.0).astype(np.float32)
        return torch.from_numpy(y_)[None], torch.from_numpy(v_)[None], torch.from_numpy(lf_)[None], f0_

    s_r = torch.load(CK_R, map_location="cpu", weights_only=False)
    cfg = s_r["cfg"]
    gen = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).eval().to(DEV)
    gen.load_state_dict(s_r["ema"])
    pl = [p.float() for p in probes(2 * P.SR, sr=P.SR, male=2)]
    ref = pl[0].numpy()
    rf = P.analyze(ref)
    ry = torch.from_numpy(T.to_y(rf["env"], rf["per"], rf["f0"]))[None]
    rv = torch.from_numpy((rf["f0"] > 0).astype(np.float32))[None]
    rm = torch.ones(1, ry.shape[-1], dtype=torch.bool)
    op = pl[-1].numpy()
    oy, ov, _, _ = front(op)

    def wy(y_: torch.Tensor) -> torch.Tensor:
        if not a.aw:
            return y_
        import snv_p0c_anchor as AW
        yn = y_[0].numpy()
        ew = AW.warp_env(np.concatenate([yn[:1] * 10 + T.PA.C0_SIL, yn[1:25] * 10], 0), a.aw)
        return torch.from_numpy(np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, yn[25:]], 0).astype(np.float32))[None]

    oy_raw, oy = oy, wy(oy)

    def fx(x_: np.ndarray, y_: torch.Tensor) -> torch.Tensor:
        n = y_.shape[-1]
        e = T.to_y(P.fixed_env(x_, n), np.zeros((4, n), np.float32), np.ones(n, np.float32))[:25]
        return torch.from_numpy(np.concatenate([e, y_[0].numpy()[25:]], 0).astype(np.float32))[None]

    ry_f, oy_f = (fx(ref, ry), fx(op, oy_raw)) if any(e for _, _, e in extra) else (ry, oy)
    with torch.no_grad():
        mu_t = Pl.mean_of(ry, rv, rm)
        mu_op = Pl.mean_of(oy, ov, torch.ones(1, oy.shape[-1], dtype=torch.bool))
        S = C.structure(ry, Pl(ry, rv, mu_t), rm)
        S_op = C.structure(oy, Pl(oy, ov, mu_op), torch.ones(1, oy.shape[-1], dtype=torch.bool)) if a.fb else None
        ens = [(Pk, Ck, Pk.mean_of(oy_f if ek else oy, ov, torch.ones(1, oy.shape[-1], dtype=torch.bool)), Ck.structure(ry, Pk(ry_f if ek else ry, rv, Pk.mean_of(ry_f if ek else ry, rv, rm)), rm), ek) for Pk, Ck, ek in extra]
    if a.fb:
        import snv_p0c_anchor as AN
        venv = lambda y_, v_: (T.y_env(y_)[0] * v_).sum(-1).numpy() / max(1.0, float(v_.sum()))
        e_t, e_o = venv(ry, rv), venv(oy_raw, ov)
        fb_alpha = AN.fit_alpha(e_o, e_t)
        c0st = lambda y_, v_: (float(y_[0, 0][v_[0] > 0.5].mean()), max(float(y_[0, 0][v_[0] > 0.5].std(unbiased=False)), 1e-3))
        c0_t, c0_o = c0st(ry, rv), c0st(oy_raw, ov)

    def fallback(y: torch.Tensor, v: torch.Tensor, lf: torch.Tensor, pl_: torch.Tensor, yo: torch.Tensor) -> torch.Tensor:
        g = torch.from_numpy(last["gate"])[None, None] if last.get("gate") is not None else torch.zeros_like(v)[:, None]
        for kind in a.fb.split(","):
            if kind.startswith("swapall"):
                yo = torch.cat([yo[:, :25] + float(kind[7:]) * (y[:, :25] - C(pl_, lf, v, S_op)[:, :25]), yo[:, 25:]], 1)
            elif kind == "c0s" or kind.startswith(("c0j", "c0g")):
                d = c0_t[0] + c0_t[1] * (y[:, :1] - c0_o[0]) / c0_o[1]
                gp = torch.from_numpy(last["gate_p"])[None, None] if last.get("gate_p") is not None else torch.zeros_like(v)[:, None]
                ok = True if kind == "c0s" else gp < float(kind[3:]) if kind.startswith("c0g") else (d - yo[:, :1]) / c0_t[1] <= float(kind[3:])
                yo = torch.cat([d if kind == "c0s" else torch.where(ok, d, yo[:, :1]), yo[:, 1:]], 1)
            else:
                assert kind == "anch", kind
                yn = y[0].numpy()
                ew = AN.warp_env(np.concatenate([yn[:1] * 10 + T.PA.C0_SIL, yn[1:25] * 10], 0), fb_alpha)
                yw = torch.from_numpy(np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, yn[25:29]], 0).astype(np.float32))[None]
                yo = torch.where(g > 0.5, yw, yo[:, :29])
        return yo

    def convert(x: torch.Tensor) -> torch.Tensor:
        y, v, lf, f0 = front(x.numpy())
        with torch.no_grad():
            yc = wy(y)
            pl_ = Pl(yc, v, mu_op)
            yo = C(pl_, lf + SEMI * T.LN2_12 * v, v, S)
            if a.fb:
                yo = fallback(y, v, lf, pl_, yo)
            yf = fx(x.numpy(), y) if any(e[4] for e in ens) else yc
            for Pk, Ck, mk, Sk, ek in ens:
                yk = Ck(Pk(yf if ek else yc, v, mk), lf + SEMI * T.LN2_12 * v, v, Sk)
                yo = yo[:, :29] + (fallback(y, v, lf, pl_, yk) if a.fb else yk)[:, :29]
            yo = yo[:, :29] / (1 + len(ens))
            cond = torch.cat([yo[:, :25], (lf + SEMI * T.LN2_12 * v)[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
            f0k = torch.from_numpy(f0 * 2 ** (SEMI / 12))[None]
            wav = gen(cond.to(DEV), TR.excitation(f0k.to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu()
        Tn = cond.shape[-1]
        return wav[: Tn * N.HOP].reshape(Tn, N.HOP).T

    t0 = time.time()
    la = future_invariance(convert, P.H, sr=P.SR, probe_list=pl[1:], n_edit=4)
    t_fi = time.time() - t0
    print("future_invariance (front" + (" D" if D is not None else "") + " → CPC → C → rvoc3hi):", la, flush=True)
    rows = [("hop (block)", 1000.0 * P.H / P.SR), ("front" + (" D" if D is not None else "") + " → CPC → C → rvoc3hi lookahead", la), ("renderer DELAY", 1000.0 * N.DELAY / P.SR)]
    ok_ledger = ledger(rows, budget_ms=30.0, reserve_ms=10.0)
    fps = P.SR / P.H
    tr_frames = int(ry.shape[-1])
    macs = {"CPC+proj": (CP.D_IN * 256 * 3 + 256 * 256 * 3 + 256 * 256 * 2 + block_macs(256, 6) + 256 * T.D_A) * fps,
            "C": (block_macs(256, 12) + (T.D_A + 2) * 256 + 256 * T.D_Y + 256 * 256 + 4 * 64 * tr_frames + 4 * 16 * tr_frames + 64 * 256) * fps,
            "rvoc3hi": float(R.macs_per_second(gen))}
    macs["CPC+proj"], macs["C"] = macs["CPC+proj"] * (1 + len(ens)), macs["C"] * (1 + len(ens))
    if D is not None:
        import train_snv_sep as SP_

        def net_macs(net) -> float:
            ch, d_out = net.inp.out_channels, net.head.out_channels
            return (SP_.D_IN * ch + len(net.blocks) * (ch * ch * 3 + ch * ch) + ch * d_out + (SP_.NFFT // 2 + 1) * (SP_.N_LIN + SP_.N_MEL) + (SP_.NFFT2 // 2 + 1) * SP_.N_MEL2
                    + 5 * SP_.NFFT * math.log2(SP_.NFFT) + 5 * SP_.NFFT2 * math.log2(SP_.NFFT2) + (3 * d_out if a.mask else 0)) * fps
        if getattr(D, "kind", None) == "gate":
            macs["gate"], macs["mask"] = net_macs(D.gate), net_macs(D.mask)
            if getattr(D, "dec", None) == "myin":
                macs["myin"] = 2 * 5 * 4096 * math.log2(4096) * fps
        else:
            macs["mask" if a.mask else "D"] = net_macs(D)
    x10 = torch.cat(pl)[: 10 * P.SR]
    t1 = time.time()
    y, v, lf, _ = front(x10.numpy())
    t_front = time.time() - t1
    torch.set_num_threads(1)
    t2 = time.time()
    with torch.no_grad():
        C(Pl(y, v, mu_op), lf, v, S)
    t_ec = time.time() - t2
    res = {"future_invariance": {"samples": la.samples, "ms": la.ms, "inconclusive": la.inconclusive, "lower_only": la.lower_only, "unbounded": la.unbounded, "probes": len(pl) - 1, "sec": round(t_fi, 1)},
           "ledger_rows_ms": {k: (v_.ms if isinstance(v_, Lookahead) else v_) for k, v_ in rows}, "ledger_pass": bool(ok_ledger), "budget_ms": 30.0, "reserve_ms": 10.0,
           "macs_per_s": {k: (None if v_ is None else round(v_ / 1e9, 3)) for k, v_ in macs.items()}, "ref_frames": tr_frames,
           "offline_cpu_10s_audio_sec": {"front_python_1proc": round(t_front, 2), "CPC_C_torch_1thread": round(t_ec, 2)}, "cpc": a.cpc, "ens": a.ens, "members": 1 + len(ens), "aw": a.aw, "circ": a.circ, "play": a.play, "lags": a.lags, "shrink": a.shrink, "cb": a.cb, "xc0": a.xc0, "enc": a.enc, "sep": a.sep, "mask": a.mask, "fb": a.fb,
           "sep_params": None if D is None else int(sum(p_.numel() for p_ in D.parameters())),
           "note": "未学習の重み(--ckpt なし)は配線の因果性の検査。RTF の判定は Rust の流しで(本表は Python のオフラインの目安)"}
    (out / "ship.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    passed = la.samples == 0 and not la.inconclusive and not la.unbounded and bool(ok_ledger)
    print("SHIP", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
