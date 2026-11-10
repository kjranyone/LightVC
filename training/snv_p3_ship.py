"""P3 の出荷ゲート(AGENTS.md・synth_nam_vc §8.2): 変換の推論の経路 全体(P0a の前段 → 凍結した CPC の演奏 → C → rvoc3hi)の未来不変性を実音声のプローブで測り、静的な遅延の台帳と演算量を書く。
経路 = 操作者の音 x → 高さ・包絡・周期性(因果)→ CPC の文脈 c − 操作者の登録の c の平均 → 線形 → C(目標の参照の構造・高さ + k 半音)→ Y → 描画器の界面 → 48 kHz の出力の塊。
登録の平均と目標の構造は登録の定数。CPC は --cpc の重み・C と線形は未学習(因果性は配線で決まる)か --ckpt。全て CPU・決定的。出力 = results/snv_p3/ship.json。
--e_lift K(paper_snv §5 M19): --enc s の E の入力の包絡の c_K〜c24 を定数に置き換える(フレームごと・定数 = --ckpt の lift_mean か 0)。
--sep <ckpt>|random(paper_snv §5 M20): 前段 v3 の代わりに前段 D(train_snv_sep・左寄せの窓の短時間スペクトル → 因果の畳み込み)の出力で E・C・描画器を駆動する(操作者の登録も D を通す)。random = 未学習の D(配線の因果性の検査)。
--mask <ckpt>|random(paper_snv §5 M23): 前段 v3 に時間周波数の重み(train_snv_mask)を掛けた特徴で駆動する。random = 未学習の重みの網(初期の重み ≈ 0.95・配線の因果性の検査)。random_gate(paper_snv §5 M27)= 未学習の門(出力の偏り 0 = 門が入力で開閉する)+ 段 I21 の重み・--dec myin で重みで濾した YIN も通す。random_det(paper_snv §5 M24)= 未学習の判定の網(出力の偏り 0 = 判定が入力で入れ替わる・2 値で外す経路の因果性の検査)。
    cd training && uv run --no-sync python snv_p3_ship.py --cpc results/snv_p2b_cpc/last.pt [--ckpt results/<tag>/last.pt] [--out results/<dir>] [--circ q] [--play wst --pca results/snv_p8b/pca_c_P2b.npy] [--enc s(P11 の E = 操作者の回路の逆)| p12(P12 の当てはめる回路・未学習の配線では θ は乱数)]
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
    ap.add_argument("--enc", choices=["cpc", "s", "p12"], default="cpc")
    ap.add_argument("--e_lift", type=int, default=0)
    ap.add_argument("--pca", default="")
    ap.add_argument("--sep", default="")
    ap.add_argument("--mask", default="")
    ap.add_argument("--dec", default="")
    a = ap.parse_args()
    out = ROOT / a.out if a.out else OUT
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    Pl, C = P3.make_play(ROOT / a.cpc, a.play, ROOT / a.pca if a.pca else None).eval(), (T.CircQ() if a.circ == "q" else T.Circ()).eval()
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
        if a.enc == "s":
            Es.load_state_dict(st["Es"])
            if st.get("lift_mean"):
                lmean.copy_(torch.tensor(st["lift_mean"]))
        else:
            Pl.proj.load_state_dict(st["proj"])
    D = None
    if a.sep:
        import train_snv_sep as SEP
        D = SEP.Sep().eval() if a.sep == "random" else SEP.load(ROOT / a.sep, DEV)
    if a.mask:
        import train_snv_mask as SEP
        if a.mask == "random_gate":
            D = SEP.Gated(SEP.MaskNet(b0=0.0, nb=1).eval(), SEP.load(ROOT / "results/snv_i21_mask_nf/last.pt", DEV)).eval()
        else:
            D = SEP.MaskNet().eval() if a.mask == "random" else SEP.MaskNet(b0=0.0).eval() if a.mask == "random_det" else SEP.load(ROOT / a.mask, DEV)
        if a.mask == "random_det":
            D.kind = "det"
        if a.dec:
            D.dec = a.dec

    def front(x: np.ndarray) -> tuple:
        if D is None:
            f = P.analyze(x)
            y_, f0_ = T.to_y(f["env"], f["per"], f["f0"]), f["f0"]
        else:
            u = SEP.features(D, x, DEV)
            y_, f0_ = u["y"], u["f0"]
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
    with torch.no_grad():
        mu_t = Pl.mean_of(ry, rv, rm)
        mu_op = Pl.mean_of(oy, ov, torch.ones(1, oy.shape[-1], dtype=torch.bool))
        S = C.structure(ry, Pl(ry, rv, mu_t), rm)

    def convert(x: torch.Tensor) -> torch.Tensor:
        y, v, lf, f0 = front(x.numpy())
        with torch.no_grad():
            yo = C(Pl(y, v, mu_op), lf + SEMI * T.LN2_12 * v, v, S)
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
    if D is not None:
        import train_snv_sep as SP_

        def net_macs(net) -> float:
            ch, d_out = net.inp.out_channels, net.head.out_channels
            return (SP_.D_IN * ch + len(SP_.DIL) * (ch * ch * 3 + ch * ch) + ch * d_out + (SP_.NFFT // 2 + 1) * (SP_.N_LIN + SP_.N_MEL) + (SP_.NFFT2 // 2 + 1) * SP_.N_MEL2
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
           "offline_cpu_10s_audio_sec": {"front_python_1proc": round(t_front, 2), "CPC_C_torch_1thread": round(t_ec, 2)}, "cpc": a.cpc, "circ": a.circ, "play": a.play, "enc": a.enc, "sep": a.sep, "mask": a.mask,
           "sep_params": None if D is None else int(sum(p_.numel() for p_ in D.parameters())),
           "note": "未学習の重み(--ckpt なし)は配線の因果性の検査。RTF の判定は Rust の流しで(本表は Python のオフラインの目安)"}
    (out / "ship.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    passed = la.samples == 0 and not la.inconclusive and not la.unbounded and bool(ok_ledger)
    print("SHIP", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
