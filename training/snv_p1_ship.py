"""P1 の出荷ゲート(AGENTS.md・synth_nam_vc §8.2): 変換の推論の経路 全体(P0a の前段 → E → C → rvoc3hi)の未来不変性を実音声のプローブで測り、静的な遅延の台帳と演算量を書く。
経路 = 操作者の音 x → 高さ・包絡・周期性(因果)→ E の入力(包絡 − 操作者の登録の平均)→ E → C(目標の参照の構造・高さ + k 半音)→ Y → 描画器の界面 → 48 kHz の出力の塊。
登録の平均と目標の構造は登録の定数(推論の前に一度)。重みは未学習(因果性は配線で決まる)か --ckpt。全て CPU・決定的(CUDA の畳み込みの非決定性は 1 bit の比較で偽の先読みを出す = 2026-11-04 に 1025 ms の偽 FAIL)。出力 = results/snv_p1/ship.json。
    cd training && uv run --no-sync python snv_p1_ship.py [--ckpt results/<tag>/last.pt]
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
from ship_check import Lookahead, future_invariance, ledger, probes

ROOT = Path(__file__).resolve().parent.parent
CK_R = ROOT / "results/rvoc3hi/snap/ema_330k.pt"
OUT = ROOT / "results/snv_p1"
SEMI = 7
DEV = "cpu"


def block_macs(ch: int, n: int, k: int = 3) -> int:
    return n * (ch * 2 * ch * k + ch * ch)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    E, C = T.Enc().eval(), T.Circ().eval()
    if a.ckpt:
        st = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=True)
        E.load_state_dict(st["E"]), C.load_state_dict(st["C"])
    s_r = torch.load(CK_R, map_location="cpu", weights_only=False)
    cfg = s_r["cfg"]
    gen = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).eval().to(DEV)
    gen.load_state_dict(s_r["ema"])
    pl = [p.float() for p in probes(2 * P.SR, sr=P.SR, male=2)]
    ref = pl[0].numpy()
    rf = P.analyze(ref)
    ry = torch.from_numpy(T.to_y(rf["env"], rf["per"], rf["f0"]))[None]
    rv = torch.from_numpy((rf["f0"] > 0).astype(np.float32))[None]
    s_, n_ = T.voiced_mean(rf["env"], rf["f0"])
    mu_t = torch.from_numpy(s_ / max(n_, 1))[None]
    op = pl[-1].numpy()
    of = P.analyze(op)
    s_, n_ = T.voiced_mean(of["env"], of["f0"])
    mu_op = torch.from_numpy(s_ / max(n_, 1))[None]
    with torch.no_grad():
        S = T.structure_of(E, C, ry, rv, torch.ones(1, ry.shape[-1], dtype=torch.bool), mu_t)

    def convert(x: torch.Tensor) -> torch.Tensor:
        f = P.analyze(x.numpy())
        y = torch.from_numpy(T.to_y(f["env"], f["per"], f["f0"]))[None]
        v = torch.from_numpy((f["f0"] > 0).astype(np.float32))[None]
        lf = torch.from_numpy(np.where(f["f0"] > 0, np.log(np.maximum(f["f0"], 1.0) / 200.0), 0.0).astype(np.float32))[None]
        with torch.no_grad():
            yo = C(E(T.e_in(y, v, mu_op)), lf + SEMI * T.LN2_12 * v, v, S)
            cond = torch.cat([yo[:, :25], (lf + SEMI * T.LN2_12 * v)[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
            f0k = torch.from_numpy(f["f0"] * 2 ** (SEMI / 12))[None]
            wav = gen(cond.to(DEV), TR.excitation(f0k.to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu()
        Tn = cond.shape[-1]
        return wav[: Tn * N.HOP].reshape(Tn, N.HOP).T

    t0 = time.time()
    la = future_invariance(convert, P.H, sr=P.SR, probe_list=pl[1:], n_edit=4)
    t_fi = time.time() - t0
    print("future_invariance (front → E → C → rvoc3hi):", la, flush=True)
    rows = [("hop (block)", 1000.0 * P.H / P.SR), ("front → E → C → rvoc3hi lookahead", la), ("renderer DELAY", 1000.0 * N.DELAY / P.SR)]
    ok_ledger = ledger(rows, budget_ms=30.0, reserve_ms=10.0)
    fps = P.SR / P.H
    tr_frames = int(ry.shape[-1])
    macs = {"E": (block_macs(256, 6) + T.D_X * 256 + 256 * T.D_A) * fps,
            "C": (block_macs(256, 12) + (T.D_A + 2) * 256 + 256 * T.D_Y + 256 * 256 + 4 * 64 * tr_frames + 4 * 16 * tr_frames + 64 * 256) * fps,
            "rvoc3hi": float(R.macs_per_second(gen))}
    x10 = torch.cat(pl)[: 10 * P.SR]
    t1 = time.time()
    f = P.analyze(x10.numpy())
    t_front = time.time() - t1
    y = torch.from_numpy(T.to_y(f["env"], f["per"], f["f0"]))[None]
    v = torch.from_numpy((f["f0"] > 0).astype(np.float32))[None]
    lf = torch.from_numpy(np.where(f["f0"] > 0, np.log(np.maximum(f["f0"], 1.0) / 200.0), 0.0).astype(np.float32))[None]
    torch.set_num_threads(1)
    t2 = time.time()
    with torch.no_grad():
        C(E(T.e_in(y, v, mu_op)), lf, v, S)
    t_ec = time.time() - t2
    res = {"future_invariance": {"samples": la.samples, "ms": la.ms, "inconclusive": la.inconclusive, "lower_only": la.lower_only, "unbounded": la.unbounded, "probes": len(pl) - 1, "sec": round(t_fi, 1)},
           "ledger_rows_ms": {k: (v_.ms if isinstance(v_, Lookahead) else v_) for k, v_ in rows}, "ledger_pass": bool(ok_ledger), "budget_ms": 30.0, "reserve_ms": 10.0,
           "macs_per_s": {k: (None if v_ is None else round(v_ / 1e9, 3)) for k, v_ in macs.items()}, "ref_frames": tr_frames,
           "offline_cpu_10s_audio_sec": {"front_python_1proc": round(t_front, 2), "E_C_torch_1thread": round(t_ec, 2)},
           "note": "未学習の重み(--ckpt なし)は配線の因果性の検査。RTF の判定は Rust の流しで(本表は Python のオフラインの目安)"}
    (OUT / "ship.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    passed = la.samples == 0 and not la.inconclusive and not la.unbounded and bool(ok_ledger)
    print("SHIP", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
