"""P 線の前段 v3(snv_p0_front.py)を Rust(crates/lightvc-core/src/p_front.rs)へ移すための定数と一致検査の固定の入出力(paper_snv §6 段 P)。
出力 <out>/: weights.bin・manifest.json(mel.fb [128][1025]・dct [25][128]・const = 前段の定数と Butterworth の b・a)・
fixture/<名>/(x.f32 = 48 kHz・f0.f32・ap.f32 [T]・env.f32 [T][25]・per.f32 [T][4] = snv_p0_front.analyze)。
    cd training && uv run --no-sync python export_pfront.py --out ../results/pfront_export
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter

import pae as PA
import snv_p0_front as P
import snv_p_feat as F

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = {"uirou_0": ROOT / "data/snv_p_feat/real/uirou_0.wav", "ef000_t03": ROOT / "data/snv_e1/ef000/t03.wav"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fb, dct = PA._np_consts()
    b, aa = butter(P.LP_ORDER, P.LP_HZ / (P.SR / 2))
    ten = {"mel.fb": torch.from_numpy(fb).float(), "dct": torch.from_numpy(dct[:25]).float()}
    const = {k: getattr(P, k) for k in ("SR", "H", "W_YIN", "LP_HZ", "LP_ORDER", "K_CAND", "C_MAX", "BETA", "LAMBDA", "GAMMA", "U_COST", "HOLD_BASE", "HOLD_STEP",
                                        "LB_HZ", "LB_MIN", "TMIN", "TMAX", "E_FLOOR", "NFFT_P", "NFFT_X", "NFFT_S", "ETA", "PAD_L")}
    const |= {"butter_b": b.tolist(), "butter_a": aa.tolist(), "NFFT": PA.NFFT, "HB": [list(x) for x in PA.HB], "C0_SIL": PA.C0_SIL}
    if P.WIN_CAP_MS < 1000.0:
        const["ENV_CAP_MS"] = P.WIN_CAP_MS
    if P.TAU_UV_HZ != 200.0:
        const["TAU_UV_HZ"] = P.TAU_UV_HZ
    f0v4 = os.environ.get("SNV_F0V4") == "1"
    if f0v4:
        const["F0_MODE"] = "v4_dual"
    assert P.WIN_K == 3.0 and not P.CT_MODE, "窓の倍率と窓の形の診断のスイッチは移植に含めていない"
    assert const["ETA"] == 0.0, "倍音の検査(ETA > 0)は移植に含めていない"
    man = {"tensors": {}, "const": const}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    for name, src in FIXTURES.items():
        x = F.load48(src)
        T = P.n_frames(len(x))
        x = x[: T * P.H]
        d = out / "fixture" / name
        d.mkdir(parents=True, exist_ok=True)
        x.astype("<f4").tofile(d / "x.f32")
        if f0v4:
            import snv_f0v4 as V4
            fn, fv, bn = V4.f0_dual_track(torch.from_numpy(x))
            f_ep = torch.where((fv <= 0) & (fn > 0), fn, fv)
            env = P.envelope(x, f_ep.numpy().astype(np.float64))
            per = P.periodicity(torch.from_numpy(x), f_ep).numpy()
            fn.numpy().astype("<f4").tofile(d / "f0.f32")
            fv.numpy().astype("<f4").tofile(d / "f0_voc.f32")
            bn.numpy().astype("<f4").tofile(d / "ap.f32")
            env.T.astype("<f4").tofile(d / "env.f32")
            per.T.astype("<f4").tofile(d / "per.f32")
            print(name, "frames", T, "voiced_net", float((fn.numpy() > 0).mean()), "voiced_voc", float((fv.numpy() > 0).mean()), flush=True)
        else:
            r = P.analyze(x)
            r["f0"].astype("<f4").tofile(d / "f0.f32")
            r["ap"].astype("<f4").tofile(d / "ap.f32")
            r["env"].T.astype("<f4").tofile(d / "env.f32")
            r["per"].T.astype("<f4").tofile(d / "per.f32")
            print(name, "frames", T, "voiced", float((r["f0"] > 0).mean()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
