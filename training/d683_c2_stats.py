"""D683: C2(連続の内容の特徴・results/c2_1/last.pt = Rust の --c2 と同じ網)の次元ごとの統計。本物の女声(female-dataset の D678 の real_train の先頭 100 本)と操作者ごとの参照(<組>/d56/_work/op_PPPP.wav)。
有音のフレーム(240 標本ごとのエネルギーが最大 −40 dB より上)。出力 = <H>/d683_c2/{female_mean.f32(64)・<組>_op_PPPP_norm.f32(256 = μ_op・σ_op・μ_f・σ_f)}。
    uv run python d683_c2_stats.py --sets d487_uiro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = H / "d683_c2"


def frames(x: np.ndarray) -> np.ndarray:
    a = torch.from_numpy(TC.prime(x.astype(np.float32)))[None].cuda()
    T = len(x) // 240
    with torch.no_grad():
        f = P.cvin_frames(a, T)[0].float().cpu().numpy().T
    n = min(T, len(f))
    e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
    return f[:n][e > e.max() - 40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P.cvin_setup(str(ROOT / "results/c2_1/last.pt"), "cuda")
    L = json.loads((ROOT / "results/conv_p0/d678/lists.json").read_text())
    fem = np.concatenate([frames(load48(Path(r["path"]))[: 10 * 48000]) for r in L["real_train"] if r["src"] == "pool"][:100])
    mu_f, sd_f = fem.mean(0), fem.std(0)
    mu_f.astype("<f4").tofile(OUT / "female_mean.f32")
    print("female frames", len(fem), "mean |mu|", float(np.abs(mu_f).mean()), "mean sd", float(sd_f.mean()), flush=True)
    for s in a.sets:
        for f in sorted((H / s / "d56/_work").glob("op_*.wav")):
            o = frames(load48(f))
            mu, sd = o.mean(0), o.std(0)
            np.concatenate([mu, sd, mu_f, sd_f]).astype("<f4").tofile(OUT / f"{s}_{f.stem}_norm.f32")
        print(s, "op mean shift |mu_op - mu_f| / sd_f", float(np.mean(np.abs(mu - mu_f) / sd_f)), "sd ratio", float(np.mean(sd / sd_f)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
