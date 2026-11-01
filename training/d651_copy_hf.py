"""D651: 実女声の録音の写し合成で、表現(CheapTrick の包絡・DCT 24)と描画(WORLD・入力の大きさの正規化・非周期性)の高域の損失を測る。
録音 = 開発の組の目標の参照(d37/_work/ref_XXX.wav・60 人)の発声の多い 8 s。腕: W0 = CheapTrick + D4C の WORLD の往復・W24 = PA.envelope の DCT 1〜24 → env_from_rows(フレームの CheapTrick の平均パワーで正規化)+ D4C・
W24P = W24 の非周期性を 4 帯の周期性(PA.periodicity の HNR → ap_from_per)から。hf / tilt は d109 と同じ式(発声の区間 = 最大から 40 dB 以内のフレーム・8 s)。規則は results/conv_p0/d651/prereg.yaml。
    uv run python d651_copy_hf.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import pyworld as pw
import scipy.signal as ss
import torch

import nvoc as N
import pae as PA
from d631_world_swap import H, ROOT, SETS, keys
from d634_artic_render import FQ, env_from_rows
from d635_split import norm
from d639_knob_hf import ap_from_per

OUT = ROOT / "results/conv_p0/d651"
CF = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
CF = CF[(CF >= 100) & (CF <= 10000)]


def hf_tilt(x: np.ndarray) -> tuple:
    x = x[: 8 * N.SR].astype(np.float64)
    x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
    n = len(x) // N.HOP
    e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
    act = e > e.max() - 40
    fq, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
    P = (np.abs(Z[:, : len(act)]) ** 2)[:, act[: Z.shape[1]]]
    hf = float(10 * np.log10(P[(fq >= 4000) & (fq < 10000)].sum() / P[(fq >= 100) & (fq < 4000)].sum()))
    Pm = P.mean(1)
    L = np.array([Pm[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in CF])
    L = 10 * np.log10(L + 1e-20)
    sel = (CF >= 300) & (CF <= 8000)
    return hf, float(np.polyfit(np.log2(CF[sel]), L[sel], 1)[0])


def best8(x: np.ndarray) -> np.ndarray:
    w = 8 * N.SR
    if len(x) <= w:
        return x
    e = np.convolve(x ** 2, np.ones(N.SR), "valid")[:: N.SR]
    i = int(np.argmax([e[j:j + 8].sum() for j in range(max(1, len(e) - 8))]))
    return x[i * N.SR:i * N.SR + w]


def job(it: tuple) -> dict:
    s, ti = it
    from d6_unit_cov import load48
    torch.set_num_threads(1)
    x = best8(load48(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav").astype(np.float64))
    f0, t = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=60.0, f0_ceil=1000.0, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, N.SR)
    ap = pw.d4c(x, f0, t, N.SR)
    T = len(f0)
    xa = np.concatenate([np.zeros(PA.A), x, np.zeros(PA.A + N.HOP * 2)])
    env = PA.envelope(xa, f0)
    per = PA.periodicity(torch.from_numpy(xa)[None].float(), torch.from_numpy(f0)[None].float())[0].numpy().T.astype(np.float64)
    rows = env[1:25].T.astype(np.float64) / 10.0
    pin = sp.mean(1, keepdims=True)
    e24 = norm(env_from_rows(rows), pin)
    out = {"key": f"{s}|{ti}"}
    for a, y in (("ORIG", x), ("W0", pw.synthesize(f0, sp, ap, N.SR, 5.0)), ("W24", pw.synthesize(f0, e24, ap, N.SR, 5.0)),
                 ("W24P", pw.synthesize(f0, e24, ap_from_per(per[:T], f0), N.SR, 5.0))):
        out[a] = hf_tilt(np.asarray(y))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    its = [(s, ti) for s in SETS for ti in sorted({int(k[7:10]) for k in keys(s)})]
    with Pool(int(os.environ.get("LVC_NPROC", "12"))) as pl:
        recs = pl.map(job, its)
    rep: dict = {"n": len(recs)}
    for a in ("ORIG", "W0", "W24", "W24P"):
        hf = np.array([r[a][0] for r in recs])
        tl = np.array([r[a][1] for r in recs])
        rep[a] = {"hf_median": round(float(np.median(hf)), 2), "tilt_median": round(float(np.median(tl)), 3)}
        if a != "ORIG":
            dh = hf - np.array([r["ORIG"][0] for r in recs])
            dt = tl - np.array([r["ORIG"][1] for r in recs])
            rep[a] |= {"d_hf_median": round(float(np.median(dh)), 2), "d_hf_p10_p90": [round(float(np.percentile(dh, 10)), 2), round(float(np.percentile(dh, 90)), 2)],
                       "d_tilt_median": round(float(np.median(dt)), 3)}
    rep["items"] = recs
    (OUT / "copy.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "items"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
