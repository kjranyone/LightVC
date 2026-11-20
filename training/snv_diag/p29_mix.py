"""段 P29 step a の a3e(results/snv_p29/prereg.yaml・学習なし・評価の天井だけ): 声を跨ぐ変換の出力のどの成分を、目標の声の包絡の並びに替えると余弦が戻るか。
組 = p29_same.py と同じ 242 組・k 0。y* = 目標の声が同じ文を読んだ発話を、p29_oracle.py と同じ DTW で源の時間に写したもの。C の出力 yo の成分を y* に 1 つずつ替えて、源の高さと有声の並びで描く。
差し替え: none / all / c0 / lo = c1〜c12 / hi = c13〜c24 / per / lo_mean / lo_dyn / lo_gv / lo_slow / lo_fast(遅い = 有声のフレームの上のガウスの平滑 σ = 8 フレーム)。
時刻を揃えた対はこの測りにだけ使う(学習・目標・損失に使わない)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p29_mix.py [sys=snv_p24_c] [out=results/snv_p29/mix.json] [mix=...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import p29_oracle as O
import p29_same as S
import probe_rig as G
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV
SPAN = {"none": [], "all": [(0, 29)], "c0": [(0, 1)], "lo": [(1, 13)], "hi": [(13, 25)], "per": [(25, 29)]}


def slow(x: np.ndarray, m: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    w = gaussian_filter1d(m.astype(np.float64), sigma, mode="constant")
    return (gaussian_filter1d(x * m[None], sigma, axis=1, mode="constant") / np.maximum(w, 1e-6)[None]).astype(np.float32)


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag = opt.get("sys", "snv_p24_c")
    mixes = opt.get("mix", "none,all,c0,lo,hi,per,lo_mean,lo_dyn,lo_gv,lo_slow,lo_fast").split(",")
    out = ROOT / opt.get("out", "results/snv_p29/mix.json")
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(trim(y16)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    gen = EV.load_gen()

    @torch.no_grad()
    def render(y: np.ndarray, f0: np.ndarray) -> np.ndarray:
        v = (f0 > 0).astype(np.float32)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0).astype(np.float32))[None].to(DEV)
        ex = TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        return gen(cond, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    P = S.pairs()
    voices = sorted({p["u"] for p in P} | {p["v"] for p in P})
    enr = {v: [T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")] for v in voices}
    du = {(v, s): T.load_utt(T.FEAT / f"d/{v}/{s}.npz") for p in P for v, s in ((p["u"], p["su"]), (p["v"], p["sv"]))}
    Et = {v: emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")).astype(np.float32) for t in ("t00", "t01", "t02")])) for v in sorted({p["v"] for p in P})}
    s_ = EV.System(tag, gen)
    rows = {m: [] for m in mixes}
    for n, p in enumerate(P):
        s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
        j = O.align(s, t)
        tv = np.flatnonzero(t["v"] > 0)
        for i in np.flatnonzero((s["v"] > 0) & (t["v"][j] == 0)):
            k = tv[np.argmin(np.abs(tv - j[i]))]
            if abs(int(k) - int(j[i])) <= 8:
                j[i] = k
        yo = s_.convert(s, enr[p["u"]], s_.structure(f"t|{p['v']}", enr[p["v"]]), 0, with_y=True)[1][0].cpu().numpy()
        L = yo.shape[1]
        ys, vm, f0 = t["y"][:, j][:, :L], s["v"][:L] > 0, s["f0"][:L]
        a, b = yo[1:13], ys[1:13]
        ma, mb = a[:, vm].mean(1, keepdims=True), b[:, vm].mean(1, keepdims=True)
        sa, sb = slow(a, vm), slow(b, vm)
        for m in mixes:
            ym = yo.copy()
            for lo, hi in SPAN.get(m, []):
                ym[lo:hi] = ys[lo:hi]
            if m.startswith("lo_"):
                new = {"lo_mean": a - ma + mb, "lo_dyn": b - mb + ma, "lo_gv": (a - ma) * (b[:, vm].std(1, keepdims=True) / np.maximum(a[:, vm].std(1, keepdims=True), 1e-6)) + ma, "lo_slow": a - sa + sb, "lo_fast": b - sb + sa}[m]
                ym[1:13, vm] = new[:, vm]
            rows[m].append({"u": p["u"], "v": p["v"], "ct": round(float(emb(w16(render(ym, f0))) @ Et[p["v"]]), 4)})
        if n % 40 == 0:
            print(n, {m: rows[m][-1]["ct"] for m in mixes}, flush=True)
    res = {"prereg": "results/snv_p29/prereg.yaml(a3e)", "sys": tag, "summary": {}, "rows": rows}
    for sx in ("ef", "em"):
        I = [i for i, r in enumerate(rows[mixes[0]]) if r["u"].startswith(sx)]
        us = sorted({rows[mixes[0]][i]["u"] for i in I})
        ct = {m: float(np.mean([rows[m][i]["ct"] for i in I])) for m in mixes}
        o = {"n": len(I), "ct": {m: round(x, 4) for m, x in ct.items()}}
        if "none" in ct and "all" in ct:
            g = np.random.default_rng(0)
            per = {m: (np.array([np.mean([rows[m][i]["ct"] - rows["none"][i]["ct"] for i in I if rows[m][i]["u"] == u]) for u in us]), np.array([np.mean([rows["all"][i]["ct"] - rows["none"][i]["ct"] for i in I if rows[m][i]["u"] == u]) for u in us])) for m in mixes}
            idx = [g.integers(0, len(us), len(us)) for _ in range(2000)]
            o["戻る割合"] = {m: [round((ct[m] - ct["none"]) / (ct["all"] - ct["none"]), 3), [round(float(np.percentile([x[i].sum() / y[i].sum() for i in idx], q)), 3) for q in (2.5, 97.5)]] for m, (x, y) in per.items()}
        res["summary"][sx] = o
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
