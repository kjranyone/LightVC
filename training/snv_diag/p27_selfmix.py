"""段 P27 補足 2(results/snv_p27/prereg.yaml の supplement_2・学習なし): 自己の reamp の出力のどの成分が本人らしさを落とすか。
A0F の 24 声 × j 3〜5 を源に、同じ声の j 0〜2 を登録と構造にして k 0 で変換し、C の出力 yo の成分を 1 つずつ本当の y に差し替えて描く(高さと有声は元のまま)。
差し替え: none / c0 / lo = c1〜c12 / hi = c13〜c24 / per = 周期性の 4 帯 / all(= 写しの描画)。値 = 目標への余弦 ct と M(ECAPA + LDA の 24 声の 1 位)・時間方向の標準偏差の比(出力 / 本当)。
補足 3: hi_mean = c13〜c24 の発話の平均だけ本当に合わせる / hi_gv = 平均からのずれを次元ごとに(本当の標準偏差 / 出力の標準偏差)倍 / hi_gv164 = ずれを一律 1.64 倍(有声のフレームの統計・無声はそのまま)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p27_selfmix.py [sys=snv_p24_c] [out=results/snv_p27/selfmix.json] [mix=none,c0,lo,hi,per,all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import probe_rig as G
import snv_p1_eval as EV
import train_rvoc as TR

ROOT = G.ROOT
DEV = EV.DEV
SPAN = {"none": [], "c0": [(0, 1)], "lo": [(1, 13)], "hi": [(13, 25)], "per": [(25, 29)], "all": [(0, 29)], "lo+hi": [(1, 25)], "c0+lo": [(0, 13)], "lo+per": [(1, 13), (25, 29)]}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tag, mixes = opt.get("sys", "snv_p24_c"), opt.get("mix", "none,c0,lo,hi,per,all").split(",")
    out = ROOT / opt.get("out", "results/snv_p27/selfmix.json")
    from snv_a0_measure import load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(trim(y16)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    lda = lambda e: proj(e, (Tz["mu"], Tz["scalings"]))
    gen = EV.load_gen()

    @torch.no_grad()
    def render(y: np.ndarray, u: dict) -> np.ndarray:
        v = u["v"]
        cond = torch.from_numpy(np.concatenate([y[:25], u["lf"][None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0).astype(np.float32))[None].to(DEV)
        ex = TR.excitation(torch.from_numpy(u["f0"].astype(np.float32))[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        return gen(cond, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    all_t, _ = G.targets()
    Et = {v: emb(np.concatenate([to16(*load(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    El = {v: lda(Et[v]) for v in all_t}
    s = EV.System(tag, gen)
    rows = {m: [] for m in mixes}
    sd = {k: [] for k in ("c0", "lo", "hi", "per")}
    err = {k: [] for k in ("c0", "lo", "hi", "per")}
    for v in all_t:
        refs = [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)]
        S = s.structure(f"t|{v}", refs)
        for j in (3, 4, 5):
            u = EV.utt(f"a0/{v}_{j}")
            _, yo = s.convert(u, refs, S, 0, with_y=True)
            yo = yo[0].cpu().numpy()
            y = u["y"][:, :yo.shape[1]]
            vm = u["v"][:yo.shape[1]] > 0
            for k, (lo, hi) in (("c0", (0, 1)), ("lo", (1, 13)), ("hi", (13, 25)), ("per", (25, 29))):
                sd[k].append(float(np.mean(yo[lo:hi, vm].std(1) / np.maximum(y[lo:hi, vm].std(1), 1e-6))))
                err[k].append(float(np.mean(np.abs(yo[lo:hi, vm] - y[lo:hi, vm]))))
            for m in mixes:
                ym = yo.copy()
                for lo, hi in SPAN.get(m, []):
                    ym[lo:hi] = y[lo:hi]
                if m.startswith("hi_"):
                    mo, mt = yo[13:25, vm].mean(1, keepdims=True), y[13:25, vm].mean(1, keepdims=True)
                    g = {"hi_mean": 1.0, "hi_gv": y[13:25, vm].std(1, keepdims=True) / np.maximum(yo[13:25, vm].std(1, keepdims=True), 1e-6), "hi_gv164": 1.64}[m]
                    ym[13:25, vm] = (yo[13:25, vm] - mo) * g + (mt if m == "hi_mean" else mo)
                e = emb(w16(render(ym, {q: u[q][:yo.shape[1]] for q in ("v", "lf", "f0")})))
                el = lda(e)
                top = max(all_t, key=lambda t: float(el @ El[t]))
                rows[m].append({"spk": v, "j": j, "m": int(top == v), "top": top, "ct": round(float(e @ Et[v]), 4)})
    res = {"prereg": "results/snv_p27/prereg.yaml(supplement_2)", "sys": tag, "summary": {m: {"n": len(R), "ct": round(float(np.mean([r["ct"] for r in R])), 4), "M": round(float(np.mean([r["m"] for r in R])), 3)} for m, R in rows.items()},
           "std_ratio(出力 / 本当・有声)": {k: round(float(np.mean(x)), 3) for k, x in sd.items()}, "l1(有声)": {k: round(float(np.mean(x)), 4) for k, x in err.items()}, "rows": rows}
    if "none" in rows and "all" in rows:
        a, b = res["summary"]["none"]["ct"], res["summary"]["all"]["ct"]
        res["recovery(不足のうち戻る割合)"] = {m: round((res["summary"][m]["ct"] - a) / (b - a), 3) for m in mixes}
    print(json.dumps({k: x for k, x in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
