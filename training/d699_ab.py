"""D699(paper_mf E-B1c・規則は results/conv_p0/d699/prereg.yaml): 出力の高さの罰 A2' を表の声区と生成器の低い f0 の描き方に分ける。生成器 N65。
--streams: H/d699/dump/<組>(LIGHTVC_COND_ONLY の書き出し)の事後の argmax で腕 65 の c0 の表を引き(train_ppgvoc.c0tab_np)→ H/d699/stream/<組>/{pi}_{ti}.f32([c0, NaN])。
--eval: v1(gen fold)の reg = LX − LL(同じ入力の対)・tot = LX − devff2 k0(N65)・low = LL − devff2 k0(N65)。
    uv run python d699_ab.py --streams
    uv run python d699_ab.py --eval --out ../results/conv_p0/d699/d699.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_ppgvoc import c0tab_np

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
DC, Q0, Q1 = 295, 31, 231
SETS = ("d693_lx", "d693_ll")


def pairs(s: str) -> list:
    return json.loads((H / s / "devset.json").read_text())["pairs"]


def streams() -> int:
    tab = np.fromfile(ROOT / "results/diag_ppgvoc65/c0_tab.f32", dtype="<f4")
    rep = {}
    for s in SETS:
        (H / "d699" / "stream" / s).mkdir(parents=True, exist_ok=True)
        n = []
        for r in pairs(s):
            x = np.fromfile(H / "d699" / "dump" / s / f"{r['pi']}_{r['ti']}.f32", dtype="<f4").reshape(-1, DC)
            c = c0tab_np(x[:, Q0:Q1].argmax(1), tab)
            np.stack([c, np.full_like(c, np.nan)], 1).astype("<f4").tofile(H / "d699" / "stream" / s / f"{r['pi']}_{r['ti']}.f32")
            n.append(len(x))
        rep[s] = {"n": len(n), "frames_min": int(min(n)), "frames_max": int(max(n))}
    print(rep, flush=True)
    return 0


def boot(d: np.ndarray, g: np.ndarray, rng: np.random.Generator) -> list:
    us = sorted(set(g))
    idx = {u: np.flatnonzero(g == u) for u in us}
    bs = [np.median(d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def evaluate(out: str) -> int:
    from d692_eval import f0_med
    from kansei_v1 import KanseiV1
    km = KanseiV1("gen")
    rng = np.random.default_rng(699)
    w = lambda s, k, d: H / s / "gate" / d / f"{k}_{d}.wav"
    lx = pairs("d693_lx")
    ll = {(r["pi"], r["j"]): r for r in pairs("d693_ll")}
    kx = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in lx]
    kl = [f"p{r['pi']:04d}_t{ll[(r['pi'], r['j'])]['ti']:03d}" for r in lx]
    kf = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in pairs("devff2")]
    v = {"LX": np.array(km.margin([w("d693_lx", k, "RUST0N65") for k in kx])), "LL": np.array(km.margin([w("d693_ll", k, "RUST0N65") for k in kl])),
         "LX_Y58": np.array(km.margin([w("d693_lx", k, "RUST0Y58") for k in kx])), "LL_Y58": np.array(km.margin([w("d693_ll", k, "RUST0Y58") for k in kl])),
         "devff2_k0": np.array(km.margin([w("devff2", k, "RUST0N65") for k in kf])), "devff2_km8": np.array(km.margin([w("devff2", k, "RUST-8N65") for k in kf]))}
    rep: dict = {"fold": "gen", "generator": "N65", "n_pairs": len(lx)}
    for nm, x in v.items():
        rep[nm] = {"median": round(float(np.median(x)), 3), "in_band": round(float((x <= 0).mean()), 3), "n": len(x)}
    for nm, s, ks in (("LX", "d693_lx", kx), ("LL", "d693_ll", kl)):
        rep[nm]["f0_out_median"] = round(float(np.nanmedian([f0_med(w(s, k, "RUST0N65")) for k in ks])), 1)
    en = lambda s, ti: float(np.exp(json.loads((H / s / "rr_en" / f"en_{ti:03d}" / "enroll.json").read_text())["mu"]))
    rep["f0_T_median"] = {"LX": round(float(np.median([en("d693_lx", r["ti"]) for r in lx])), 1),
                          "LL": round(float(np.median([en("d693_ll", ll[(r["pi"], r["j"])]["ti"]) for r in lx])), 1)}
    g = np.array([r["pi"] for r in lx])
    d = v["LX"] - v["LL"]
    reg = float(np.median(d))
    f0 = float(np.median(v["devff2_k0"]))
    tot = float(np.median(v["LX"])) - f0
    low = float(np.median(v["LL"])) - f0
    rep["reg"] = {"paired_median": round(reg, 3), "ci95_by_input": boot(d, g, rng), "frac_LX_worse": round(float((d > 0).mean()), 3)}
    rep["tot"] = round(tot, 3)
    rep["low"] = round(low, 3)
    rep["share_reg"] = round(reg / tot, 3) if tot > 0 else None
    rep["share_low"] = round(low / tot, 3) if tot > 0 else None
    rep["Y58_reg_paired_median"] = round(float(np.median(v["LX_Y58"] - v["LL_Y58"])), 3)
    vt = "no_penalty" if tot <= 0.5 else "table_main" if reg >= 0.6 * tot else "render_main" if low >= 0.6 * tot else "mixed"
    rep["verdict_rule"] = vt
    print(json.dumps({k: rep[k] for k in ("LX", "LL", "devff2_k0", "devff2_km8", "reg", "tot", "low", "share_reg", "share_low", "Y58_reg_paired_median", "f0_T_median", "verdict_rule")},
                     ensure_ascii=False), flush=True)
    Path(out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.streams:
        return streams()
    return evaluate(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
