"""段 P78 step a(results/snv_p78/prereg.yaml・学習なし・ASR なし): 変換の出力を符号で読み直したとき、源の読みとどれだけ食い違うか。
系・組・試行は段 P74 step a(p74_conv.py)と同じ。CER は既に在る行(results/snv_p74/conv_b.json・conv_d.json)を当てる。
読み = PlayW.feat。R = 源の有声のフレームでの、源の読みと出力の読みの余弦の平均(wav = 出力の音を前段で読み直す・env = 回路の出力の包絡をそのまま渡す・oth = もう 1 つの系の符号)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p78_reread.py [sys=snv_p30_c] [oth=snv_p24_c] [sets=cv,x:d_clean,x:d_rv6,x:d_rv6b] [out=results/snv_p78/reread.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as I13
import p18_align as AL
import snv_p1_eval as EV

ROOT = EV.ROOT
DEV = EV.DEV
SHIFT = 20


def feat(sy: EV.System, u: dict, st: tuple, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.from_numpy(u["v"])[None].to(DEV)
    yy = torch.from_numpy(u["y"])[None].to(DEV) if y is None else y
    with torch.no_grad():
        return sy.Pl.feat(yy, v, st)[0], v[0] > 0.5


def cosm(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor, lag: int = 0) -> float:
    n = min(a.shape[-1], b.shape[-1]) - abs(lag)
    if n <= 0:
        return 0.0
    aa, bb, mm = (a[:, :n], b[:, lag:lag + n], m[:n]) if lag >= 0 else (a[:, -lag:-lag + n], b[:, :n], m[-lag:-lag + n])
    c = torch.nn.functional.cosine_similarity(aa, bb, dim=0)
    return float((c * mm).sum() / mm.sum().clamp(min=1.0))


def reread(sy: EV.System, so: EV.System, src: dict, enr: list[dict], w: np.ndarray, yo: torch.Tensor | None, st_out: tuple, st_out_o: tuple) -> dict:
    a, va = feat(sy, src, sy.op_stat(enr))
    u2 = sy.prep(I13.utt_of(np.asarray(w, np.float64)))
    b, vb = feat(sy, u2, st_out)
    n = min(len(va), len(vb))
    m = (va[:n] & vb[:n]).float()
    r = [cosm(a, b, m, lag) for lag in range(-4, 5)]
    out = {"R_wav": max(r), "lag": int(np.argmax(r)) - 4, "floor": cosm(a, b, m, SHIFT), "vboth": float(m.sum() / va.float().sum().clamp(min=1.0))}
    if yo is not None:
        e, _ = feat(sy, src, st_out, yo[:, : src["y"].shape[0]])
        out["R_env"] = cosm(a, e, va.float())
    ao, vo = feat(so, so.prep(src), so.op_stat(enr))
    bo, vbo = feat(so, so.prep(I13.utt_of(np.asarray(w, np.float64))), st_out_o)
    n = min(len(vo), len(vbo))
    mo = (vo[:n] & vbo[:n]).float()
    out["R_oth"] = max(cosm(ao, bo, mo, lag) for lag in range(-4, 5))
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])) if len(pos) and len(neg) else float("nan")


def rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def assoc(R: np.ndarray, C: np.ndarray, spk: np.ndarray) -> dict:
    g = np.random.default_rng(0)
    us = sorted(set(spk))
    f = lambda ix: (auc(-R[ix][C[ix] > 0.5], -R[ix][C[ix] <= 0.5]), float(np.corrcoef(rank(R[ix]), rank(C[ix]))[0, 1]))
    bs = np.array([f(np.concatenate([np.flatnonzero(spk == us[j]) for j in g.integers(0, len(us), len(us))])) for _ in range(2000)])
    a, s = f(np.arange(len(R)))
    return {"n": int(len(R)), "CER > 0.5": int((C > 0.5).sum()), "AUC": round(a, 3), "AUC の区間": [round(float(np.nanpercentile(bs[:, 0], q)), 3) for q in (2.5, 97.5)],
            "順位相関": round(s, 3), "順位相関の区間": [round(float(np.nanpercentile(bs[:, 1], q)), 3) for q in (2.5, 97.5)]}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tag, otag = opt.get("sys", "snv_p30_c"), opt.get("oth", "snv_p24_c")
    names = opt.get("sets", "cv,x:d_clean,x:d_rv6,x:d_rv6b").split(",")
    out = ROOT / opt.get("out", "results/snv_p78/reread.json")
    old = [r for p in ("conv_b", "conv_d") for r in json.loads((ROOT / f"results/snv_p74/{p}.json").read_text())["rows"] if r["sys"] == tag]
    cer = {(r["set"], r["src"]): r for r in old}
    gen = EV.load_gen()
    sy, so = EV.System(tag, gen), EV.System(otag, gen)
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"] and (nm, x["src"]) in cer][: int(opt.get("nmax", 10 ** 6))] for nm, its in AL.build_sets(names, 64, split).items()}
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    St = {t: sy.structure(f"t|{t}", trefs[t]) for t in tgts}
    rows = []
    for nm, its in sets.items():
        for it in its:
            enr, c = it["enr"][:3], cer[(nm, it["src"])]
            src = sy.prep(it["u"])
            So = sy.structure(f"self|{nm}|{it['src']}", enr)
            w, yo = sy.convert(it["u"], enr, So, 0, with_y=True)
            row = {"set": nm, "spk": it["spk"], "src": it["src"], "cer_self": c["self"], "cer_x": c["x"],
                   "self": reread(sy, so, src, enr, w, yo, sy.op_stat(enr), so.op_stat(enr)), "x": []}
            wc = sy.render(src, torch.from_numpy(src["y"])[None].to(DEV), 0)
            row["copy"] = reread(sy, so, src, enr, wc, None, sy.op_stat(enr), so.op_stat(enr))
            for t in tgts:
                w, yo = sy.convert(it["u"], enr, St[t], 7, with_y=True)
                row["x"].append(reread(sy, so, src, enr, w, yo, sy.op_stat(trefs[t]), so.op_stat(trefs[t])))
            rows.append(row)
        rs = [r for r in rows if r["set"] == nm]
        print(nm, len(rs), {k: round(float(np.median([r[p][k] for r in rs])), 3) for p in ("self", "copy") for k in ("R_wav",)}, "x", round(float(np.median([x["R_wav"] for r in rs for x in r["x"]])), 3), flush=True)
    res = {"prereg": "results/snv_p78/prereg.yaml(step_a)", "sys": tag, "oth": otag, "tgts": tgts, "sets": {}}
    for nm in sets:
        rs = [r for r in rows if r["set"] == nm]
        med = lambda f: round(float(np.median(f)), 3)
        lv = {"源": len(rs), "CER(自己 / 別の声)": [round(float(np.mean([r["cer_self"] for r in rs])), 4), round(float(np.mean([np.mean(r["cer_x"]) for r in rs])), 4)],
              "写し": {k: med([r["copy"][k] for r in rs]) for k in ("R_wav", "R_oth", "floor", "vboth")},
              "自己の経路": {k: med([r["self"][k] for r in rs]) for k in ("R_wav", "R_env", "R_oth", "floor", "vboth")},
              "別の声": {k: med([x[k] for r in rs for x in r["x"]]) for k in ("R_wav", "R_env", "R_oth", "floor", "vboth")}}
        spk = np.array([r["spk"] for r in rs for _ in r["x"]])
        C = np.array([c for r in rs for c in r["cer_x"]])
        lv["別の声の試行で CER を当てる"] = {k: assoc(np.array([x[k] for r in rs for x in r["x"]]), C, spk) for k in ("R_wav", "R_env", "R_oth")}
        lv["自己の経路の試行で CER を当てる"] = {k: assoc(np.array([r["self"][k] for r in rs]), np.array([r["cer_self"] for r in rs]), np.array([r["spk"] for r in rs])) for k in ("R_wav", "R_env", "R_oth")}
        res["sets"][nm] = lv
    out.write_text(json.dumps(res | {"rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
