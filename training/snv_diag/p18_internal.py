"""段 P18 B(results/snv_p18/prereg.yaml・学習なし・ASR なし): 源の発話ごとの内部の量。
  sup64・sup128 = 演奏(線形 64)/ 白くした符号(128)の有声のフレームから、族の中の学習の声の銀行(E1 の学習の声 150 × t05・t06・各々 t00〜t02 の統計で白くする)への 10 番目の近傍の距離の中央値を、
                  族の中の検証の声(36 声 × t05・t06)の同じ量の中央値で割った比(1 = 族の中)
  tgt64 = 源の演奏から、割り当ての目標 2 声の参照の演奏への最も近い距離の中央値(同じ割り方)
  effT・maxT / effS・maxS = 目標の構造 / 自分の構造での注意(C の 4 段目)の実効のフレーム数と最大の重み(頭の平均・有声のフレームの中央値・k 7)
  skew・kurt = 白くした符号の次元ごとの歪度・尖度 − 3 の絶対値の平均 / wherr = 登録の統計で白くした源の共分散と単位行列の差(フロベニウス / √128)
  cond・erank = 登録の共分散(縮みの前)の条件数の log10・実効の階数 / covd = 登録の共分散(縮みの後)の行列の対数と、銀行の声の平均との距離
組: cv(devA の先頭 n 人 × s0・s1)・valm(E1 の検証の男声 × t05〜t08)・valf(検証の女声 × t05)・org_g0・org_w・org_g0_ch・org_w_ch・jv。
devA と生成した声の模型は results/snv_p15/rows.json の CER(他の声の構造 k 7・自己)に結ぶ。出力 = results/snv_p18/internal.json。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p18_internal.py [sys=snv_p8_wst] [n=64] [out=results/snv_p18/internal.json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_rig as G
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV
KNN = 10
KEYS = ("sup64", "sup128", "tgt64", "effT", "maxT", "effS", "maxS", "skew", "kurt", "wherr", "cond", "erank", "covd")


def tt(u: dict, k: str) -> torch.Tensor:
    return torch.from_numpy(u[k])[None].to(DEV)


@torch.no_grad()
def stat_of(Pl, enr: list[dict]) -> dict:
    ry, rv, rm, _ = EV.struct_inputs(enr)
    st = Pl.mean_of(ry, rv, rm)
    w = (rv * rm.float())[:, None].double()
    S = Pl.stack(Pl.ctx(ry, rv), st[0]).double()
    cov = (torch.einsum("bit,bjt->bij", S * w, S) / w.sum(-1).clamp(min=1.0)[..., None])[0]
    e = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
    D = cov.shape[-1]
    es, q = torch.linalg.eigh((1 - Pl.shrink) * cov + Pl.shrink * cov.diagonal().sum() / D * torch.eye(D, dtype=cov.dtype, device=cov.device))
    return {"st": st, "cond": float(torch.log10(e[-1] / e[0])), "erank": float(e.sum() ** 2 / (e * e).sum()), "logc": (q * es.clamp(min=1e-12).log()) @ q.T}


@torch.no_grad()
def play_of(Pl, u: dict, st: tuple) -> tuple:
    y, v = tt(u, "y"), tt(u, "v")
    W = Pl.white(y, v, st)
    return Pl(y, v, st), W, v


@torch.no_grad()
def kth(X: torch.Tensor, bank: torch.Tensor, k: int) -> float:
    d = torch.cat([torch.cdist(X[i:i + 512], bank).topk(k, largest=False).values[:, -1] for i in range(0, len(X), 512)])
    return float(d.median())


@torch.no_grad()
def attn(C, a: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, S: dict) -> tuple[float, float]:
    h = C.inp(torch.cat([a, lf[:, None], v[:, None]], 1))
    for i, blk in enumerate(C.blocks):
        h = blk(h, S["z"])
        if i == 3:
            B, _, n = h.shape
            q = C.q(h.transpose(1, 2)).view(B, n, C.heads, C.dk).transpose(1, 2)
            p = (q @ S["k"].transpose(-1, -2) / math.sqrt(C.dk)).softmax(-1)
            m = v[0] > 0.5
            return float((1.0 / (p * p).sum(-1)).mean(1)[0][m].median()), float(p.max(-1).values.mean(1)[0][m].median())
    raise AssertionError


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, tag = int(opt.get("n", 64)), opt.get("sys", "snv_p8_wst")
    out = ROOT / opt.get("out", "results/snv_p18/internal.json")
    s = EV.System(tag, EV.load_gen())
    assert s.p3
    Pl, C = s.Pl, s.C
    name = tag.partition("@")[0]
    split = json.loads((ROOT / "results" / name / "split.json").read_text())
    e1 = lambda v, t: T.load_utt(T.FEAT / f"e1/{v}/{t}.npz")
    g = torch.Generator(device="cpu").manual_seed(0)
    b64, b128, logc = [], [], []
    for v in split["train"][:: max(1, len(split["train"]) // 150)][:150]:
        so = stat_of(Pl, [e1(v, t) for t in ("t00", "t01", "t02")])
        logc.append(so["logc"])
        for t in ("t05", "t06"):
            A, W, vv = play_of(Pl, e1(v, t), so["st"])
            idx = torch.nonzero(vv[0] > 0.5)[:, 0]
            idx = idx[torch.randperm(len(idx), generator=g)[:300].to(idx.device)]
            b64.append(A[0][:, idx].T)
            b128.append(W[0][:, idx].T)
    b64, b128, logc_mu = torch.cat(b64), torch.cat(b128), torch.stack(logc).mean(0)
    print("銀行", tuple(b64.shape), tuple(b128.shape), flush=True)
    all_t, tg4 = G.targets()
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    tst = {v: s.structure(f"t|{v}|3", trefs[v]) for v in tg4}
    tplay = {}
    for v in tg4:
        ry, rv, rm, _ = EV.struct_inputs(trefs[v])
        tplay[v] = Pl(ry, rv, Pl.mean_of(ry, rv, rm))[0][:, rv[0] > 0.5].T
    val = split["val"]
    sets = {"cv": G.cv_items(n),
            "valm": [{"set": "valm", "spk": v, "idx": i, "src": f"{v}|{t}", "u": e1(v, t), "enr": [e1(v, o) for o in ("t00", "t01", "t02")], "valid": True} for i, v in enumerate([x for x in val if x.startswith("em")]) for t in ("t05", "t06", "t07", "t08")],
            "valf": [{"set": "valf", "spk": v, "idx": i, "src": f"{v}|t05", "u": e1(v, "t05"), "enr": [e1(v, o) for o in ("t00", "t01", "t02")], "valid": True} for i, v in enumerate([x for x in val if x.startswith("ef")])],
            "jv": G.jv_items()} | {f"org_{k}": G.org_items(k) for k in G.ORG}
    rows = []
    cache: dict = {}
    for name_, its in sets.items():
        for it in its:
            key = (it["set"], it["spk"], tuple(id(u) for u in it["enr"]))
            if key not in cache:
                cache[key] = stat_of(Pl, it["enr"])
            so = cache[key]
            A, W, v = play_of(Pl, it["u"], so["st"])
            m = v[0] > 0.5
            a, w = A[0][:, m].T, W[0][:, m].T
            z = (w - w.mean(0)) / w.std(0).clamp(min=1e-6)
            cw = (w - w.mean(0)).T @ (w - w.mean(0)) / max(1, len(w) - 1)
            tgs = (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4])
            lf = tt(it["u"], "lf") + 7 * T.LN2_12 * v
            SS = s.structure(f"self|p18b|{it['set']}|{it['spk']}|{it['src']}", it["enr"])
            eT = [attn(C, A, lf, v, tst[x]) for x in tgs]
            eS = attn(C, A, lf, v, SS)
            s.cache.pop(f"self|p18b|{it['set']}|{it['spk']}|{it['src']}", None)
            rows.append({"set": it["set"], "spk": it["spk"], "src": it["src"], "valid": bool(it["valid"]), "n_voiced": int(m.sum()),
                         "sup64": kth(a, b64, KNN), "sup128": kth(w, b128, KNN), "tgt64": float(np.mean([float(torch.cdist(a, tplay[x]).min(1).values.median()) for x in tgs])),
                         "effT": float(np.mean([e[0] for e in eT])), "maxT": float(np.mean([e[1] for e in eT])), "effS": eS[0], "maxS": eS[1],
                         "skew": float((z ** 3).mean(0).abs().mean()), "kurt": float(((z ** 4).mean(0) - 3).abs().mean()),
                         "wherr": float(torch.linalg.norm(cw - torch.eye(cw.shape[0], device=cw.device)) / math.sqrt(cw.shape[0])),
                         "cond": so["cond"], "erank": so["erank"], "covd": float(torch.linalg.norm(so["logc"] - logc_mu))})
        print(name_, len(its), flush=True)
    ref = {k: float(np.median([r[k] for r in rows if r["set"] in ("valm", "valf")])) for k in ("sup64", "sup128", "tgt64")}
    for r in rows:
        for k in ref:
            r[k] = r[k] / ref[k]
    pop = {st: {k: [round(float(q), 3) for q in np.percentile([r[k] for r in rows if r["set"] == st and r["valid"]], [25, 50, 75])] for k in KEYS} for st in sets}
    P15 = json.loads((ROOT / "results/snv_p15/rows.json").read_text())["rows"]
    idx = {it["spk"]: it["idx"] for it in sets["cv"]}
    cross: dict = {}
    slf: dict = {}
    for r in P15["base4"]:
        if r["k"] == 7 and r["tgt"] in (tg4[idx[r["spk"]] % 4], tg4[(idx[r["spk"]] + 2) % 4]):
            cross.setdefault(r["src"], []).append(r["cer"])
    for r in P15["self"]:
        slf[r["src"]] = r["cer"]
    for c in ("org_g0", "org_w", "org_g0_ch", "org_w_ch"):
        for r in P15[c]:
            if r["tgt"] is None:
                slf[f"{r['src']}|{c}"] = r["cer"]
            else:
                cross.setdefault(f"{r['src']}|{c}", []).append(r["cer"])
    for r in rows:
        k = r["src"] if r["set"] == "cv" else r["src"] + "|" + r["set"]
        if k in cross and k in slf:
            r["cer_cross"], r["cer_self"] = float(np.mean(cross[k])), float(slf[k])
    cv = [r for r in rows if r["set"] == "cv" and r["valid"] and "cer_cross" in r]
    spk = sorted({r["spk"] for r in cv})
    agg = lambda k: np.array([np.mean([r[k] for r in cv if r["spk"] == sp]) for sp in spk])
    rho = {"n_src": len(cv), "n_spk": len(spk)}
    for k in KEYS:
        x = agg(k)
        rho[k] = {"cross": round(float(spearmanr(x, agg("cer_cross"))[0]), 3), "self": round(float(spearmanr(x, agg("cer_self"))[0]), 3), "gap": round(float(spearmanr(x, agg("cer_cross") - agg("cer_self"))[0]), 3),
                  "src_cross": round(float(spearmanr([r[k] for r in cv], [r["cer_cross"] for r in cv])[0]), 3)}
    res = {"prereg": "results/snv_p18/prereg.yaml", "sys": tag, "ref_val_median(割る前)": ref, "pop(25 / 50 / 75 %)": pop, "rho_devA_speakers": rho, "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
