"""段 P31 補足 1(results/snv_p31/prereg.yaml の supplement_1・学習なし・閉じた形の ridge だけ): 自己の経路の細部の落ちは、符号の文脈 c そのものか、PCA 32 の切り方か。
符号 2 つ(snv_p24_xcpc / snv_p30_dcpc・各々の PCA と符号帳)× 表現の段 4 つ(c = 文脈 − 登録の平均 / pw = PCA 32 / w0_s6 = 登録の統計で白くする / w0_s6_cmn128 = 演奏の式・p18_nnc.code と同じ作り)で、
因果の 0〜100 ms の積み重ね(6 枚)から、同じフレームの包絡(c0・lo = c1〜c12・hi = c13〜c24・その声の登録の有声の平均を引き、当てはめの声の標準偏差で割る)への ridge の読み出し。
当てはめ = E1 の学習の声 train[::8] × t05・t06。λ = train[4::8] × t05 で 4 点から選ぶ。測る先 = 検証の声 × t05〜t08 と devA(先頭 n 人の有効な源・登録 e0〜e7)の有声のフレームの R²(声ごと)。
目標を遅い成分(65 ms の移動平均)と速い成分(残り)に分けた R² も並べる。対の差(新 − 対照)は声で束ねる。記録だけ: 検証の声の c の分散のうち PCA 32 が持つ割合。
段 P32 step a: codes=名前:符号の腕:PCA の基底[:符号帳],...(最初が基準)・reps=pw,w0_s6 で、基底の次元を替えた式を並べる(基底の列の数が次元)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p31_readout.py [n=64] [out=results/snv_p31/readout.json] [codes=...] [reps=...] [prereg=...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_nnc as N
import probe_rig as G
import train_snv_p1 as T
import train_snv_p3 as P3

ROOT = G.ROOT
DEV = "cuda"
CODES = {"snv_p24_xcpc": ("results/snv_p24/pca_c_xcpc.npy", "results/snv_p24/codebook_k128.npz"), "snv_p30_dcpc": ("results/snv_p30/pca_c.npy", "results/snv_p30/codebook_k128.npz")}
REPS = ("c", "pw", "w0_s6", "w0_s6_cmn128")
LAGS = (0, 4, 8, 12, 16, 20)
BANDS = {"c0": (0, 1), "lo": (1, 13), "hi": (13, 25)}
COMPS = ("full", "slow", "fast")
SLOW = 13
LAMS = (1e-3, 1e-2, 1e-1, 1.0)


def groups(n: int) -> dict:
    split = json.loads((ROOT / "results/snv_p30_c/split.json").read_text())
    e1 = lambda v, t: T.load_utt(T.FEAT / f"e1/{v}/{t}.npz")
    ev = lambda vs, ts: [{"g": v, "us": [e1(v, t) for t in ts], "enr": [e1(v, t) for t in ("t00", "t01", "t02")]} for v in vs]
    cv: dict = {}
    for it in G.cv_items(n):
        if it["valid"]:
            cv.setdefault(it["spk"], {"g": it["spk"], "us": [], "enr": it["enr"]})["us"].append(it["u"])
    return {"fit": ev(split["train"][::8], ("t05", "t06")), "sel": ev(split["train"][4::8], ("t05",)), "val": ev(split["val"], ("t05", "t06", "t07", "t08")), "cv": list(cv.values())}


def target(g: dict) -> list[torch.Tensor]:
    m = np.concatenate([u["y"][:25][:, u["v"] > 0.5] for u in g["enr"]], 1).mean(1, keepdims=True)
    return [torch.from_numpy(u["y"][:25] - m).T.to(DEV).double() for u in g["us"]]


def split3(y: torch.Tensor) -> torch.Tensor:
    p = SLOW // 2
    s = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(y.T[None], (p, p), mode="replicate"), SLOW, 1)[0].T
    return torch.cat([y, s, y - s], 1)


@torch.no_grad()
def feats(Pl, g: dict, rep: str) -> list[torch.Tensor]:
    us = g["us"] + g["enr"]
    X = N.code(Pl, us, list(range(len(g["us"]), len(us))), rep)[: len(g["us"])]
    return [torch.cat([torch.cat([x[:1].expand(l, -1), x[: len(x) - l]]) if l else x for l in LAGS] + [torch.ones_like(x[:, :1])], 1) for x in X]


def mask(g: dict) -> list[torch.Tensor]:
    return [torch.from_numpy(u["v"] > 0.5).to(DEV) for u in g["us"]]


def r2(Y: torch.Tensor, P: torch.Tensor) -> dict:
    o = {}
    for ci, c in enumerate(COMPS):
        for b, (a, z) in BANDS.items():
            y, p = Y[:, 25 * ci + a : 25 * ci + z], P[:, 25 * ci + a : 25 * ci + z]
            o[f"{b}_{c}"] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean(0)) ** 2).sum().clamp(min=1e-9))
    return o


def boot(d: np.ndarray) -> list[float]:
    g = np.random.default_rng(0)
    bs = [float(np.mean(g.choice(d, len(d)))) for _ in range(4000)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


@torch.no_grad()
def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = ROOT / opt.get("out", "results/snv_p31/readout.json")
    G_ = groups(int(opt.get("n", 64)))
    print({k: (len(v), sum(len(g["us"]) for g in v)) for k, v in G_.items()}, flush=True)
    tg = {k: [[split3(y) for y in target(g)] for g in v] for k, v in G_.items()}
    mk = {k: [mask(g) for g in v] for k, v in G_.items()}
    sd = torch.cat([y[m][:, :25] for ys, ms in zip(tg["fit"], mk["fit"]) for y, m in zip(ys, ms)]).std(0).clamp(min=1e-6).repeat(3)
    codes = {x.split(":")[0]: tuple(x.split(":")[1:]) for x in opt["codes"].split(",")} if "codes" in opt else {k: (k, *v) for k, v in CODES.items()}
    reps = tuple(opt.get("reps", ",".join(REPS)).split(","))
    res = {"prereg": opt.get("prereg", "results/snv_p31/prereg.yaml(supplement_1)"), "lags": LAGS, "rows": {}, "lam": {}, "pca32_share(検証の声)": {}}
    for code, (ck, pca, *cb) in codes.items():
        Pl = P3.PlayW(ROOT / "results" / ck / "last.pt", ROOT / pca, lags=(0,), shrink=0.6).to(DEV).eval()
        N.REPS["w0_s6_cmn128"]["cb"] = cb[0] if cb else None
        c = torch.cat([x[m] for g, ms in zip(G_["val"], mk["val"]) for x, m in zip(N.code(Pl, g["us"] + g["enr"], list(range(len(g["us"]), len(g["us"]) + 3)), "c")[: len(g["us"])], ms)]).double()
        res["pca32_share(検証の声)"][code] = round(float(((c @ Pl.P.double()) ** 2).sum() / (c ** 2).sum()), 4)
        for rep in reps:
            F = {k: [feats(Pl, g, rep) for g in v] for k, v in G_.items()}
            D = F["fit"][0][0].shape[1]
            A, B = torch.zeros(D, D, dtype=torch.float64, device=DEV), torch.zeros(D, 75, dtype=torch.float64, device=DEV)
            for xs, ys, ms in zip(F["fit"], tg["fit"], mk["fit"]):
                for x, y, m in zip(xs, ys, ms):
                    xm = x[m].double()
                    A += xm.T @ xm
                    B += xm.T @ (y[m] / sd)
            s = A.diagonal()[:-1].mean()
            reg = torch.eye(D, dtype=torch.float64, device=DEV)
            reg[-1, -1] = 0.0
            Ws = {lam: torch.linalg.solve(A + lam * s * reg, B) for lam in LAMS}

            def ev(k: str, W: torch.Tensor) -> list[dict]:
                rows = []
                for g, xs, ys, ms in zip(G_[k], F[k], tg[k], mk[k]):
                    X, Y = torch.cat([x[m] for x, m in zip(xs, ms)]).double(), torch.cat([y[m] for y, m in zip(ys, ms)]) / sd
                    rows.append({"g": g["g"]} | r2(Y, X @ W))
                return rows

            sel = {lam: float(np.mean([np.mean([r[f"{b}_full"] for b in BANDS]) for r in ev("sel", W)])) for lam, W in Ws.items()}
            lam = max(sel, key=sel.get)
            res["lam"][f"{code}|{rep}"] = {"chosen": lam, "sel_r2": {str(k): round(v, 4) for k, v in sel.items()}, "dim": D}
            for k in ("val", "cv"):
                res["rows"][f"{code}|{rep}|{k}"] = ev(k, Ws[lam])
            print(code, rep, "λ", lam, {k: {b: round(float(np.mean([r[f"{b}_full"] for r in res["rows"][f"{code}|{rep}|{k}"]])), 4) for b in BANDS} for k in ("val", "cv")}, flush=True)
            del F
        del Pl
    old = list(codes)[0]
    summ: dict = {}
    for k in ("val", "cv"):
        for new, rep in [(c, r) for c in list(codes)[1:] for r in reps]:
            a, b = res["rows"][f"{new}|{rep}|{k}"], res["rows"][f"{old}|{rep}|{k}"]
            assert [r["g"] for r in a] == [r["g"] for r in b]
            o = {}
            for key in [f"{bd}_{c}" for c in COMPS for bd in BANDS]:
                d = np.array([x[key] - y[key] for x, y in zip(a, b)])
                o[key] = {"対照": round(float(np.mean([y[key] for y in b])), 4), "新": round(float(np.mean([x[key] for x in a])), 4), "対の差": [round(float(d.mean()), 4), boot(d)]}
            summ[f"{k}|{rep}" if len(codes) == 2 else f"{k}|{rep}|{new}"] = o
    drop = lambda k, rep, c="full": -float(np.mean([summ[f"{k}|{rep}"][f"{bd}_{c}"]["対の差"][0] for bd in ("c0", "lo")]))
    res["summary"] = summ
    res["fork"] = {} if len(codes) != 2 or set(reps) != set(REPS) else {k: {"落ち(c0 と lo の平均)": {rep: round(drop(k, rep), 4) for rep in REPS}, "比(c の段 / 演奏の式の段)": round(drop(k, "c") / drop(k, "w0_s6_cmn128"), 3) if abs(drop(k, "w0_s6_cmn128")) > 1e-6 else None,
                       "速い成分の落ち / 遅い成分の落ち(演奏の式の段)": [round(drop(k, "w0_s6_cmn128", "fast"), 4), round(drop(k, "w0_s6_cmn128", "slow"), 4)]} for k in ("val", "cv")}
    print(json.dumps(res["pca32_share(検証の声)"], ensure_ascii=False))
    print(json.dumps(res["fork"], ensure_ascii=False, indent=1))
    for k, o in summ.items():
        print(k, json.dumps({q: v for q, v in o.items() if q.endswith("_full")}, ensure_ascii=False))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
