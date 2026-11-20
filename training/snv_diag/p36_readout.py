"""段 P36 step a2(results/snv_p36/prereg.yaml の step_a2・学習なし・閉じた形の ridge だけ): 足された音の下で、符号と演奏はきれいな包絡の情報を線形に持っているか(C が読めるか)。
入力 = 本物の非音声の音源を足した版(data/snv_p_feat_rec2・声ごとに場 k 0 / 1・登録 t00〜t02 も同じ場)の符号。目標 = 同じ発話のきれいな包絡(c0・lo = c1〜c12・hi = c13〜c24・その声のきれいな登録の有声の平均を引き、当てはめの声の標準偏差で割る)。
読み出し 2 つ: W_c = きれいな入力で当てはめる(今の C の立場)・W_n = 足した入力で当てはめる(足した入力で学んだ読み手の上限)。因果の 0〜100 ms の積み重ね(6 枚)。
当てはめ = E1 の学習の声 train[::8] × t05・t06 × 場 2 つ。λ = train[4::8] × t05 で 4 点から選ぶ。測る先 = 検証の声 × t05〜t08 × 場 2 つの、きれいな音で有声のフレームの R²(場ごと)。
並べる値: cc = きれいな入力 + W_c・nc = 足した入力 + W_c・nn = 足した入力 + W_n。場の背景の種類と SN 比で層に分け、声で束ねた区間を出す。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p36_readout.py [out=results/snv_p36/readout.json]
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
REC = ROOT / "data/snv_p_feat_rec2"
CODES = {"snv_p30_dcpc": ("results/snv_p30/pca_c.npy", "results/snv_p30/codebook_k128.npz", ("c", "w0_s6_cmn128")),
         "snv_p25_xrcpc": ("results/snv_p24/pca_c_xcpc.npy", None, ("c",)),
         "snv_p24_xcpc": ("results/snv_p24/pca_c_xcpc.npy", "results/snv_p24/codebook_k128.npz", ("c", "w0_s6_cmn128"))}
ENR = ("t00", "t01", "t02")
LAGS = (0, 4, 8, 12, 16, 20)
BANDS = {"c0": (0, 1), "lo": (1, 13), "hi": (13, 25)}
LAMS = (1e-3, 1e-2, 1e-1, 1.0)


def stratum(room: dict) -> str:
    bg, snr = room["bg"], room.get("snr_db", 99.0)
    if bg == "none":
        return "none"
    return f"{bg}|{'<10' if snr < 10 else '10-20' if snr < 20 else '>=20'}"


def groups() -> dict:
    split = json.loads((ROOT / "results/snv_p30_c/split.json").read_text())
    rooms = json.loads((REC / "rooms.json").read_text())
    e1 = lambda v, t: T.load_utt(T.FEAT / f"e1/{v}/{t}.npz")
    rc = lambda v, t, k: T.load_utt(REC / f"e1/{v}/{t}_k{k}.npz")

    def ev(vs: list[str], ts: tuple) -> list[dict]:
        out = []
        for v in vs:
            us, enr = [e1(v, t) for t in ts], [e1(v, t) for t in ENR]
            for k in (0, 1):
                xus, xenr = [rc(v, t, k) for t in ts], [rc(v, t, k) for t in ENR]
                assert all(a["y"].shape == b["y"].shape for a, b in zip(us + enr, xus + xenr)), v
                out.append({"g": f"{v}|k{k}", "voice": v, "k": k, "st": stratum(rooms[v][str(k)]), "snr": rooms[v][str(k)].get("snr_db"), "us": us, "enr": enr, "xus": xus, "xenr": xenr})
        return out
    return {"fit": ev(split["train"][::8], ("t05", "t06")), "sel": ev(split["train"][4::8], ("t05",)), "val": ev(split["val"], ("t05", "t06", "t07", "t08"))}


def target(g: dict) -> list[torch.Tensor]:
    m = np.concatenate([u["y"][:25][:, u["v"] > 0.5] for u in g["enr"]], 1).mean(1, keepdims=True)
    return [torch.from_numpy(u["y"][:25] - m).T.to(DEV).double() for u in g["us"]]


@torch.no_grad()
def feats(Pl, us: list[dict], enr: list[dict], rep: str) -> list[torch.Tensor]:
    X = N.code(Pl, us + enr, list(range(len(us), len(us) + len(enr))), rep)[: len(us)]
    return [torch.cat([torch.cat([x[:1].expand(l, -1), x[: len(x) - l]]) if l else x for l in LAGS] + [torch.ones_like(x[:, :1])], 1) for x in X]


def r2(Y: torch.Tensor, P: torch.Tensor) -> dict:
    return {b: float(1 - ((Y[:, a:z] - P[:, a:z]) ** 2).sum() / ((Y[:, a:z] - Y[:, a:z].mean(0)) ** 2).sum().clamp(min=1e-9)) for b, (a, z) in BANDS.items()}


def boot(d: np.ndarray) -> list[float]:
    g = np.random.default_rng(0)
    bs = d[g.integers(0, len(d), (20000, len(d)))].mean(1)
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def accum(A: torch.Tensor, B: torch.Tensor, xs: list, ys: list, ms: list, sd: torch.Tensor) -> None:
    for x, y, m in zip(xs, ys, ms):
        xm = x[m].double()
        A += xm.T @ xm
        B += xm.T @ (y[m] / sd)


def solve(A: torch.Tensor, B: torch.Tensor) -> dict:
    s = A.diagonal()[:-1].mean()
    reg = torch.eye(A.shape[0], dtype=torch.float64, device=DEV)
    reg[-1, -1] = 0.0
    return {lam: torch.linalg.solve(A + lam * s * reg, B) for lam in LAMS}


def score(xs: list, ys: list, ms: list, sd: torch.Tensor, W: torch.Tensor) -> dict:
    X, Y = torch.cat([x[m] for x, m in zip(xs, ms)]).double(), torch.cat([y[m] for y, m in zip(ys, ms)]) / sd
    return r2(Y, X @ W)


@torch.no_grad()
def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = ROOT / opt.get("out", "results/snv_p36/readout.json")
    G_ = groups()
    print({k: (len(v), sum(len(g["us"]) for g in v)) for k, v in G_.items()}, flush=True)
    st_n: dict = {}
    for g in G_["val"]:
        st_n[g["st"]] = st_n.get(g["st"], 0) + 1
    print("検証の場の層", json.dumps(st_n, ensure_ascii=False), flush=True)
    tg = {k: [target(g) for g in v] for k, v in G_.items()}
    mk = {k: [[torch.from_numpy(u["v"] > 0.5).to(DEV) for u in g["us"]] for g in v] for k, v in G_.items()}
    sd = torch.cat([y[m] for ys, ms in zip(tg["fit"], mk["fit"]) for y, m in zip(ys, ms)]).std(0).clamp(min=1e-6)
    res: dict = {"prereg": "results/snv_p36/prereg.yaml(step_a2)", "lags": LAGS, "strata(検証の場の数)": st_n, "rows": {}, "lam": {}, "summary": {}}
    for code, (pca, cb, reps) in CODES.items():
        Pl = P3.PlayW(ROOT / "results" / code / "last.pt", ROOT / pca, lags=(0,), shrink=0.6).to(DEV).eval()
        N.REPS["w0_s6_cmn128"]["cb"] = cb
        for rep in reps:
            fx = lambda g: {"c": feats(Pl, g["us"], g["enr"], rep), "n": feats(Pl, g["xus"], g["xenr"], rep)}
            D = fx(G_["fit"][0])["c"][0].shape[1]
            AB = {nm: (torch.zeros(D, D, dtype=torch.float64, device=DEV), torch.zeros(D, 25, dtype=torch.float64, device=DEV)) for nm in ("c", "n")}
            for g, ys, ms in zip(G_["fit"], tg["fit"], mk["fit"]):
                F = fx(g)
                for nm in AB:
                    accum(*AB[nm], F[nm], ys, ms, sd)
            Ws = {nm: solve(*AB[nm]) for nm in AB}
            sel: dict = {nm: {lam: [] for lam in LAMS} for nm in AB}
            for g, ys, ms in zip(G_["sel"], tg["sel"], mk["sel"]):
                F = fx(g)
                for nm in AB:
                    for lam in LAMS:
                        sel[nm][lam].append(float(np.mean(list(score(F[nm], ys, ms, sd, Ws[nm][lam]).values()))))
            W = {}
            for nm in AB:
                m_ = {lam: float(np.mean(v)) for lam, v in sel[nm].items()}
                lam = max(m_, key=m_.get)
                W[nm] = Ws[nm][lam]
                res["lam"][f"{code}|{rep}|W_{nm}"] = {"chosen": lam, "sel_r2": {str(k): round(v, 4) for k, v in m_.items()}}
            R: dict = {"cc": [], "nc": [], "nn": []}
            for g, ys, ms in zip(G_["val"], tg["val"], mk["val"]):
                F = fx(g)
                hd = {"g": g["g"], "voice": g["voice"], "st": g["st"]}
                R["cc"].append(hd | score(F["c"], ys, ms, sd, W["c"]))
                R["nc"].append(hd | score(F["n"], ys, ms, sd, W["c"]))
                R["nn"].append(hd | score(F["n"], ys, ms, sd, W["n"]))
            del F, AB, Ws
            res["rows"][f"{code}|{rep}"] = R
            S: dict = {}
            for st in sorted(st_n):
                ix = [i for i, g in enumerate(G_["val"]) if g["st"] == st]
                o: dict = {"n": len(ix)}
                for b in BANDS:
                    v = {q: np.array([R[q][i][b] for i in ix]) for q in R}
                    o[b] = {q: round(float(x.mean()), 4) for q, x in v.items()} | {"nn−nc": [round(float((v["nn"] - v["nc"]).mean()), 4), boot(v["nn"] - v["nc"])], "cc−nn": [round(float((v["cc"] - v["nn"]).mean()), 4), boot(v["cc"] - v["nn"])]}
                S[st] = o
            res["summary"][f"{code}|{rep}"] = S
            print(code, rep, json.dumps({st: {b: [o[b]["cc"], o[b]["nc"], o[b]["nn"]] for b in ("c0", "lo")} for st, o in S.items()}, ensure_ascii=False), flush=True)
        del Pl
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
