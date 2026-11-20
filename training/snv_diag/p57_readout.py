"""段 P57 step b の検査 I(results/snv_p57/prereg.yaml の step_b_check・学習なし・閉じた形の ridge だけ): 族の混ぜを通した入力の下で、凍結した符号と演奏の式は、きれいな包絡の情報を線形に持っているか。
p36_readout.py と同じ測り(読み出し W_c = きれいな入力で当てはめ・W_n = 足した入力で当てはめ・cc / nc / nn・因果の 0〜100 ms の積み重ね)を、段 P57 の族の特徴で行う。
rec = fr57(研究の形の前段を通した特徴)と v357(同じ混ぜを前段 v3 だけで解析した特徴)。版 k0 = Irodori の声(きれいな側 = data/snv_p_feat)・k1 = 擬似の操作者の版(きれいな側 = data/snv_p_feat_x16)。登録 t00〜t02 も同じ場。
層 = 場の背景の種類と SN 比(項目の台帳)。出力 = 層ごとの cc・nc・nn と、利得 nn − nc・残り cc − nn(声で束ねた区間)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p57_readout.py [recs=fr57,v357] [code=snv_p30_dcpc] [dn=a:f(段 P61・足した入力の源と登録の包絡から雑音の床を引く)] [ks=0,1] [out=results/snv_p57/readout.json]
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
import p36_readout as R36
import train_snv_p1 as T
import train_snv_p3 as P3

ROOT = R36.ROOT
DEV = R36.DEV
CODE, PCA, CB, REPS = "snv_p30_dcpc", "results/snv_p30/pca_c.npy", "results/snv_p30/codebook_k128.npz", ("c", "w0_s6_cmn128")
CLEAN = {0: T.FEAT, 1: ROOT / "data/snv_p_feat_x16"}


def stratum(m: dict) -> str:
    if m["bg"] == "none":
        return "none"
    return f"{m['bg']}|{'<5' if m['snr'] < 5 else '5-12' if m['snr'] < 12 else '>=12'}"


def trim(a: dict, b: dict) -> tuple[dict, dict]:
    n = min(a["y"].shape[1], b["y"].shape[1])
    cut = lambda u: {k: (v[..., :n] if isinstance(v, np.ndarray) and v.ndim and v.shape[-1] >= n and k in ("y", "lf", "v", "f0") else v) for k, v in u.items()}
    return cut(a), cut(b)


def groups(rec: Path, k: int, dn: tuple | None = None) -> dict:
    split = json.loads((ROOT / "results/snv_p30_c/split.json").read_text())

    def ev(vs: list[str], ts: tuple) -> list[dict]:
        out = []
        for v in vs:
            if not all((rec / f"e1/{v}/{t}_k{k}.npz").exists() for t in ts + R36.ENR):
                continue
            pr = [trim(T.load_utt(CLEAN[k] / f"e1/{v}/{t}.npz"), T.load_utt(rec / f"e1/{v}/{t}_k{k}.npz")) for t in ts + R36.ENR]
            m = json.loads((rec / f"e1/{v}/{ts[0]}_k{k}.json").read_text())
            us, xus, xenr = [p[0] for p in pr[: len(ts)]], [p[1] for p in pr[: len(ts)]], [p[1] for p in pr[len(ts):]]
            if dn is not None:
                import snv_dn as DN
                xus, xenr = DN.dn_utts(xus, xenr, *dn)
            out.append({"g": f"{v}|k{k}", "voice": v, "k": k, "st": stratum(m), "us": us, "enr": [p[0] for p in pr[len(ts):]], "xus": xus, "xenr": xenr})
        return out
    return {"fit": ev(split["train"][::8], ("t05", "t06")), "sel": ev(split["train"][4::8], ("t05",)), "val": ev(split["val"], ("t05", "t06", "t07", "t08"))}


@torch.no_grad()
def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = ROOT / opt.get("out", "results/snv_p57/readout.json")
    code = opt.get("code", CODE)
    res: dict = {"prereg": "results/snv_p57/prereg.yaml(step_b_check)", "code": code, "lags": R36.LAGS, "summary": {}, "n_groups": {}}
    Pl = P3.PlayW(ROOT / "results" / code / "last.pt", ROOT / PCA, lags=(0,), shrink=0.6).to(DEV).eval()
    N.REPS["w0_s6_cmn128"]["cb"] = CB
    for rname in opt.get("recs", "fr57,v357").split(","):
        for k in [int(q) for q in opt.get("ks", "0,1").split(",")]:
            G_ = groups(ROOT / f"data/snv_p_feat_{rname}", k, tuple(float(q) for q in opt["dn"].split(":")) if opt.get("dn") else None)
            res["n_groups"][f"{rname}|k{k}"] = {q: len(v) for q, v in G_.items()}
            st_n: dict = {}
            for g in G_["val"]:
                st_n[g["st"]] = st_n.get(g["st"], 0) + 1
            tg = {q: [R36.target(g) for g in v] for q, v in G_.items()}
            mk = {q: [[torch.from_numpy(u["v"] > 0.5).to(DEV) for u in g["us"]] for g in v] for q, v in G_.items()}
            sd = torch.cat([y[m] for ys, ms in zip(tg["fit"], mk["fit"]) for y, m in zip(ys, ms)]).std(0).clamp(min=1e-6)
            for rep in REPS:
                fx = lambda g: {"c": R36.feats(Pl, g["us"], g["enr"], rep), "n": R36.feats(Pl, g["xus"], g["xenr"], rep)}
                D = fx(G_["fit"][0])["c"][0].shape[1]
                AB = {nm: (torch.zeros(D, D, dtype=torch.float64, device=DEV), torch.zeros(D, 25, dtype=torch.float64, device=DEV)) for nm in ("c", "n")}
                for g, ys, ms in zip(G_["fit"], tg["fit"], mk["fit"]):
                    F = fx(g)
                    for nm in AB:
                        R36.accum(*AB[nm], F[nm], ys, ms, sd)
                Ws = {nm: R36.solve(*AB[nm]) for nm in AB}
                sel: dict = {nm: {lam: [] for lam in R36.LAMS} for nm in AB}
                for g, ys, ms in zip(G_["sel"], tg["sel"], mk["sel"]):
                    F = fx(g)
                    for nm in AB:
                        for lam in R36.LAMS:
                            sel[nm][lam].append(float(np.mean(list(R36.score(F[nm], ys, ms, sd, Ws[nm][lam]).values()))))
                W = {nm: Ws[nm][max(sel[nm], key=lambda lam: float(np.mean(sel[nm][lam])))] for nm in AB}
                Rr: dict = {"cc": [], "nc": [], "nn": []}
                for g, ys, ms in zip(G_["val"], tg["val"], mk["val"]):
                    F = fx(g)
                    Rr["cc"].append(R36.score(F["c"], ys, ms, sd, W["c"]))
                    Rr["nc"].append(R36.score(F["n"], ys, ms, sd, W["c"]))
                    Rr["nn"].append(R36.score(F["n"], ys, ms, sd, W["n"]))
                S: dict = {}
                for st in sorted(st_n) + ["music|all"]:
                    ix = [i for i, g in enumerate(G_["val"]) if (g["st"] == st or (st == "music|all" and g["st"].startswith("music")))]
                    if not ix:
                        continue
                    o: dict = {"n": len(ix)}
                    for b in R36.BANDS:
                        v = {q: np.array([Rr[q][i][b] for i in ix]) for q in Rr}
                        o[b] = {q: round(float(x.mean()), 4) for q, x in v.items()} | {"nn−nc": [round(float((v["nn"] - v["nc"]).mean()), 4), R36.boot(v["nn"] - v["nc"])], "cc−nn": [round(float((v["cc"] - v["nn"]).mean()), 4), R36.boot(v["cc"] - v["nn"])],
                                                                                 "利得の割合": round(float((v["nn"] - v["nc"]).mean() / max(1e-9, (v["cc"] - v["nc"]).mean())), 3)}
                    S[st] = o
                res["summary"][f"{rname}|k{k}|{rep}"] = S
                print(rname, f"k{k}", rep, json.dumps({st: [o["n"], o["lo"]["cc"], o["lo"]["nc"], o["lo"]["nn"], o["lo"]["利得の割合"]] for st, o in S.items()}, ensure_ascii=False), flush=True)
                del AB, Ws
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
