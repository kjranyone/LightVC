"""段 P18 C(results/snv_p18/prereg.yaml の補足 1・学習なし): 内容を合わせた外れ。外の調音の物差し(SPARC の EMA 12 次元・物差しとしてだけ)で源のフレームと目標の声のフレームを対にし、
白くした符号 W(128)と演奏 A(線形 64)がその対の間でどれだけ合うか・どの類の写像で合うようになるかを測る。
対 = 源の有声のフレーム t に、目標の声の参照(j 0〜5 の有声のフレーム)から、EMA(話者ごとに標準化・遅れ 0 / 40 / 80 / 120 ms を積む)が最も近いフレーム j*(t)。
量(源の発話 × 割り当ての目標 2 声):
  ceil = EMA どうしの対の相関(物差しの上限)/ nnc_W・nnc_A = 表現の最近傍で選んだフレームとの EMA の相関(snv_p2_nnc.py と同じ)
  r2_* ・cos_* = 対の目標の側の符号を、源の側の符号(写像の後)がどれだけ説明するか(1 − 残差 / 目標の分散)と余弦の平均。写像 = id(恒等)・orth(直交)・lin(線形 + 定数・縮み 0.1)。
  写像は、その話者の登録のフレームの対だけで当てはめ、源の発話で測る(_x = もう 1 つの割り当ての目標で測る)。size_* = ‖写像 − I‖ / √D。
  common = devA の話者を半分に分け、片方の登録で当てはめた 1 つの線形の写像を、もう片方の源で測る(本物の声に共通の成分)。
組: valf・valm(E1 の検証の声)・trm(E1 の学習の男声の先頭 48 声 × t05・段 P70 = 男声を除いて学んだ符号にとっての、学習に無い族の中の男声)・x16(検証の声の擬似の操作者の版 = 段 P16 の族)・org_g0・org_w・org_g0_ch・org_w_ch・jv・cv(devA)。
conv=1(診断・製品の候補でない): 登録で当てはめた写像(orth・lin)を源の白くした符号に掛けてから変換し、CER と M を測る(線形の置き違いを知っていたら戻るかの上限)。組 = org_g0(対照)・org_w_ch・cv。
出力 = results/snv_p18/align.json(conv=1 は align_conv.json)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p18_align.py [sys=snv_p8_wst] [n=64] [conv=1] [sets=...]
"""
from __future__ import annotations

import json
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
SP = G.SP
LAGS = (0, 8, 16, 24)
T_HOP = 240 / 48000
RIDGE = 0.1
SETS = ("valf", "valm", "x16", "org_g0", "org_w", "org_g0_ch", "org_w_ch", "jv", "cv")
CONV = ("org_g0", "org_w_ch", "cv")
K = 7


def ema_at(npz: str, n: int) -> torch.Tensor:
    e = np.load(npz)["ema"][:, :12]
    t = np.arange(n) * T_HOP
    return torch.from_numpy(np.stack([np.interp(t, np.arange(len(e)) / 50.0, e[:, k]) for k in range(12)], 1).astype(np.float32)).to(DEV)


def stack(e: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.cat([e[:1].expand(l, -1), e[: len(e) - l]]) if l else e for l in LAGS], 1)


def tt(u: dict, k: str) -> torch.Tensor:
    return torch.from_numpy(u[k])[None].to(DEV)


@torch.no_grad()
def unit(Pl, utts: list[dict], emas: list[str], enr: list[int]) -> tuple[list[dict], tuple]:
    ry, rv, rm, _ = EV.struct_inputs([utts[i] for i in enr])
    st = Pl.mean_of(ry, rv, rm)
    out = []
    for u, p in zip(utts, emas):
        y, v = tt(u, "y"), tt(u, "v")
        W = Pl.white(y, v, st)
        m = v[0] > 0.5
        e = ema_at(p, y.shape[-1])
        out.append({"W": W[0].T[m], "A": Pl(y, v, st)[0].T[m], "E": e[m], "Ec": stack(e)[m]})
    E = torch.cat([o["E"] for o in out])
    mu, sd = E.mean(0), E.std(0).clamp(min=1e-6)
    for o in out:
        o["E"], o["Ec"] = (o["E"] - mu) / sd, (o["Ec"] - mu.repeat(len(LAGS))) / sd.repeat(len(LAGS))
    return out, st


def cat(us: list[dict]) -> dict:
    return {k: torch.cat([u[k] for u in us]) for k in ("W", "A", "E", "Ec")}


@torch.no_grad()
def nn(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.cdist(q[i:i + 1024], k).argmin(1) for i in range(0, len(q), 1024)])


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(0), b - b.mean(0)
    return float(((a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0)).clamp(min=1e-8)).mean())


def fit(X: torch.Tensor, Y: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    D = X.shape[1]
    I = torch.eye(D, dtype=torch.float64, device=X.device)
    if kind == "id":
        return I.float(), torch.zeros(D, device=X.device)
    X, Y = X.double(), Y.double()
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    if kind == "orth":
        U, _, Vt = torch.linalg.svd(Xc.T @ Yc)
        M = U @ Vt
    else:
        M = torch.linalg.solve(Xc.T @ Xc + RIDGE * len(X) * (Xc * Xc).sum() / (len(X) * D) * I, Xc.T @ Yc)
    return M.float(), (my - mx @ M).float()


def score(X: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
    return float(1 - ((X - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()), float(torch.nn.functional.cosine_similarity(X, Y, dim=1).mean())


def measure(src: dict, enr: dict, pools: dict, v: str, vx: str) -> dict:
    P, Px = pools[v], pools[vx]
    je, jx, jf = nn(src["Ec"], P["Ec"]), nn(src["Ec"], Px["Ec"]), nn(enr["Ec"], P["Ec"])
    r = {"ceil": corr(src["E"], P["E"][je]), "n": int(len(je))}
    for key in ("W", "A"):
        r[f"nnc_{key}"] = corr(src["E"], P["E"][nn(src[key], P[key])])
        for kind in ("id", "orth", "lin"):
            M, b = fit(enr[key], P[key][jf], kind)
            r[f"r2_{kind}_{key}"], r[f"cos_{kind}_{key}"] = score(src[key] @ M + b, P[key][je])
            r[f"r2_{kind}_{key}_x"], r[f"cos_{kind}_{key}_x"] = score(src[key] @ M + b, Px[key][jx])
            if kind != "id":
                r[f"size_{kind}_{key}"] = float(torch.linalg.norm(M - torch.eye(M.shape[0], device=M.device)) / M.shape[0] ** 0.5)
    return r


def build_sets(names: list[str], n: int, split: dict) -> dict:
    e1 = lambda v, t: T.load_utt(T.FEAT / f"e1/{v}/{t}.npz")
    x16 = lambda v, t: T.load_utt(ROOT / f"data/snv_p_feat_x16/e1/{v}/{t}.npz")
    ev = lambda nm, vs, ts, ld: [{"set": nm, "spk": v, "idx": i, "src": f"{v}|{t}", "u": ld(v, t), "enr": [ld(v, o) for o in ("t00", "t01", "t02")], "valid": True,
                                  "ema": [str(SP / f"e1/{v}/{t}.npz")] + [str(SP / f"e1/{v}/{o}.npz") for o in ("t00", "t01", "t02")]} for i, v in enumerate(vs) for t in ts]
    val = split["val"]
    rec2 = lambda v, t: T.load_utt(ROOT / f"data/snv_p_feat_rec2/e1/{v}/{t}.npz")
    rv = lambda: [{"set": "rec2", "spk": f"{v}|k{k}", "idx": i, "src": f"{v}|t05|k{k}", "u": rec2(v, f"t05_k{k}"), "enr": [rec2(v, f"{o}_k{k}") for o in ("t00", "t01", "t02")], "valid": True,
                   "ema": [str(SP / f"e1/{v}/t05.npz")] + [str(SP / f"e1/{v}/{o}.npz") for o in ("t00", "t01", "t02")]} for i, v in enumerate(val) for k in (0, 1)]
    rx = lambda: [x | {"set": "rec2x", "enr": [rec2(x["spk"].split("|")[0], f"{o}_k{1 - int(x['spk'][-1])}") for o in ("t00", "t01", "t02")]} for x in rv()]
    mk = {"valf": lambda: ev("valf", [x for x in val if x.startswith("ef")][::2], ("t05",), e1), "valm": lambda: ev("valm", [x for x in val if x.startswith("em")], ("t05", "t06"), e1),
          "trm": lambda: ev("trm", [x for x in split["train"] if x.startswith("em")][:48], ("t05",), e1),
          "x16": lambda: ev("x16", val[::2], ("t05",), x16), "rec2": rv, "rec2x": rx, "jv": G.jv_items, "cv": lambda: G.cv_items(n)} | {f"org_{k}": (lambda k=k: G.org_items(k)) for k in G.ORG}
    return {nm: G.ext_items(nm[2:]) if nm.startswith("x:") else mk[nm]() for nm in names}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, tag, conv = int(opt.get("n", 64)), opt.get("sys", "snv_p8_wst"), opt.get("conv") == "1"
    names = [c for c in opt.get("sets", ",".join(CONV if conv else SETS)).split(",")]
    out = ROOT / opt.get("out", f"results/snv_p18/align{'_conv' if conv else ''}.json")
    split = json.loads((ROOT / "results" / tag.partition("@")[0] / "split.json").read_text())
    sets = build_sets(names, n, split)
    rig = G.Rig(tag) if conv else None
    s = rig.s if conv else EV.System(tag, EV.load_gen())
    assert s.p3
    Pl = s.Pl
    _, tg4 = G.targets()
    pools = {}
    for v in tg4:
        us, _ = unit(Pl, [EV.utt(f"a0/{v}_{j}") for j in range(6)], [str(SP / f"a0/{v}_{j}.npz") for j in range(6)], [0, 1, 2])
        pools[v] = cat(us)
    print("目標の参照のフレーム", {v: len(p["E"]) for v, p in pools.items()}, flush=True)
    rows, units = [], {}
    for nm, its in sets.items():
        for it in its:
            us, st = unit(Pl, [it["u"]] + it["enr"], it["ema"], list(range(1, 1 + len(it["enr"]))))
            src, enr = us[0], cat(us[1:])
            tgs = (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4])
            units[(nm, it["src"])] = (it, src, enr, st, tgs)
            if not conv:
                for v, vx in (tgs, tgs[::-1]):
                    rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "valid": bool(it["valid"]), "tgt": v} | measure(src, enr, pools, v, vx))
        print(nm, len(its), flush=True)
    if not conv:
        keys = [k for k in rows[0] if k not in ("set", "spk", "src", "valid", "tgt", "n")]
        pop = {nm: {k: round(float(np.median([r[k] for r in rows if r["set"] == nm and r["valid"]])), 3) for k in keys} for nm in sets}
        res = {"prereg": "results/snv_p18/prereg.yaml(補足 1)", "sys": tag, "pop(中央値)": pop}
        if "cv" in sets:
            cv = [u for (nm, _), u in units.items() if nm == "cv"]
            spk = sorted({u[0]["spk"] for u in cv})
            A, B = set(spk[::2]), set(spk[1::2])
            com = {}
            for key in ("W", "A"):
                X = torch.cat([u[2][key] for u in cv if u[0]["spk"] in A and u[0]["src"].endswith("_s0")])
                Y = torch.cat([pools[u[4][0]][key][nn(u[2]["Ec"], pools[u[4][0]]["Ec"])] for u in cv if u[0]["spk"] in A and u[0]["src"].endswith("_s0")])
                M, b = fit(X, Y, "lin")
                sc = [score(u[1][key] @ M + b, pools[v][key][nn(u[1]["Ec"], pools[v]["Ec"])]) for u in cv if u[0]["spk"] in B and u[0]["valid"] for v in u[4]]
                si = [score(u[1][key], pools[v][key][nn(u[1]["Ec"], pools[v]["Ec"])]) for u in cv if u[0]["spk"] in B and u[0]["valid"] for v in u[4]]
                com[key] = {"r2_common": round(float(np.median([x[0] for x in sc])), 3), "cos_common": round(float(np.median([x[1] for x in sc])), 3), "r2_id": round(float(np.median([x[0] for x in si])), 3),
                            "cos_id": round(float(np.median([x[1] for x in si])), 3), "size": round(float(torch.linalg.norm(M - torch.eye(M.shape[0], device=M.device)) / M.shape[0] ** 0.5), 3), "n_fit_frames": int(len(X))}
            res["common(devA の半分で当てはめ・残りで測る)"] = com
            P15 = json.loads((ROOT / "results/snv_p15/rows.json").read_text())["rows"]
            idx = {it["spk"]: it["idx"] for it in sets["cv"]}
            cross: dict = {}
            for r in P15["base4"]:
                if r["k"] == 7 and r["tgt"] in (tg4[idx[r["spk"]] % 4], tg4[(idx[r["spk"]] + 2) % 4]):
                    cross[(r["src"], r["tgt"])] = r["cer"]
            slf = {r["src"]: r["cer"] for r in P15["self"]}
            R = [r | {"cer": cross[(r["src"], r["tgt"])], "gap": cross[(r["src"], r["tgt"])] - slf[r["src"]]} for r in rows if r["set"] == "cv" and r["valid"] and (r["src"], r["tgt"]) in cross]
            sk = sorted({r["spk"] for r in R})
            ag = lambda k: np.array([np.mean([r[k] for r in R if r["spk"] == sp]) for sp in sk])
            res["rho_devA_speakers"] = {"n_spk": len(sk)} | {k: {"cross": round(float(spearmanr(ag(k), ag("cer"))[0]), 3), "gap": round(float(spearmanr(ag(k), ag("gap"))[0]), 3)} for k in keys}
        res["rows"] = rows
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        return 0

    @torch.no_grad()
    def convert(u: dict, st: tuple, S: dict, M: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        y, v = tt(u, "y"), tt(u, "v")
        lf = tt(u, "lf") + K * T.LN2_12 * v
        a = Pl.proj(Pl.white(y, v, st).transpose(1, 2) @ M + b).transpose(1, 2)
        yo = s.C(a, lf, v, S)
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"] * 2 ** (K / 12))[None].to(DEV)
        return s.gen(cond, EV.TR.excitation(f0, cond.shape[-1] * EV.N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[EV.N.DELAY:].astype(np.float32)

    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p18/prereg.yaml(補足 1)", "sys": tag, "k": K, "rows": {}, "summary": {}}
    for nm in names:
        if nm in res["rows"] and opt.get("redo") != "1":
            continue
        R = []
        for (sn, _), (it, src, enr, st, tgs) in units.items():
            if sn != nm:
                continue
            for v in tgs:
                S = s.structure(f"t|{v}|3", rig.trefs[v])
                jf = nn(enr["Ec"], pools[v]["Ec"])
                for kind in ("id", "orth", "lin"):
                    M, b = fit(enr["W"], pools[v]["W"][jf], kind)
                    R.append(rig.row(it, convert(it["u"], st, S, M, b), mode=kind, tgt=v, k=K))
        res["rows"][nm] = R
        res["summary"][nm] = {kind: (lambda X: {"n": len(X), "cer_median": round(float(np.median([r["cer"] for r in X])), 4), "cer_mean": round(float(np.mean([r["cer"] for r in X])), 4),
                                                "M": round(float(np.mean([r["top"] == r["tgt"] for r in X])), 3)})([r for r in R if r["mode"] == kind and r["valid"]]) for kind in ("id", "orth", "lin")}
        print(nm, json.dumps(res["summary"][nm], ensure_ascii=False), flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
