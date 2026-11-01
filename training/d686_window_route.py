"""D686: 男声の漏れが生成器の入力のどの流れの窓の統計に乗るか(誤差の分析・推論だけ)。官能指数 v1(gen を外した fold)の窓ごとの値と、LIGHTVC_DUMP_COND の cond_ppg(295 次元 / 5 ms)から作る窓ごとの特徴を並べる。
規則は results/conv_p0/d686/prereg.yaml。描画は results/conv_p0/d686/run.sh。
    uv run python d686_window_route.py --stage check | margin | fit
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from d678_kansei_det import HOPW, OUT as KOUT, WIN, embed, wavlm

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = ROOT / "results/conv_p0/d686"
DUMP = H / "d686_dump"
D_ALL, D_COND, K = 295, 31, 200
ARMS = [("d487_uiro", 12, "RUST12", 1), ("d487_uiro", 5, "RUST5Q3", 1), ("devff2", 0, "RUST0", 0), ("devff2", -5, "RUST-5", 0), ("devff2", -12, "RUST-12", 0)]
SRC = [("d487_uiro", 12, 1), ("devff2", 0, 0)]
FEATS = ["lf0", "vfrac", "vuv_sw", "jit", "jump", "f0_rng", "c0_mean", "c0_sd", "c0_d", "q_rate", "q_ent"]
CENT = 1200.0 / np.log(2.0)


def keys(s: str) -> list:
    return [(r["pi"], r["ti"], f"p{r['pi']:04d}_t{r['ti']:03d}") for r in json.loads((H / s / "devset.json").read_text())["pairs"]]


def wins(path: Path) -> tuple:
    import soundfile as sf
    import soxr
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = soxr.resample(x.mean(1), sr, 16000)[: 8 * 16000]
    n = 1 + max(0, (len(x) - WIN) // HOPW)
    W = np.stack([np.pad(x[i * HOPW:i * HOPW + WIN], (0, max(0, WIN - len(x[i * HOPW:i * HOPW + WIN])))) for i in range(n)])
    e = 10 * np.log10((W ** 2).mean(1) + 1e-12)
    keep = np.where(e > e.max() - 30)[0]
    W = W[keep]
    W = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-7)
    return W.astype(np.float32), keep


def wfeat(C: np.ndarray, i: int) -> dict:
    f0 = 2400 + i * 50
    c = C[f0:f0 + 100]
    lf, vo = c[:, 25].astype(np.float64), c[:, 26] > 0.5
    r: dict = {"vfrac": float(vo.mean()), "vuv_sw": float(np.abs(np.diff(vo.astype(int))).sum())}
    if vo.sum() >= 5:
        r["lf0"] = float(np.median(lf[vo]))
        r["f0_rng"] = float((np.percentile(lf[vo], 90) - np.percentile(lf[vo], 10)) * CENT)
    both = vo[1:] & vo[:-1]
    if both.sum() >= 4:
        d = np.abs(np.diff(lf))[both] * CENT
        r["jit"] = float(np.median(d))
        r["jump"] = float(np.mean(d > 100))
    c0 = c[:, 0].astype(np.float64)
    r |= {"c0_mean": float(c0.mean()), "c0_sd": float(c0.std()), "c0_d": float(np.abs(np.diff(c0)).mean())}
    q = c[:, D_COND:D_COND + K].astype(np.float64)
    am = q.argmax(1)
    r["q_rate"] = float((np.diff(am) != 0).sum())
    r["q_ent"] = float(np.mean(-(q * np.log(q + 1e-12)).sum(1)))
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["check", "margin", "fit"])
    a = ap.parse_args()
    if a.stage == "check":
        import soundfile as sf
        rep = {}
        for s, k, v, _ in ARMS:
            n_eq = n = nd = 0
            for pi, ti, key in keys(s):
                p0, p1 = H / s / "gate" / v / f"{key}_{v}.wav", H / s / "gate" / f"{v}DMP" / f"{key}_{v}DMP.wav"
                if not (p0.exists() and p1.exists()):
                    continue
                n += 1
                n_eq += int(np.array_equal(sf.read(p0, dtype="float32")[0], sf.read(p1, dtype="float32")[0]))
                nd += int((DUMP / f"{s}_{k}" / f"{pi}_{ti}.f32").stat().st_size % (4 * D_ALL) == 0)
            rep[v] = {"n": n, "bit_equal": n_eq, "dump_ok": nd}
            print(s, v, rep[v], flush=True)
        (OUT / "check.json").write_text(json.dumps(rep, indent=1))
        return 0
    if a.stage == "margin":
        m = pickle.loads((KOUT / "model_gen.pkl").read_bytes())
        sc, clf, tau = m["scaler"], m["clf"], m["tau"]
        net = wavlm()
        rows = []
        for s, k, v, male in ARMS:
            for pi, ti, key in keys(s):
                W, keep = wins(H / s / "gate" / v / f"{key}_{v}.wav")
                if len(W) == 0:
                    continue
                dv = clf.decision_function(sc.transform(embed(net, W).astype(np.float32))) - tau
                C = np.fromfile(DUMP / f"{s}_{k}" / f"{pi}_{ti}.f32", "<f4").reshape(-1, D_ALL)
                for j, i in enumerate(keep):
                    rows.append({"set": s, "k": k, "arm": v, "male": male, "utt": f"{s}|{k}|{key}", "win": int(i), "m": float(dv[j]), "utt_m": float(dv.mean())} | wfeat(C, int(i)))
            print(v, len(rows), flush=True)
        for s, k, male in SRC:
            for pi, ti, key in keys(s):
                W, keep = wins(H / s / "d56" / f"{key}_SRC.wav")
                if len(W) == 0:
                    continue
                dv = clf.decision_function(sc.transform(embed(net, W).astype(np.float32))) - tau
                C = np.fromfile(DUMP / f"{s}_{k}" / f"{pi}_{ti}.f32", "<f4").reshape(-1, D_ALL)
                for j, i in enumerate(keep):
                    f = wfeat(C, int(i))
                    if "lf0" in f:
                        f["lf0"] -= k / 12 * np.log(2.0)
                    rows.append({"set": s, "k": k, "arm": "SRC", "male": male, "utt": f"{s}|SRC|{key}", "win": int(i), "m": float(dv[j]), "utt_m": float(dv.mean())} | f)
            print("SRC", s, len(rows), flush=True)
        (OUT / "windows.json").write_text(json.dumps(rows))
        return 0
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr
    allrows = json.loads((OUT / "windows.json").read_text())
    rep: dict = {"spearman_within_utt": {}}
    for nm, sel in (("src_male", lambda r: r["arm"] == "SRC" and r["male"] == 1), ("src_female", lambda r: r["arm"] == "SRC" and r["male"] == 0),
                    ("conv_male", lambda r: r["arm"] != "SRC" and r["male"] == 1), ("conv_female", lambda r: r["arm"] != "SRC" and r["male"] == 0)):
        g = [r for r in allrows if sel(r)]
        um = {}
        for r in g:
            um.setdefault(r["utt"], []).append(r["m"])
        um = {u: float(np.mean(v)) for u, v in um.items()}
        rho = {}
        for f in FEATS:
            gg = [r for r in g if np.isfinite(r.get(f, np.nan))]
            fm = {}
            for r in gg:
                fm.setdefault(r["utt"], []).append(r[f])
            fm = {u: float(np.mean(v)) for u, v in fm.items()}
            rho[f] = round(float(spearmanr([r["m"] - um[r["utt"]] for r in gg], [r[f] - fm[r["utt"]] for r in gg])[0]), 3) if len(gg) > 20 else None
        rep["spearman_within_utt"][nm] = {"n": len(g), "median_m": round(float(np.median([r["m"] for r in g])), 2)} | rho
    rows = [r for r in allrows if r["arm"] != "SRC"]
    rep["n_windows"] = len(rows)
    loc = {}
    for v in sorted({r["arm"] for r in rows}):
        g = [r for r in rows if r["arm"] == v]
        per = {}
        for r in g:
            per.setdefault(r["utt"], []).append(r["m"])
        top = []
        for ms in per.values():
            pos = np.sort(np.maximum(np.array(ms), 0))[::-1]
            if pos.sum() > 0:
                top.append(pos[: max(1, int(np.ceil(0.25 * len(pos))))].sum() / pos.sum())
        loc[v] = {"n_win": len(g), "frac_pos": round(float(np.mean([r["m"] > 0 for r in g])), 3), "top25_share": round(float(np.median(top)), 3) if top else None,
                  "median_m": round(float(np.median([r["m"] for r in g])), 2), "within_utt_sd": round(float(np.median([np.std(x) for x in per.values()])), 2)}
        loc[v] |= {f: round(float(np.nanmedian([r.get(f, np.nan) for r in g])), 3) for f in FEATS}
    rep["by_arm"] = loc
    ok = [r for r in rows if r["vfrac"] >= 0.2 and all(np.isfinite(r.get(f, np.nan)) for f in FEATS)]
    rep["n_reg"] = len(ok)
    y = np.array([r["m"] for r in ok])
    male = np.array([r["male"] for r in ok], float)
    X = np.array([[r[f] for f in FEATS] for r in ok])
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    grp = np.array([r["utt"] for r in ok])
    col = {f: Z[:, i] for i, f in enumerate(FEATS)}

    def beta(extra: list) -> float:
        A = np.column_stack([np.ones(len(y)), male, col["lf0"]] + [col[f] for f in extra])
        return float(np.linalg.lstsq(A, y, rcond=None)[0][1])
    b0 = beta([])
    rep["beta_male_base"] = round(b0, 3)
    rep["shrink_single"] = {f: round(1 - beta([f]) / b0, 3) for f in FEATS if f != "lf0"}
    rep["shrink_all"] = round(1 - beta([f for f in FEATS if f != "lf0"]) / b0, 3)
    rep["shrink_drop_one"] = {f: round(1 - beta([g for g in FEATS if g not in ("lf0", f)]) / b0, 3) for f in FEATS if f != "lf0"}
    gap = float(y[male == 1].mean() - y[male == 0].mean())

    def gb(cols: list) -> float:
        P = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(Z, y, grp):
            P[te] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=686).fit(X[tr][:, cols], y[tr]).predict(X[te][:, cols])
        return float(P[male == 1].mean() - P[male == 0].mean())
    gp, ga = gb([FEATS.index("lf0")]), gb(list(range(len(FEATS))))
    rep["gb"] = {"gap_actual": round(gap, 3), "gap_pitch_only": round(gp, 3), "gap_all": round(ga, 3), "explained_beyond_pitch": round((ga - gp) / (gap - gp), 3)}
    rep["gb_single"] = {f: round((gb([FEATS.index("lf0"), FEATS.index(f)]) - gp) / (gap - gp), 3) for f in FEATS if f != "lf0"}
    mx = max(rep["shrink_single"].values())
    rep["verdict"] = "route" if mx >= 0.5 else ("distributed" if rep["shrink_all"] >= 0.5 else "unexplained")
    rep["route_feature"] = max(rep["shrink_single"], key=rep["shrink_single"].get) if mx >= 0.5 else None
    ml = [loc[v] for v in loc if v.startswith("RUST12") or v.startswith("RUST5")]
    rep["localization"] = ["global" if x["frac_pos"] >= 0.8 and (x["top25_share"] or 0) < 0.45 else ("local" if (x["top25_share"] or 0) >= 0.6 else "mixed") for x in ml]
    (OUT / "fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
