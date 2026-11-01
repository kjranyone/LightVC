"""D239(学習なし・results/conv_p0/d239_prereg.yaml): 出力の局所の質感の分布と目標本人の別の録音のそれとの MMD² は、変換の耳の不合格を再現するか。
パッチ = D238 と同じ 200 次元 → D238 の PCA 60(本物の主成分の分散で白色化)・有音フレームから各 300 個・ガウス核(幅 = 本物のパッチの対の距離の中央値)・不偏推定。
    uv run python d239_mmd_ear.py --s <dev の親> --out ../results/conv_p0/d239.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import d238_local_anom as A

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
NP = 300
HALF = 4 * A.SR
FAIL_GROUPS = ("FAIL_cand_ab1", "FAIL_prodcand", "FAIL_convg1", "FAIL_conv_ab1", "dev:G03S10ZH", "dev:RRQSDC19LE")
EARMAP = {"cand_ab1": {"cand": "FAIL_cand_ab1", "base": "FAIL_cand_ab1", "knn": "KNN_ear"},
          "prodcand_ab1": {"cand_gamma05_color_ft360k": "FAIL_prodcand", "cand_gamma07_color_ft360k": "FAIL_prodcand"},
          "convg1_ab1": {"table_only": "FAIL_convg1", "warp_gamma07": "FAIL_convg1", "warp_gamma10": "FAIL_convg1"},
          "conv_ab1": {"same": "FAIL_conv_ab1", "other": "FAIL_conv_ab1"}}
DEVF = {"KNN": ("d135_real", "KNN"), "G03S10ZH": ("d114", "G03S10ZH"), "RRQSDC19LE": ("d139", "RRQSDC19LE")}
_M: dict = {}


def patches(spec: tuple) -> np.ndarray:
    path, part = spec
    if not _M:
        m = pickle.loads((ROOT / "results/conv_p0/d238.model.pkl").read_bytes())
        _M["pca"] = m["pca"]
        _M["sd"] = np.sqrt(m["pca"].explained_variance_)
    X, act = A.feats(path) if part == "all" else feats_part(path, part)
    if act.sum() == 0:
        return np.zeros((0, 60))
    Z = _M["pca"].transform(X[act].astype(np.float64)) / _M["sd"]
    return Z


def feats_part(path: str, part: str) -> tuple:
    import soundfile as sf
    import tempfile
    x = A.load48(path)
    seg = x[:HALF] if part == "h1" else x[HALF:2 * HALF]
    if len(seg) < A.SR:
        return np.zeros((0, 200), np.float32), np.zeros(0, bool)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, seg, A.SR)
        return A.feats(f.name)


def mmd2(X: np.ndarray, Y: np.ndarray, bw: float) -> float:
    def k(a, b):
        d = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T
        return np.exp(-np.maximum(d, 0) / (2 * bw ** 2))
    Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
    n, m = len(X), len(Y)
    return float((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2 * Kxy.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    S = Path(a.s)
    base = ROOT / "female-dataset"
    d237 = json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]
    pairs = []
    for r in d237:
        if r["grp"] != "REAL_POOL":
            continue
        ws = sorted((base / r["id"]).glob("*.wav"))
        i = len(ws) // 2
        pairs.append(("REAL_POOL", r["id"], (str(ws[i]), "all"), (str(ws[(i + 1) % len(ws)]), "all"), ""))
        pairs.append(("REAL_SPLIT", r["id"], (str(ws[i]), "h1"), (str(ws[i]), "h2"), ""))
    for d, mp in EARMAP.items():
        key = json.loads((EB / d / "_key_聴取後に開く.json").read_text())["trials"]
        for t, v in key.items():
            if d == "cand_ab1" and t not in ("cand_01", "cand_02", "cand_03"):
                continue
            for L, form in v["map"].items():
                if form in mp:
                    pairs.append((mp[form], f"{d}/{t}/{L}", (str(EB / d / t / f"{L}.wav"), "all"), (str(EB / d / t / "R.wav"), "all"), form))
    key = json.loads((EB / "rvoc_ab3/_key_聴取後に開く.json").read_text())["trials"]
    for t, v in key.items():
        src = [L for L, f in v["map"].items() if f == "source"][0]
        for L, form in v["map"].items():
            g = {"rvoc": "PASS_rvoc", "bigvgan_v2": "PASS_bigvgan", "source": "EAR_REAL_SPLIT"}[form]
            pairs.append((g, f"rvoc_ab3/{t}/{L}", (str(EB / "rvoc_ab3" / t / f"{L}.wav"), "h1"), (str(EB / "rvoc_ab3" / t / f"{src}.wav"), "h2"), form))
    ok = {m["ti"] for m in json.loads((S / "dev/cen_reg/meta.json").read_text()) if m["ok"]}
    for ti in sorted(ok):
        pairs.append(("DEV_REAL", f"dev/ref_{ti:03d}", (str(S / "dev/d37/_work" / f"ref_{ti:03d}.wav"), "all"), (str(S / "dev/cen_reg" / f"cen_{ti:03d}.wav"), "all"), "REF"))
    for nm, (dd, c) in DEVF.items():
        for f in sorted((S / "dev" / dd).glob(f"*_{c}.wav")):
            ti = int(f.name.split("_t")[1][:3])
            if ti in ok:
                pairs.append((f"dev:{nm}", f"dev/{f.name}", (str(f), "all"), (str(S / "dev/cen_reg" / f"cen_{ti:03d}.wav"), "all"), nm))
    specs = sorted({p[2] for p in pairs} | {p[3] for p in pairs})
    with Pool(a.workers) as pl:
        Z = dict(zip(specs, pl.map(patches, specs)))
    print("patches", len(specs), flush=True)
    rng = np.random.default_rng(0)
    pick = lambda z: z[rng.choice(len(z), min(NP, len(z)), replace=False)]
    real = np.concatenate([pick(Z[p[2]]) for p in pairs if p[0] == "REAL_POOL"])
    sub = real[rng.choice(len(real), 2000, replace=False)]
    d = np.sqrt(np.maximum((sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None] - 2 * sub @ sub.T, 0))
    bw = float(np.median(d[np.triu_indices(len(sub), 1)]))
    rows = []
    for g, i, ys, xs, form in pairs:
        if len(Z[ys]) < 50 or len(Z[xs]) < 50:
            continue
        Y, X = pick(Z[ys]), pick(Z[xs])
        rows.append({"grp": g, "id": i, "form": form, "mmd": mmd2(Y, X, bw), "spread": float(np.trace(np.cov(Z[ys].T)) / np.trace(np.cov(Z[xs].T))), "ny": len(Z[ys]), "nx": len(Z[xs])})
    rep: dict = {"bw": bw}
    vals = lambda g, m: np.array([r[m] for r in rows if r["grp"] == g], float)
    rep["real_pool_p15_p85"] = [round(float(np.percentile(vals("REAL_POOL", "mmd"), q)), 5) for q in (15, 85)]
    rep["real_split_p15_p85"] = [round(float(np.percentile(vals("REAL_SPLIT", "mmd"), q)), 5) for q in (15, 85)]
    grps = sorted({r["grp"] for r in rows})
    rep["medians"] = {g: {"mmd": round(float(np.median(vals(g, "mmd"))), 5), "spread": round(float(np.median(vals(g, "spread"))), 3), "n": len(vals(g, "mmd"))} for g in grps}
    pass_ok = all(rep["medians"][g]["mmd"] <= rep["real_split_p15_p85"][1] for g in ("PASS_rvoc", "PASS_bigvgan"))
    det = {g: rep["medians"][g]["mmd"] > rep["real_pool_p15_p85"][1] for g in FAIL_GROUPS}
    rep["axis"] = {"pass_ok": pass_ok, "fail_detected": det, "axis_ok": pass_ok and all(det.values())}
    rep["trial1"] = {r["form"]: round(r["mmd"], 5) for r in rows if r["id"].startswith("cand_ab1/cand_01/")}
    kg = {r["id"].split("_", 1)[0] + r["id"].split("_t")[1][:3]: r["mmd"] for r in rows if r["grp"] == "dev:KNN"}
    gg = {r["id"].split("_", 1)[0] + r["id"].split("_t")[1][:3]: r["mmd"] for r in rows if r["grp"] == "dev:G03S10ZH"}
    dd_ = [kg[k] - gg[k] for k in sorted(set(kg) & set(gg))]
    rep["dev_knn_minus_base"] = {"mean": round(float(np.mean(dd_)), 5), "frac_neg": round(float(np.mean(np.array(dd_) < 0)), 3), "n": len(dd_)}
    rep["verdict"] = "MMD_EXPLAINS" if rep["axis"]["axis_ok"] else "NONE"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep | {"rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
