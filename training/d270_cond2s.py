"""D270(学習なし・results/conv_p0/d270_prereg.yaml): 出力部に入る条件(31 行)の二標本。本物の女声の条件(devff の実女声の入力の写し合成の条件)対 男声 → 女声(dev)/ 女声 → 女声(devff)の変換の条件。
    uv run python d270_cond2s.py --s <dev の親> --out ../results/conv_p0/d270.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
COMMON = ["--causal-ct", "--rvoc", "results/rvoc3hi/export", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export"]
CONV = ["--floor-apply", "--warp", "--resid", "0.7", "--color", "--mix", "0"]
BUNDLES = {"c0": [0], "c1_4": list(range(1, 5)), "c5_12": list(range(5, 13)), "c13_24": list(range(13, 25)), "lf0": [25], "per": list(range(27, 31))}


def job(j: tuple) -> None:
    cmd, dump = j
    if not Path(dump).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_COND": str(dump)}, check=True, capture_output=True)


def feats(c: np.ndarray) -> dict:
    v = c[:, 26] > 0.5
    out = {}
    for name, rows in BUNDLES.items():
        f = []
        for r in rows:
            x = c[v, r] if name in ("lf0", "per") or v.sum() > 20 else c[:, r]
            dx = np.abs(np.diff(x)) if len(x) > 2 else np.zeros(1)
            f += [x.mean(), x.std(), dx.mean()]
            if name not in ("lf0", "per"):
                xa = c[:, r]
                f += [xa.mean(), xa.std(), np.abs(np.diff(xa)).mean()]
        out[name] = np.array(f)
    out["vfrac"] = np.array([v.mean()])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    S = Path(a.s)
    od = S / "d270"
    od.mkdir(exist_ok=True)
    jobs, sets = [], {"REAL": [], "MF": [], "FF": []}
    F, D = S / "devff", S / "dev"
    for f in sorted((F / "d56").glob("p00*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        op = F / "d85_work" / f"op_{pi:04d}"
        d = od / f"REAL_{f.stem}.f32"
        jobs.append(([BIN, "artic", "--enroll", op, "--operator-enroll", op, "--mix", "1"] + COMMON + ["-i", f, "-o", od / "_tmp_r.wav"], d)); sets["REAL"].append((d, ti))
    for name, base in (("MF", D), ("FF", F)):
        for f in sorted((base / "d56").glob("p00*_SRC.wav"))[: a.n]:
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            al = "1.3" if name == "MF" else "1.0"
            d = od / f"{name}_{f.stem}.f32"
            jobs.append(([BIN, "artic", "--enroll", base / "d85_work" / f"en_{ti:03d}", "--operator-enroll", base / "d85_work" / f"op_{pi:04d}"] + COMMON + CONV + ["--alpha", al, "-i", f, "-o", od / f"_tmp_{name}.wav"], d)); sets[name].append((d, ti))
    jobs = [(c[:-1] + [str(od / f"_o_{Path(dd).stem}.wav")], dd) for c, dd in jobs]
    with Pool(10) as pl:
        list(pl.imap_unordered(job, jobs))
    print("dumped", len(jobs), flush=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Fe = {k: [(feats(np.fromfile(d, "<f4").reshape(-1, 31)), ti) for d, ti in v] for k, v in sets.items()}
    rep: dict = {}
    for cand in ("MF", "FF"):
        rep[cand] = {}
        for bname in list(BUNDLES) + ["vfrac", "ALL"]:
            get = (lambda f: np.concatenate([f[b] for b in list(BUNDLES) + ["vfrac"]])) if bname == "ALL" else (lambda f, b=bname: f[b])
            Xr = np.array([get(f) for f, _ in Fe["REAL"]])
            Xc = np.array([get(f) for f, _ in Fe[cand]])
            tg = np.array([ti for _, ti in Fe[cand]])
            Xa = np.concatenate([Xr, Xc])
            ya = np.r_[np.zeros(len(Xr)), np.ones(len(Xc))]
            fa = np.r_[np.arange(len(Xr)) % 5, tg % 5]
            pr = np.zeros(len(ya))
            for k in range(5):
                te = fa == k
                sc = StandardScaler().fit(Xa[~te])
                c2 = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xa[~te]), ya[~te])
                pr[te] = c2.predict_proba(sc.transform(Xa[te]))[:, 1]
            rep[cand][bname] = round(float(roc_auc_score(ya, pr)), 4)
        print(cand, rep[cand], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
