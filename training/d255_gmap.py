"""D255(学習なし・results/conv_p0/d255_prereg.yaml): 表の単位ごとの引きを使わない全体写像(--global-map)の変換の目標関数 v1 と本人らしさ。
--set devff(FF 60・α 1)/ dev(MF 60・TTS 男声・α 1.3)。
    uv run python d255_gmap.py --s <dev の親> --set devff --alpha 1 --out ../results/conv_p0/d255_ff.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target_dev/release/lightvc-app"
COMMON = ["--mix", "0", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--warp", "--causal-ct", "--global-map"]
RUNGS = {"GM": ["--rvoc", "results/rvoc3hi/export", "FIXED"], "GMF": ["--rvoc", "results/rvoc3hi/export"], "GMX": ["--rvoc", "results/rvtab_x/export", "FIXED"]}


def job(cmd: list) -> None:
    if not Path(cmd[-1]).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--set", default="devff", choices=("devff", "dev"))
    ap.add_argument("--alpha", default="1")
    a = ap.parse_args()
    S = Path(a.s)
    F = S / a.set
    od = F / "d255"
    od.mkdir(exist_ok=True)
    srcs = sorted(f for f in (F / "d56").glob("p00*_SRC.wav"))
    jobs = []
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        for r, opt in RUNGS.items():
            jobs.append([BIN, "artic", "--enroll", F / "d85_work" / f"en_{ti:03d}", "--operator-enroll", F / "d85_work" / f"op_{pi:04d}"] + COMMON + [o for o in opt if o != "FIXED"] + (["--alpha", a.alpha] if "FIXED" in opt else []) + ["-i", f, "-o", od / f"{f.stem[:-4]}_{r}.wav"])
    with Pool(a.workers) as pl:
        list(pl.imap_unordered(job, jobs))
    print("rendered", len(jobs), flush=True)
    groups = {r: sorted(od.glob(f"p00*_{r}.wav")) for r in RUNGS}
    groups["PROD"] = sorted((F / "d114").glob("p00*_G03S10ZH.wav"))
    if a.set == "dev":
        groups["RRQ"] = sorted((F / "d139").glob("p00*_RRQSDC19LE.wav"))
    groups["KNN"] = sorted((F / "d135_real").glob("p00*_KNN.wav"))
    W = F / "ldaD255"
    W.mkdir(exist_ok=True)
    for f in sorted((S / "dev/cen_reg").glob("cen_*.wav")):
        if not (W / f.name).exists():
            os.symlink(f, W / f.name)
    for g, files in groups.items():
        for f in files:
            dst = W / (f.name if g not in ("PROD", "KNN") else f.name)
            if not dst.exists():
                os.symlink(f, dst)
    env = os.environ | {"OMP_NUM_THREADS": "2"}
    npz = F / "d255_lda.npz"
    tag = f"d255{a.set}"
    subprocess.run(["uv", "run", "python", "d4b_lda_verifier.py", "--wav_dir", str(W), "--out", str(ROOT / f"results/conv_p0/{tag}_lda.json"), "--dump", str(npz)], cwd=ROOT / "training", env=env, check=True, capture_output=True)
    subprocess.run(["uv", "run", "python", "d146_identity_triangulate.py", "--dir", str(W), "--tag", f"{tag}_tri"], cwd=ROOT / "training", env=env, check=True, capture_output=True)
    print("identity measured", flush=True)
    import d181_verdict as V
    from rr17c_verdict import ci
    Rk = V.ranks(npz, ROOT / f"results/conv_p0/{tag}_tri.json")
    ok = {m["ti"] for m in json.loads((S / "dev/cen_reg/meta.json").read_text()) if m["ok"]}
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    base = ROOT / "female-dataset"
    R = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((base / r["id"]).glob("*.wav"))
            R.append(emb(ws[len(ws) // 2]))
    R = np.array(R)
    rf = np.arange(len(R)) % 5
    rep: dict = {}
    for g, files in groups.items():
        X = np.array([emb(f) for f in files])
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        Xa = np.concatenate([R, X])
        ya = np.r_[np.zeros(len(R)), np.ones(len(X))]
        fa = np.r_[rf, tg % 5]
        pr = np.zeros(len(ya))
        for k in range(5):
            te = fa == k
            sc = StandardScaler().fit(Xa[~te])
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
            pr[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
        c = {"PROD": "G03S10ZH", "KNN": "KNN", "RRQ": "RRQSDC19LE"}.get(g, g)
        rep[g] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "p_fake_median": round(float(np.median(pr[len(R):])), 4), "n": len(X)}
        for m in ("LDA", "ECAPA"):
            lv = [Rk[m][c][k] for k in Rk[m].get(c, {}) if int(k.split("_t")[1]) in ok]
            rep[g][f"{m}_mean_rank"] = round(float(np.mean(lv)), 2) if lv else None
            if g != "KNN" and c in Rk[m]:
                rows = [("ff", int(k.split("_t")[1]), Rk[m][c][k] - Rk[m]["KNN"][k]) for k in sorted(set(Rk[m][c]) & set(Rk[m]["KNN"])) if int(k.split("_t")[1]) in ok]
                rep[g][f"{m}_minus_KNN"] = ci(rows, np.random.default_rng(0))
        print(g, rep[g], flush=True)
    idok = lambda g: all(rep[g][f"{m}_minus_KNN"]["rank_diff"] <= 0 for m in ("LDA", "ECAPA"))
    rep["id_ok"] = [g for g in RUNGS if idok(g)]
    print("id_ok", rep["id_ok"], {g: rep[g]["p_fake_median"] for g in groups}, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
