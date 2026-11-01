"""D242(学習なし・results/conv_p0/d242_prereg.yaml): 自己変換(AA・devff の p0100〜)で製品の経路を段ごとに外し、目標関数 v1(本物の女声と区別できないこと・D95 の AUC と p_fake)で壊す段を特定する。
    uv run python d242_aa_ladder.py --s <dev の親> --out ../results/conv_p0/d242.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
COMMON = ["--mix", "0", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--f0-conf", "0.5", "--warp"]
RUNGS = {"C3HI": ["--rvoc", "results/rvoc3hi/export", "--alpha", "1", "--resid", "1"],
         "C0": ["--rvoc", "results/rvtab_x/export", "--alpha", "1", "--resid", "1"],
         "G07": ["--rvoc", "results/rvtab_x/export", "--alpha", "1", "--resid", "0.7"],
         "G03": ["--rvoc", "results/rvtab_x/export", "--alpha", "1", "--resid", "0.3"],
         "G00": ["--rvoc", "results/rvtab_x/export", "--alpha", "1", "--resid", "0"],
         "W13": ["--rvoc", "results/rvtab_x/export", "--alpha", "1.3", "--resid", "0.3", "--color"]}


def job(cmd: list) -> None:
    if not Path(cmd[-1]).exists():
        subprocess.run([str(c) for c in cmd], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S = Path(a.s)
    F = S / "devff"
    od = F / "d242"
    od.mkdir(exist_ok=True)
    srcs = sorted(f for f in (F / "d56").glob("p01*_SRC.wav"))
    jobs = []
    for f in srcs:
        ti = int(f.name[7:10])
        en = F / "d85_work" / f"en_{ti:03d}"
        for r, opt in RUNGS.items():
            jobs.append([BIN, "artic", "--enroll", en, "--operator-enroll", en] + COMMON + opt + ["-i", f, "-o", od / f"{f.stem[:-4]}_{r}.wav"])
    with Pool(a.workers) as pl:
        list(pl.imap_unordered(job, jobs))
    print("rendered", len(jobs), flush=True)
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
    groups = {r: sorted(od.glob(f"p01*_{r}.wav")) for r in RUNGS}
    groups["PROD"] = sorted((F / "d114").glob("p01*_G03S10ZH.wav"))
    groups["SRC"] = srcs
    rep: dict = {}
    rf = np.arange(len(R)) % 5
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
        rep[g] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "p_fake_median": round(float(np.median(pr[len(R):])), 4), "n": len(X)}
        print(g, rep[g], flush=True)
    A = lambda g: rep[g]["auc"]
    rep["verdict"] = ("RENDERER" if A("C3HI") <= 0.95 and A("C0") > 0.95 else "FRONTEND" if A("C3HI") > 0.95 else "TABLE" if A("C0") <= 0.95 and A("G03") > 0.95
                      else "WARP_CAL" if A("G03") <= 0.95 and (A("W13") > 0.95 or A("PROD") > 0.95) else "NONE_FAIL")
    print("verdict", rep["verdict"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
