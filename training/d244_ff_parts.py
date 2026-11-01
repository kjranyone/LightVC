"""D244(学習なし・results/conv_p0/d244_prereg.yaml): 女声 → 女声の偽物らしさ(目標関数 v1)の出どころの分解。OSELF(操作者の写し合成)・MIX1(包絡は操作者のまま・高さと周期性だけ目標へ)・TABM0(操作者の包絡 + 単位ごとの話者差)。
    uv run python d244_ff_parts.py --s <dev の親> --out ../results/conv_p0/d244.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d242_aa_ladder import job

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
COMMON = ["--rvoc", "results/rvtab_x/export", "--per-p75", "--med-from-operator", "--f4", "results/f4_1/export", "--f0-conf", "0.5"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    S = Path(a.s)
    F = S / "devff"
    od = F / "d244"
    od.mkdir(exist_ok=True)
    srcs = sorted(F / "d56" / f.name for f in (F / "d56").glob("p00*_SRC.wav"))
    jobs = []
    for f in srcs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        en, op = F / "d85_work" / f"en_{ti:03d}", F / "d85_work" / f"op_{pi:04d}"
        st = f.stem[:-4]
        jobs.append([BIN, "artic", "--enroll", op, "--operator-enroll", op, "--mix", "1"] + COMMON + ["-i", f, "-o", od / f"{st}_OSELF.wav"])
        jobs.append([BIN, "artic", "--enroll", en, "--operator-enroll", op, "--mix", "1"] + COMMON + ["-i", f, "-o", od / f"{st}_MIX1.wav"])
        jobs.append([BIN, "artic", "--enroll", en, "--operator-enroll", op, "--mix", "0"] + COMMON + ["-i", f, "-o", od / f"{st}_TABM0.wav"])
    with Pool(a.workers) as pl:
        list(pl.imap_unordered(job, jobs))
    print("rendered", len(jobs), flush=True)
    import torch
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
    R = np.array([emb(sorted((base / r["id"]).glob("*.wav"))[len(list((base / r["id"]).glob("*.wav"))) // 2])
                  for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"] if r["grp"] == "REAL_POOL"])
    rf = np.arange(len(R)) % 5
    groups = {r: sorted(od.glob(f"p00*_{r}.wav")) for r in ("OSELF", "MIX1", "TABM0")}
    groups["FG10"] = sorted((F / "d243").glob("p00*_FG10.wav"))
    groups["SRC"] = srcs
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
        rep[g] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "p_fake_median": round(float(np.median(pr[len(R):])), 4), "n": len(X)}
        print(g, rep[g], flush=True)
    p = lambda g: rep[g]["p_fake_median"]
    rep["verdict"] = ("FLOOR" if p("OSELF") >= 0.8 else "TABLE_DIFF" if p("MIX1") <= p("OSELF") + 0.15 and p("TABM0") >= p("MIX1") + 0.2
                      else "PITCH_PER" if p("MIX1") >= p("OSELF") + 0.2 else "MIXED")
    print("verdict", rep["verdict"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
