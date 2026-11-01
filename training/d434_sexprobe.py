"""D434(results/conv_p0/d434_prereg.yaml): 内容をそろえた男女の対(data/pair_tts_v4l)で、C1 の柔らかい事後の男女の情報が声道長・高さの伸縮で動くか。
D276〜D278 は内容の交絡した組(女声 TTS は全員同じ 10 文)だったので測り直す。CPU。
    uv run python d434_sexprobe.py --out ../results/conv_p0/d434.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pair_tts_v4l"
VAR = {1: ("raw", "w1", "wf18", "a115", "a13", "wf18e115"), 0: ("raw", "w1", "wf056e087", "a087")}


def world(x: np.ndarray, kf: float, ke: float) -> np.ndarray:
    import pyworld as pw
    xd = x.astype(np.float64)
    f0, t = pw.dio(xd, 48000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    f0 = pw.stonemask(xd, f0, t, 48000)
    sp = pw.cheaptrick(xd, f0, t, 48000)
    ap = pw.d4c(xd, f0, t, 48000)
    if ke != 1.0:
        nb = sp.shape[1]
        src = np.arange(nb) / ke
        sp = np.stack([np.interp(src, np.arange(nb), r) for r in sp])
        ap = np.stack([np.interp(src, np.arange(nb), r) for r in ap])
    y = pw.synthesize(f0 * kf, sp, ap, 48000, 5.0)[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    return (y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))).astype(np.float32)


def one(job: tuple) -> tuple:
    g, spk, w = job
    import torch
    torch.set_num_threads(1)
    import c1_content as CC
    import train_c1 as T1
    from d6_unit_cov import load48
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"]))
    c1.load_state_dict(st1["net"])
    c1.eval()
    mf = CC.MelFront()
    x = load48(w)
    n = len(x) // 240
    out = {}
    with torch.no_grad():
        for v in VAR[g]:
            al = None
            if v == "raw":
                xv = x
            elif v == "w1":
                xv = world(x, 1.0, 1.0)
            elif v == "wf18":
                xv = world(x, 1.8, 1.0)
            elif v == "wf18e115":
                xv = world(x, 1.8, 1.15)
            elif v == "wf056e087":
                xv = world(x, 0.56, 1 / 1.15)
            else:
                xv, al = x, {"a115": 1.15, "a13": 1.3, "a087": 0.87}[v]
            xt = torch.from_numpy(T1.prime(xv))[None]
            mel = mf(xt, torch.tensor([al])) if al else mf(xt)
            out[v] = c1(mel)[0].T.softmax(1).numpy()[-n:].astype(np.float32)
    return g, spk, w.name, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_txt", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    jobs = []
    for g, pre in ((1, "pm"), (0, "pf")):
        for s in sorted(D.glob(f"{pre}*")):
            for w in sorted(s.glob("t*.wav"))[: a.n_txt]:
                jobs.append((g, s.name, w))
    with Pool(a.workers) as pl:
        data = list(pl.imap_unordered(one, jobs, chunksize=2))
    print("utterances", len(data), flush=True)
    num = lambda s: int(s[2:])
    fr = lambda p: np.log(p[::5] + 1e-4)
    ut = lambda p: np.r_[np.log(p + 1e-4).mean(0), np.log(p + 1e-4).std(0)]
    rep: dict = {"n_utt": len(data)}
    fold = {k: k % 5 for k in range(100)}
    for kind, fx in (("frame", fr), ("utt", ut)):
        sc_m = {v: [] for v in VAR[1]}
        sc_f = {v: [] for v in VAR[0]}
        for k in range(5):
            tr = [d for d in data if fold[num(d[1])] != k]
            te = [d for d in data if fold[num(d[1])] == k]
            X = np.concatenate([np.atleast_2d(fx(d[3]["raw"])) for d in tr])
            y = np.concatenate([[d[0]] * len(np.atleast_2d(fx(d[3]["raw"]))) for d in tr])
            s = StandardScaler().fit(X)
            m = LogisticRegression(C=0.1 if kind == "frame" else 0.05, max_iter=4000).fit(s.transform(X), y)
            for d in te:
                for v in VAR[d[0]]:
                    p = m.predict_proba(s.transform(np.atleast_2d(fx(d[3][v]))))[:, 1]
                    (sc_m if d[0] == 1 else sc_f)[v].append(p)
        cat = lambda L: np.concatenate(L)
        fr0 = cat(sc_f["raw"])
        mr0 = cat(sc_m["raw"])
        rep[kind] = {}
        for v in VAR[1]:
            pm = cat(sc_m[v])
            rep[kind][f"male_{v}"] = {"auc_vs_female_raw": round(float(roc_auc_score(np.r_[np.ones(len(pm)), np.zeros(len(fr0))], np.r_[pm, fr0])), 4), "p_male_median": round(float(np.median(pm)), 3)}
        for v in VAR[0]:
            pf = cat(sc_f[v])
            rep[kind][f"female_{v}"] = {"auc_male_raw_vs": round(float(roc_auc_score(np.r_[np.ones(len(mr0)), np.zeros(len(pf))], np.r_[mr0, pf])), 4), "p_male_median": round(float(np.median(pf)), 3)}
        print(kind, json.dumps(rep[kind]), flush=True)
    rep["argmax_keep"] = {}
    for g in (1, 0):
        for v in VAR[g][1:]:
            rep["argmax_keep"][f"{'male' if g else 'female'}_{v}"] = round(float(np.mean(np.concatenate([d[3][v].argmax(1) == d[3]["raw"].argmax(1) for d in data if d[0] == g]))), 3)
    print("argmax_keep", json.dumps(rep["argmax_keep"]), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
