"""N0c(学習なし): 声に依らない発声の種類の物差し(CAT_x)と、音素の無い発声の相対の本人らしさ(ID_rel)の較正。事前登録 = results/snv_n0c/prereg.yaml。
    cd training && uv run --no-sync python snv_n0c_measure.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.stats import binom

from snv_a0_measure import DEV, ROOT, load, to16
from snv_a0b_measure import proj
from snv_n0_measure import cat_feat, trim
from snv_diag.n0_real_nv_id import PAT

N0C = ROOT / "results/snv_n0c"
N0 = ROOT / "results/snv_n0"
A0 = ROOT / "results/snv_a0"
CATS = ("moan", "scream", "sigh", "pant", "cry", "laugh")


def auc(own: list, other: list) -> float:
    o, n = np.asarray(own)[:, None], np.asarray(other)
    return float(np.mean([(o[i] > n[i]).mean() for i in range(len(o))]))


def main() -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def ecapa(y: np.ndarray, sr: int) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(to16(y, sr)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    its = json.loads((N0C / "wav/items.json").read_text())
    X1, X2, y, g, tx = [], [], [], [], []
    for it in its:
        w, sr = load(N0C / f"wav/{it['voice']}_{it['cat']}_{it['t']}_{it['k']}.wav")
        w = trim(w)
        f = cat_feat(to16(w, sr))
        X1.append(f)
        X2.append(np.concatenate([f, ecapa(w, sr)]))
        y.append(CATS.index(it["cat"]))
        g.append(it["voice"])
        tx.append(it["t"])
    X1, X2, y, g, tx = map(np.asarray, (X1, X2, y, g, tx))
    res: dict = {"n": int(len(y))}
    upper = float(binom.ppf(0.975, len(y), 1 / len(CATS)) / len(y))
    for name, X in (("primary", X1), ("secondary", X2)):
        pred = cross_val_predict(clf(), X, y, groups=g, cv=GroupKFold(6))
        yp = np.random.default_rng(0).permutation(y)
        pperm = cross_val_predict(clf(), X, yp, groups=g, cv=GroupKFold(6))
        acc, accp = float(np.mean(pred == y)), float(np.mean(pperm == yp))
        lo = acc - 1.96 * np.sqrt(acc * (1 - acc) / len(y))
        txt = []
        for tr_t, te_t in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            for tr, te in GroupKFold(6).split(X, y, g):
                a = tr[np.isin(tx[tr], tr_t)]
                b = te[np.isin(tx[te], te_t)]
                txt.extend(list(clf().fit(X[a], y[a]).predict(X[b]) == y[b]))
        res[f"CAT_x_{name}"] = {"acc": round(acc, 3), "ci95_lo": round(float(lo), 3), "perm_acc": round(accp, 3), "chance_upper95": round(upper, 3),
                                "pass": bool(lo >= 0.80 and accp <= upper),
                                "pass_point": bool(acc >= 0.80 and accp <= upper),
                                "per_class": {c: round(float(np.mean(pred[y == i] == i)), 3) for i, c in enumerate(CATS)},
                                "confusion": {c: {c2: int(np.sum((y == i) & (pred == j))) for j, c2 in enumerate(CATS)} for i, c in enumerate(CATS)},
                                "text_heldout_acc": round(float(np.mean(txt)), 3)}
        n0 = [it for it in json.loads((N0 / "wav/items.json").read_text()) if it["cat"] in CATS]
        Z, zy = [], []
        for it in n0:
            w, sr = load(N0 / f"wav/{it['voice']}_{it['cat']}_{it['k']}.wav")
            w = trim(w)
            f = cat_feat(to16(w, sr))
            Z.append(f if name == "primary" else np.concatenate([f, ecapa(w, sr)]))
            zy.append(CATS.index(it["cat"]))
        m = clf().fit(X, y)
        res[f"CAT_x_{name}"]["n0_ref_conditioned_acc"] = round(float(np.mean(m.predict(np.stack(Z)) == np.asarray(zy))), 3)
        print(name, {k: v for k, v in res[f"CAT_x_{name}"].items() if k != "confusion"}, flush=True)

    E = lambda y_, sr: proj(ecapa(y_, sr), Tt)
    fem = [f"pf{k:03d}" for k in range(24)]
    enr = {}
    for v in fem:
        ys = [load(A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)]
        enr[v] = E(np.concatenate([w for w, _ in ys]), ys[0][1])
    M = np.stack([enr[v] for v in fem])
    own, oth = defaultdict(list), defaultdict(list)
    for v in fem:
        for c in CATS:
            for k in (0, 1):
                w, sr = load(N0 / f"wav/{v}_{c}_{k}.wav")
                s = M @ E(trim(w), sr)
                i = fem.index(v)
                own["n0_nv"].append(s[i])
                oth["n0_nv"].append(np.delete(s, i))
        for j in (3, 4, 5):
            s = M @ E(*load(A0 / f"wav/{v}_{j}.wav"))
            i = fem.index(v)
            own["n0_normal"].append(s[i])
            oth["n0_normal"].append(np.delete(s, i))

    by = defaultdict(lambda: {"normal": [], "nv": []})
    for lab in sorted((ROOT / "female-dataset").glob("*/*.lab")):
        w = lab.with_suffix(".wav")
        if not w.exists():
            continue
        t = lab.read_text(encoding="utf-8").strip()
        if len(t) <= 16 and PAT.search(t):
            by[lab.parent.name]["nv"].append(w)
        elif len(t) >= 20:
            by[lab.parent.name]["normal"].append(w)
    rng = random.Random(0)
    tst = sorted(s for s, d in by.items() if len(d["normal"]) >= 4 and len(d["nv"]) >= 1)
    pool = rng.sample(sorted(s for s, d in by.items() if len(d["normal"]) >= 3 and s not in tst), 23)

    def rd(p: Path) -> tuple:
        a, sr = sf.read(str(p))
        return (a.mean(1) if a.ndim > 1 else a).astype(np.float32), sr

    renr = {}
    for s in tst + pool:
        ys = [rd(p) for p in sorted(by[s]["normal"])[:3]]
        renr[s] = E(np.concatenate([w for w, _ in ys]), ys[0][1])
    for s in tst:
        R = np.stack([renr[s]] + [renr[u] for u in pool])
        s4 = R @ E(*rd(sorted(by[s]["normal"])[3]))
        own["real_normal"].append(s4[0])
        oth["real_normal"].append(s4[1:])
        for p in sorted(by[s]["nv"])[:4]:
            w, sr = rd(p)
            w = trim(w)
            if len(w) / sr >= 0.25:
                sv = R @ E(w, sr)
                own["real_nv"].append(sv[0])
                oth["real_nv"].append(sv[1:])
    res["ID_rel"] = {k: {"auc": round(auc(own[k], oth[k]), 3), "n": len(own[k])} for k in own}
    res["ID_rel"]["pass"] = bool(res["ID_rel"]["n0_nv"]["auc"] >= 0.75 and res["ID_rel"]["real_nv"]["auc"] >= 0.75)
    print("ID_rel", res["ID_rel"], flush=True)
    (N0C / "n0c_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
