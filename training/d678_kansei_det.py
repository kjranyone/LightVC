"""D678: 官能指数 v1 = 凍結の WavLM-large の 0.5 s の窓の本物 / 合成の線形の検出器(族を学習から外す fold で、耳に出した合成を全部落とし本物を通すか)。
規則・L / I / C は results/conv_p0/d678/prereg.yaml。
    uv run python d678_kansei_det.py --stage list | feat | fit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = ROOT / "results/conv_p0/d678"
FEAT = H / "d678_feat"
LAYERS = (3, 6, 9)
WIN, HOPW = 8000, 4000
WORLD = re.compile(r"^(GU|REFK|REFW|WAP|WEG|WFG|WRT|GW_|GS_|GM_|S\d+_)")
GEN = re.compile(r"^(OURS\d+(RY|Y|R$|Rg|$)|RUST7$)")
REAL_DIRS = {"REAL"}


def family(dirname: str) -> str | None:
    if dirname in REAL_DIRS:
        return None
    if WORLD.match(dirname):
        return "world"
    if GEN.match(dirname):
        return "gen"
    return "render"


def ear_family(r: dict) -> str:
    s, sy = r["sess"], r["sys"]
    if s in ("zsvc_ab", "zsvc_ab2", "ddsp_ab", "ddsp_sim", "ddsp_chorus", "dspvc_p0"):
        return "old"
    if s == "mf_v5_ear":
        return "world" if sy.startswith("GU") else "gen"
    if s == "now_v6_dev":
        return {"gu": "world", "prev": "gen", "now": "render"}[sy]
    return "gen"


def build_lists() -> dict:
    import d676_kansei_band as K
    rng = random.Random(678)
    ear = K.stimuli()
    tg = set()
    for k in ("mf_v5_ear", "mf_v3_ear", "ff_v3_ear", "ff_ab_ear", "mf_pitch_ear", "mf_pitch_ear2"):
        kf = [p for p in (K.EB / k).glob("*.json") if "key" in p.name]
        if kf:
            tg |= {v.get("target") for v in json.loads(kf[0].read_text()).values() if isinstance(v, dict) and v.get("target")}
    for s in ("dev", "dev2", "dev3", "devff", "devff2", "devff3"):
        f = H / s / "devset.json"
        if f.exists():
            d = json.loads(f.read_text())
            for t in d.get("targets", []):
                tg.add(t if isinstance(t, str) else t.get("spk", t.get("id", "")))
    base = ROOT / "female-dataset"
    spk = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in tg)
    rng.shuffle(spk)
    pick = lambda s: str(sorted((base / s).glob("*.wav"))[len(list((base / s).glob("*.wav"))) // 2])
    L: dict = {"real_train": [], "real_calib": [], "real_test_pool": [], "real_test_male": [], "synth_train": [], "ear": [], "tts": []}
    L["real_train"] += [{"path": pick(s), "src": "pool"} for s in spk[:500]]
    L["real_calib"] += [{"path": pick(s), "src": "pool"} for s in spk[500:575]]
    L["real_test_pool"] += [{"path": pick(s), "src": "pool"} for s in spk[575:725]]
    for s in ("dev2", "dev3"):
        L["real_train"] += [{"path": str(f), "src": f"ref_{s}"} for f in sorted((H / s / "d37/_work").glob("ref_*.wav"))]
    for d in ("devvctkm", "d487_uiro"):
        fs = sorted((H / d / "d56").glob("*_SRC.wav"))
        L["real_train"] += [{"path": str(f), "src": f"male_{d}"} for f in fs[: len(fs) // 2]]
        L["real_test_male"] += [{"path": str(f), "src": f"male_{d}"} for f in fs[len(fs) // 2:]]
    L["real_train"] += [{"path": str(f), "src": "devff_op"} for f in sorted((H / "devff" / "d56").glob("*_SRC.wav"))]
    seen = set()
    for s in ("dev2", "dev3"):
        for d in sorted((H / s / "d570").iterdir()):
            if not d.is_dir():
                continue
            fs = sorted(d.glob("*.wav"))
            if not fs:
                continue
            real_dir = Path(os.path.realpath(fs[0])).parent
            if (s, str(real_dir)) in seen:
                continue
            seen.add((s, str(real_dir)))
            fam = family(real_dir.name)
            if fam is None:
                continue
            for f in rng.sample(fs, min(10, len(fs))):
                L["synth_train"].append({"path": os.path.realpath(f), "fam": fam, "sys": real_dir.name, "set": s})
    for r in ear:
        if r["role"] == "synth":
            L["ear"].append(r | {"fam": ear_family(r)})
        elif r["role"] == "real":
            L["ear"].append(r | {"fam": "real"})
        else:
            L["tts"].append(r)
    return L


def wavlm():
    import torch
    from transformers import WavLMModel
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval()
    return net


def windows(path: str) -> np.ndarray:
    import soundfile as sf
    import soxr
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = soxr.resample(x.mean(1), sr, 16000)[: 8 * 16000]
    n = 1 + max(0, (len(x) - WIN) // HOPW)
    W = np.stack([np.pad(x[i * HOPW:i * HOPW + WIN], (0, max(0, WIN - len(x[i * HOPW:i * HOPW + WIN])))) for i in range(n)])
    e = 10 * np.log10((W ** 2).mean(1) + 1e-12)
    W = W[e > e.max() - 30]
    W = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-7)
    return W.astype(np.float32)


def embed(net, W: np.ndarray) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(W), 64):
            hs = net(torch.from_numpy(W[i:i + 64]).cuda(), output_hidden_states=True).hidden_states
            out.append(torch.cat([hs[l].mean(1) for l in LAYERS], 1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["list", "feat", "fit"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "list":
        L = build_lists()
        (OUT / "lists.json").write_text(json.dumps(L, ensure_ascii=False))
        print({k: len(v) for k, v in L.items()}, flush=True)
        from collections import Counter
        print("synth_train fam", Counter(r["fam"] for r in L["synth_train"]), "ear fam", Counter(r["fam"] for r in L["ear"]), flush=True)
        return 0
    if a.stage == "feat":
        L = json.loads((OUT / "lists.json").read_text())
        net = wavlm()
        FEAT.mkdir(parents=True, exist_ok=True)
        paths = sorted({r["path"] for v in L.values() for r in v})
        idx = {}
        rng = np.random.default_rng(678)
        synth = {r["path"] for r in L["synth_train"]}
        for i, p in enumerate(paths):
            W = windows(p)
            if len(W) == 0:
                continue
            if p in synth and len(W) > 12:
                W = W[np.sort(rng.choice(len(W), 12, replace=False))]
            idx[p] = embed(net, W)
            if i % 200 == 0:
                print("feat", i, len(paths), flush=True)
        np.savez(FEAT / "feats.npz", **{f"f{i}": v for i, v in enumerate(idx.values())})
        (FEAT / "keys.json").write_text(json.dumps(list(idx.keys())))
        print("feat done", len(idx), flush=True)
        return 0
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    L = json.loads((OUT / "lists.json").read_text())
    z = np.load(FEAT / "feats.npz")
    keys = json.loads((FEAT / "keys.json").read_text())
    F = {k: z[f"f{i}"].astype(np.float32) for i, k in enumerate(keys)}
    has = lambda rows: [r for r in rows if r["path"] in F]
    rep: dict = {}
    models = {}
    for fold in ("world", "gen", "render", "none"):
        Xr = np.concatenate([F[r["path"]] for r in has(L["real_train"])])
        Xs = np.concatenate([F[r["path"]] for r in has(L["synth_train"]) if r["fam"] != fold])
        X = np.concatenate([Xr, Xs])
        y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xs))])
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
        score = lambda p: float(clf.decision_function(sc.transform(F[p])).mean())
        tau = float(np.percentile([score(r["path"]) for r in has(L["real_calib"])], 95))
        models[fold] = (score, tau)
        import pickle
        (OUT / f"model_{fold}.pkl").write_bytes(pickle.dumps({"scaler": sc, "clf": clf, "tau": tau, "layers": LAYERS}))
        rep[fold] = {"n_real_win": int(len(Xr)), "n_synth_win": int(len(Xs)), "tau": round(tau, 3),
                     "test_pool_fail": round(float(np.mean([score(r["path"]) > tau for r in has(L["real_test_pool"])])), 3),
                     "test_male_fail": round(float(np.mean([score(r["path"]) > tau for r in has(L["real_test_male"])])), 3),
                     "ear_real_fail": round(float(np.mean([score(r["path"]) > tau for r in has(L["ear"]) if r["fam"] == "real"])), 3),
                     "tts_fail": round(float(np.mean([score(r["path"]) > tau for r in has(L["tts"])])), 3)}
        print(fold, rep[fold], flush=True)
    rows = []
    for r in has(L["ear"]):
        if r["fam"] == "real":
            continue
        fold = r["fam"] if r["fam"] in ("world", "gen", "render") else "none"
        sc_, tau = models[fold]
        s = sc_(r["path"])
        rows.append({"sess": r["sess"], "sys": r["sys"], "fam": r["fam"], "fold": fold, "score": round(s, 3), "tau": round(tau, 3), "fail": bool(s > tau)})
    (OUT / "ear_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0))
    by = lambda key: {v: round(float(np.mean([x["fail"] for x in rows if x[key] == v])), 3) for v in sorted({x[key] for x in rows})}
    rep["ear_synth_fail"] = round(float(np.mean([x["fail"] for x in rows])), 3)
    rep["ear_synth_fail_by_family"] = by("fam")
    rep["ear_synth_fail_by_session"] = by("sess")
    f = rep["none"]
    rep["valid"] = bool(all(rep[k]["test_pool_fail"] <= 0.10 and rep[k]["test_male_fail"] <= 0.15 and rep[k]["ear_real_fail"] <= 0.15 for k in ("world", "gen", "render", "none"))
                        and rep["ear_synth_fail"] >= 0.90 and all(v >= 0.80 for v in rep["ear_synth_fail_by_family"].values()))
    (OUT / "fit.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("ear_synth_fail", "ear_synth_fail_by_family", "ear_synth_fail_by_session", "valid")}, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
