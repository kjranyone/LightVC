"""A7a(学習なし・事前検査): 演奏に声門の次元を足すと、学習に無い叫び・喘ぎ・混ざりが演奏の上で外れるか(results/snv_a7a/prereg.yaml)。
表現 P = SPARC 15・PG = P + 声門 9・G = 声門 9・A = log-mel 32(天井)。学習の分(570 声の普通の発話 + 見た 4 種)の最近傍までの距離で、
開発の組 E1V の 30 声ごとに 抜く 3 種 対 見た 4 種 のフレームの AUROC を出し、声で束ねる。声の漏れ(QDA)は記録だけ。出力 = results/snv_a7a/probe.json。
seg W = amendment_1 の区間の段(因果の窓 W フレーム・hop 11 の z の平均と標準偏差)。出力 = results/snv_a7a/probe_seg{W}.json。
    cd training && uv run --no-sync python snv_diag/a7a_glot_probe.py
    cd training && uv run --no-sync python snv_diag/a7a_glot_probe.py seg 43
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT
from snv_glot import NAMES
from train_snv_a2 import FEAT, REF_UTT, VAL_VOICES, playing

HELD = ("scream", "moan", "mixed")
SEEN = ("sigh", "pant", "cry", "laugh")
ACT = -9.0
N_HALF = 150000
P_N, P_V = 0.08, 0.4
N_BOOT = 2000
SEG_HOP = 11
OUT = ROOT / "results/snv_a7a"


def reps(mel: Path, sp: Path, gl: Path) -> tuple[dict, np.ndarray]:
    m = np.load(mel).astype(np.float32)
    n = m.shape[1]
    p = playing(sp, n)
    g = np.load(gl)
    a = m.reshape(32, 4, n).mean(1).T
    act = m.mean(0) > ACT
    return {"P": p, "G": g, "PG": np.concatenate([p, g], 1), "A": a}, act


def voices() -> tuple[list[str], list[str], dict, list]:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    nv = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0]
    nfiles = {}
    for v in man["voices"]:
        fs = [f["t"] for f in v["files"] if (FEAT / f"mel/e1/{v['voice']}/{f['t']}.npy").exists() and (FEAT / f"sparc/e1/{v['voice']}/{f['t']}.npz").exists()]
        nfiles[v["voice"]] = fs
    has_v = {it["voice"] for it in nv if it["cat"] not in HELD and (FEAT / f"mel/e1nv/{it['voice']}/{it['cat']}_0.npy").exists() and (FEAT / f"sparc/e1nv/{it['voice']}/{it['cat']}_0.npz").exists()}
    names = sorted(v for v, fs in nfiles.items() if len(fs) >= REF_UTT + 1 and v in has_v)
    return names[:-VAL_VOICES], names[-VAL_VOICES:], nfiles, nv


def paths(kind: str, voice: str, stem: str) -> tuple[Path, Path, Path]:
    return FEAT / f"mel/{kind}/{voice}/{stem}.npy", FEAT / f"sparc/{kind}/{voice}/{stem}.npz", FEAT / f"glot/{kind}/{voice}/{stem}.npy"


def nn_dist(Q: np.ndarray, R: torch.Tensor) -> np.ndarray:
    if len(Q) == 0:
        return np.zeros(0, np.float32)
    q = torch.from_numpy(Q).float().to(DEV)
    return torch.cat([torch.cdist(q[i:i + 1024], R).min(1).values for i in range(0, len(q), 1024)]).cpu().numpy()


def segs(z: np.ndarray, act: np.ndarray, w: int) -> np.ndarray:
    ends = list(range(w - 1, len(z), SEG_HOP)) or [len(z) - 1]
    out = []
    for e in ends:
        a = act[max(0, e - w + 1): e + 1]
        if a.mean() >= 0.5:
            x = z[max(0, e - w + 1): e + 1][a]
            out.append(np.concatenate([x.mean(0), x.std(0)]))
    return np.array(out, np.float32).reshape(-1, 2 * z.shape[1])


def boot_mean(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def main() -> int:
    rng = np.random.default_rng(0)
    seg = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "seg" else 0
    tr, val, nfiles, nv = voices()
    print("train voices", len(tr), "val voices", len(val), val[0], val[-1], flush=True)
    pool: dict = {"n": {}, "v": {}}
    for v in tr:
        items = [("n", "e1", t) for t in nfiles[v]] + [("v", "e1nv", f"{c}_0") for c in SEEN]
        for side, kind, stem in items:
            mp, sp, gp = paths(kind, v, stem)
            if not (mp.exists() and sp.exists() and gp.exists()):
                assert side == "v", (v, stem)
                continue
            r, act = reps(mp, sp, gp)
            keep = act & (rng.random(len(act)) < (P_N if side == "n" else P_V))
            for k, x in r.items():
                pool[side].setdefault(k, []).append(x[keep])
    ref = {}
    for k in ("P", "G", "PG", "A"):
        xn, xv = np.concatenate(pool["n"][k]), np.concatenate(pool["v"][k])
        ref[k] = np.concatenate([xn[rng.choice(len(xn), N_HALF, replace=False)], xv[rng.choice(len(xv), min(N_HALF, len(xv)), replace=False)]])
    print("ref frames", {k: len(x) for k, x in ref.items()}, "pool sizes n/v", len(np.concatenate(pool["n"]["P"])), len(np.concatenate(pool["v"]["P"])), flush=True)
    del pool
    stats = {k: (x.mean(0), x.std(0) + 1e-4) for k, x in ref.items()}
    fstats = stats
    if seg:
        pool = {"n": {}, "v": {}}
        for v in tr:
            items = [("n", "e1", t) for t in nfiles[v]] + [("v", "e1nv", f"{c}_0") for c in SEEN]
            for side, kind, stem in items:
                pp = paths(kind, v, stem)
                if not all(x.exists() for x in pp):
                    continue
                r, act = reps(*pp)
                for k, x in r.items():
                    pool[side].setdefault(k, []).append(segs((x - fstats[k][0]) / fstats[k][1], act, seg))
        ref = {}
        for k in ("P", "G", "PG", "A"):
            xn, xv = np.concatenate(pool["n"][k]), np.concatenate(pool["v"][k])
            ref[k] = np.concatenate([xn[rng.choice(len(xn), min(N_HALF, len(xn)), replace=False)], xv[rng.choice(len(xv), min(N_HALF, len(xv)), replace=False)]])
        print("seg", seg, "ref windows", {k: len(x) for k, x in ref.items()}, "pool sizes n/v", len(np.concatenate(pool["n"]["P"])), len(np.concatenate(pool["v"]["P"])), flush=True)
        del pool
        stats = {k: (x.mean(0), x.std(0) + 1e-4) for k, x in ref.items()}
    q: dict = {}
    for v in val:
        for c in HELD + SEEN:
            mp, sp, gp = paths("e1nv", v, f"{c}_0")
            r, act = reps(mp, sp, gp)
            q[(v, c)] = {k: (segs((x - fstats[k][0]) / fstats[k][1], act, seg) if seg else x[act]) for k, x in r.items()}
    variants = {"P": ("P", None), "PG": ("PG", None), "G": ("G", None), "A": ("A", None)}
    for j, nm in enumerate(NAMES):
        drop = {15 + j, 39 + j} if seg else {15 + j}
        variants[f"PG-{nm}"] = ("PG", [i for i in range(48 if seg else 24) if i not in drop])
    res: dict = {"prereg": "results/snv_a7a/prereg.yaml" + (" amendment_1" if seg else ""), "unit": f"seg{seg}" if seg else "frame",
                 "n_ref": {k: int(len(x)) for k, x in ref.items()}, "n_query": int(sum(len(d["P"]) for d in q.values())),
                 "n_query_by_cat": {c: int(sum(len(q[(v, c)]["P"]) for v in val)) for c in HELD + SEEN}, "rep": {}}
    per_voice: dict = {}
    for name, (base, cols) in variants.items():
        mu, sd = stats[base]
        Rz = (ref[base] - mu) / sd
        if cols is not None:
            Rz = Rz[:, cols]
        R = torch.from_numpy(Rz).float().to(DEV)
        dist = {}
        for key, d in q.items():
            z = (d[base] - mu) / sd
            dist[key] = nn_dist(z if cols is None else z[:, cols], R)
        del R
        auc_all, auc_cat = [], {c: [] for c in HELD}
        for v in val:
            s = np.concatenate([dist[(v, c)] for c in SEEN])
            h = np.concatenate([dist[(v, c)] for c in HELD])
            if len(s) and len(h):
                auc_all.append(roc_auc_score(np.r_[np.zeros(len(s)), np.ones(len(h))], np.r_[s, h]))
            for c in HELD:
                if len(s) and len(dist[(v, c)]):
                    auc_cat[c].append(roc_auc_score(np.r_[np.zeros(len(s)), np.ones(len(dist[(v, c)]))], np.r_[s, dist[(v, c)]]))
        per_voice[name] = {"all": np.array(auc_all), **{c: np.array(x) for c, x in auc_cat.items()}}
        hs = np.concatenate([dist[k] for k in dist if k[1] in HELD])
        ss = np.concatenate([dist[k] for k in dist if k[1] in SEEN])
        res["rep"][name] = {"auc_all": boot_mean(per_voice[name]["all"], rng), **{f"auc_{c}": boot_mean(per_voice[name][c], rng) for c in HELD},
                            "held_over_seen_median": round(float(np.median(hs) / np.median(ss)), 3),
                            "median_by_cat": {c: round(float(np.median(np.concatenate([dist[(v, c)] for v in val]))), 3) for c in HELD + SEEN}}
        print(name, res["rep"][name], flush=True)
    for part in ("all", "scream"):
        dPG = per_voice["PG"][part] - per_voice["P"][part]
        gap = per_voice["A"][part] - per_voice["P"][part]
        idx = rng.integers(0, len(dPG), (N_BOOT, len(dPG)))
        cl = dPG[idx].mean(1) / gap[idx].mean(1)
        res[f"diff_PG_minus_P_{part}"] = boot_mean(dPG, rng)
        res[f"closure_{part}"] = [round(float(dPG.mean() / gap.mean()), 3), round(float(np.percentile(cl, 2.5)), 3), round(float(np.percentile(cl, 97.5)), 3)]
    res["ablation_drop_auc_all"] = {nm: round(float(per_voice["PG"]["all"].mean() - per_voice[f"PG-{nm}"]["all"].mean()), 4) for nm in NAMES}
    leak: dict = {}
    for k in () if seg else ("P", "G", "PG", "A"):
        mu, sd = stats[k]
        Xf, yf, Xt, ut = [], [], [], []
        for vi, v in enumerate(val):
            fs = nfiles[v]
            for ti, t in enumerate(fs):
                r, act = reps(*paths("e1", v, t))
                z = ((r[k] - mu) / sd)[act]
                if ti < len(fs) // 2:
                    Xf.append(z); yf.append(np.full(len(z), vi))
                else:
                    Xt.append(z); ut.append(vi)
        clf = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(np.concatenate(Xf), np.concatenate(yf))
        utt = [int(np.argmax(clf.predict_log_proba(x).sum(0)) == y) for x, y in zip(Xt, ut)]
        fr = float(np.mean(np.concatenate([clf.predict(x) == y for x, y in zip(Xt, ut)])))
        leak[k] = {"utt_acc": round(float(np.mean(utt)), 3), "frame_acc": round(fr, 3)}
    if not seg:
        res["voice_leak_qda_record_only"] = leak
    valid = res["rep"]["A"]["auc_all"][0] >= 0.65
    pos = res["diff_PG_minus_P_all"][1] > 0
    big = res["closure_all"][0] >= 0.5 and res["closure_scream"][0] >= 0.5
    res["verdict"] = "INCONCLUSIVE" if not valid else ("PASS" if pos and big else ("PARTIAL" if pos else "FAIL"))
    print(json.dumps({k: v for k, v in res.items() if k != "rep"}, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (f"probe_seg{seg}.json" if seg else "probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
