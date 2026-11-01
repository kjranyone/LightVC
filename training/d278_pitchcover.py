"""D278: 事後に残る男女の情報は声の高さか(D277 で周波数軸の伸縮は無効)。女声を WORLD(dio + stonemask・cheaptrick・d4c)で高さだけ k 倍にして再合成した C1 の事後を、
D277 と同じ判別器で当てる。k = 1(WORLD の加工だけの対照)・0.7・0.5。男声は k = 1・2。CPU。
    uv run python d278_pitchcover.py --out ../results/conv_p0/d278.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
FK = {0: ("raw", 1.0, 0.7, 0.5), 1: ("raw", 1.0, 2.0)}


def world_shift(x: np.ndarray, k: float) -> np.ndarray:
    import pyworld as pw
    xd = x.astype(np.float64)
    f0, t = pw.dio(xd, 48000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    f0 = pw.stonemask(xd, f0, t, 48000)
    sp = pw.cheaptrick(xd, f0, t, 48000)
    ap = pw.d4c(xd, f0, t, 48000)
    y = pw.synthesize(f0 * k, sp, ap, 48000, 5.0)[: len(x)]
    return (np.pad(y, (0, len(x) - len(y))) * (np.abs(x).max() / (np.abs(y).max() + 1e-9))).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(2)
    import c1_content as CC
    import train_c1 as T1
    from d6_unit_cov import load48
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])); c1.load_state_dict(st1["net"]); c1.eval()
    mf = CC.MelFront()
    rng = random.Random(276)
    data = []
    for g, root in ((1, ROOT / "data/male_tts_corpus"), (0, ROOT / "data/female_tts_corpus")):
        spks = sorted(d for d in root.iterdir() if d.is_dir())
        rng.shuffle(spks)
        for s in spks[:a.n_spk]:
            ws = sorted(s.glob("*.wav"))
            for w in rng.sample(ws, min(3, len(ws))):
                x = load48(w)
                if len(x) < 48000:
                    continue
                n = len(x) // 240
                ps = {}
                with torch.no_grad():
                    for k in FK[g]:
                        xk = x if k == "raw" else world_shift(x, k)
                        ps[k] = c1(mf(torch.from_numpy(T1.prime(xk))[None]))[0].T.softmax(1).numpy()[-n:]
                data.append((g, s.name, ps))
    d_id = 0.0
    spk = sorted({d[1] for d in data})
    order = rng.sample(spk, len(spk))
    test_spk = set(order[: len(order) // 3])
    feat = lambda p: np.log(p[::5] + 1e-4)
    Xtr = np.concatenate([feat(d[2]["raw"]) for d in data if d[1] not in test_spk])
    ytr = np.concatenate([[d[0]] * len(feat(d[2]["raw"])) for d in data if d[1] not in test_spk])
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xtr), ytr)
    pm = np.concatenate([m.predict_proba(sc.transform(feat(d[2]["raw"])))[:, 1] for d in data if d[1] in test_spk and d[0] == 1])
    rep: dict = {"male_raw_p_median": round(float(np.median(pm)), 3), "n_test_spk": len(test_spk)}
    for g in (0, 1):
        for k in FK[g]:
            pf = np.concatenate([m.predict_proba(sc.transform(feat(d[2][k])))[:, 1] for d in data if d[1] in test_spk and d[0] == g])
            keep = float(np.mean(np.concatenate([(d[2][k].argmax(1) == d[2]["raw"].argmax(1)) for d in data if d[0] == g])))
            auc = roc_auc_score(np.r_[np.ones(len(pm)), np.zeros(len(pf))], np.r_[pm, pf])
            key = f"{'male' if g else 'female'}_k{k}"
            rep[key] = {"auc_vs_male_raw": round(float(auc), 4), "p_male_median": round(float(np.median(pf)), 3), "argmax_keep_vs_raw": round(keep, 3)}
            print(key, rep[key], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
