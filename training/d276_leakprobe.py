"""D276: C1 の事後に残る性別の情報の探針(diag_ppgvoc1 の分岐「SR は良いが PV が悪い」の材料)。同じ TTS 系(Irodori v3)の男声 30 人 × 3 と女声 30 人 × 3 の
C1 の事後(Python・Rust と差 0)で、フレーム単位と 0.5 秒の窓の平均で男女を当てる(話者でまとめた 5 分割・ロジスティック回帰)。事後の形 = 柔らかい(τ 1)・鋭い(τ 0.5)・硬い(one-hot)。
文の集合は男女で違いうる(窓の平均は音素の頻度の差と交絡)。CPU。
    uv run python d276_leakprobe.py --out ../results/conv_p0/d276.json
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
                with torch.no_grad():
                    lg = c1(mf(torch.from_numpy(T1.prime(x))[None]))[0].T.numpy()
                n = len(x) // 240
                data.append((g, s.name, lg[-n:]))
    print("clips", len(data), "male", sum(d[0] for d in data), flush=True)
    spk = sorted({d[1] for d in data})
    fold = {s: i % 5 for i, s in enumerate(rng.sample(spk, len(spk)))}

    def forms(lg: np.ndarray) -> dict:
        t = torch.from_numpy(lg)
        hard = torch.nn.functional.one_hot(t.argmax(1), t.shape[1]).float()
        return {"soft": t.softmax(1).numpy(), "sharp_t0.5": (t / 0.5).softmax(1).numpy(), "hard": hard.numpy()}

    rep: dict = {}
    for name in ("soft", "sharp_t0.5", "hard"):
        for lvl in ("frame", "win0.5s"):
            X, y, f = [], [], []
            for g, s, lg in data:
                p = forms(lg)[name]
                if lvl == "frame":
                    idx = np.arange(0, len(p), 5)
                    X.append(np.log(p[idx] + 1e-4) if name != "hard" else p[idx]); y += [g] * len(idx); f += [fold[s]] * len(idx)
                else:
                    W = 100
                    for o in range(0, len(p) - W + 1, W):
                        X.append(p[o:o + W].mean(0, keepdims=True)); y.append(g); f.append(fold[s])
            X, y, f = np.concatenate(X), np.array(y), np.array(f)
            pr = np.zeros(len(y))
            for k in range(5):
                te = f == k
                sc = StandardScaler().fit(X[~te])
                m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(X[~te]), y[~te])
                pr[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
            rep[f"{name}/{lvl}"] = {"auc": round(float(roc_auc_score(y, pr)), 4), "n": int(len(y))}
            print(name, lvl, rep[f"{name}/{lvl}"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
