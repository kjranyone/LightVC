"""D277: 入力側の摂動(女声の C1 入力の周波数軸を α 倍に縮める・MelFront の alpha)で、事後の分布が男声の側を覆えるか(diag_ppgvoc1 の次の腕の材料)。
D276 と同じ組(Irodori v3 の男声 30 人 × 3・女声 30 人 × 3)。男女の判別器(柔らかい事後・フレーム単位・ロジスティック)を摂動なしの学習の話者で作り、保留の話者の
(a) 男声・(b) 女声を α で縮めた事後 を当てる。指標 = 男声 対 縮めた女声の AUC(0.5 = 区別できない)・P(男) の中央値・内容の保存(α 1 との argmax の一致)。CPU。
    uv run python d277_warpcover.py --out ../results/conv_p0/d277.json
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
ALPHAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)


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
                xp = torch.from_numpy(T1.prime(x))[None]
                ps = {}
                with torch.no_grad():
                    for al in (ALPHAS if g == 0 else (1.0,)):
                        ps[al] = c1(mf(xp, torch.tensor([al])))[0].T.softmax(1).numpy()[-n:]
                data.append((g, s.name, ps))
    assert np.allclose(data[-1][2][1.0], data[-1][2][1.0]), "恒等"
    with torch.no_grad():
        x0 = torch.from_numpy(T1.prime(load48(sorted((ROOT / "data/female_tts_corpus").iterdir())[0].glob("*.wav").__next__())))[None]
        d_id = float((mf(x0, torch.tensor([1.0])) - mf(x0)).abs().max())
    print("恒等検査 alpha=1 と alpha なしの最大差", d_id, flush=True)
    spk = sorted({d[1] for d in data})
    order = rng.sample(spk, len(spk))
    test_spk = set(order[: len(order) // 3])
    feat = lambda p: np.log(p[::5] + 1e-4)
    Xtr = np.concatenate([feat(d[2][1.0]) for d in data if d[1] not in test_spk])
    ytr = np.concatenate([[d[0]] * len(feat(d[2][1.0])) for d in data if d[1] not in test_spk])
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Xtr), ytr)
    pm = np.concatenate([m.predict_proba(sc.transform(feat(d[2][1.0])))[:, 1] for d in data if d[1] in test_spk and d[0] == 1])
    rep: dict = {"identity_maxdiff": d_id, "male_p_median": round(float(np.median(pm)), 3), "n_test_spk": len(test_spk)}
    for al in ALPHAS:
        pf = np.concatenate([m.predict_proba(sc.transform(feat(d[2][al])))[:, 1] for d in data if d[1] in test_spk and d[0] == 0])
        keep = float(np.mean(np.concatenate([(d[2][al].argmax(1) == d[2][1.0].argmax(1)) for d in data if d[0] == 0])))
        auc = roc_auc_score(np.r_[np.ones(len(pm)), np.zeros(len(pf))], np.r_[pm, pf])
        rep[f"alpha_{al}"] = {"auc_male_vs_warped_female": round(float(auc), 4), "p_male_median": round(float(np.median(pf)), 3), "argmax_keep": round(keep, 3)}
        print(al, rep[f"alpha_{al}"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
