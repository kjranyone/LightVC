"""C1 の検査 I の探針(学習前): 教師(ContentVec の単位の事後)そのものが性別をどれだけ運ぶか。

コードブック = train_c1.build_codebook(学習データだけ・学習の比率で標本・本番の学習もこれを読む: results/<tag>/codebook.pt)。
材料 = VCTK の評価話者(英語・男 24 / 女 45・発話 41〜59・学習に入れない)= 言語は揃え、性別だけ違う。
  単位の純度: 各単位に割り当たるフレームの多数派の性別の割合(フレーム数で重み)。偶然(全体の男女比)と並べる。
  線形探針: フレームの事後(K 次元)から性別をロジスティック回帰(話者を分けて 5 分割)。偶然の正解率と並べる。

    uv run python c1_probe_teacher.py --tag c1_1
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
import nvoc as N
import train_c1 as T1
import train_f0est as TF


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1_1")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.05)
    a = ap.parse_args()
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from train_ddsp_vc import load48
    dev = "cuda"
    out = T1.ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    real, tts = TF.female_rows()
    groups, probs = [TF.male_rows("train"), real, tts, T1.vctk_female_rows()], [0.5, 0.2, 0.2, 0.1]
    tch = T1.Teacher(dev, torch.zeros(a.k, 768), a.tau)
    cbp = out / "codebook.pt"
    if cbp.exists():
        Cb = torch.load(cbp)
    else:
        Cb = T1.build_codebook(tch, groups, probs, a.k, dev, 3000)
        torch.save(Cb, cbp)
    tch.C = Cb.to(dev)
    ev_f, ev_m = T1.eval_vctk()
    X, y, g = [], [], []
    for gi, (spks, lab) in enumerate(((sorted(ev_m), 1), (sorted(ev_f), 0))):
        for s in spks:
            for u in range(41, 47):
                w = T1.VC / "wav48" / s / f"{s}_{u:03d}.wav"
                if not w.exists():
                    continue
                x = load48(str(w)).astype(np.float64)
                x16 = torch.from_numpy(resample_poly(x, 1, 3).astype(np.float32))[None].to(dev)
                p = tch.post(x16)[0].cpu().numpy()
                h = tch.feats(x16)[0]
                e = (x16[0].unfold(0, 400, 320) ** 2).mean(1).cpu().numpy()[:len(p)]
                keep = 10 * np.log10(e + 1e-12) > 10 * np.log10(e.max() + 1e-12) - 40
                X.append(p[keep]); y.append(np.full(keep.sum(), lab)); g.append(np.full(keep.sum(), hash(s) % 5))
    X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
    u = X.argmax(1)
    pur, tot = 0.0, 0
    for j in range(a.k):
        m = u == j
        if m.sum():
            pur += max(y[m].mean(), 1 - y[m].mean()) * m.sum()
            tot += m.sum()
    base = max(y.mean(), 1 - y.mean())
    accs = []
    for f in range(5):
        tr, te = g != f, g == f
        if te.sum() == 0 or len(set(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(np.log(X[tr] + 1e-6), y[tr])
        accs.append(float(clf.score(np.log(X[te] + 1e-6), y[te])))
    rep = {"frames": int(len(y)), "male_frac": round(float(y.mean()), 3), "chance": round(float(base), 3),
           "unit_gender_purity_weighted": round(pur / tot, 3), "linear_probe_gender_acc_speaker_split": round(float(np.mean(accs)), 3),
           "note": "VCTK 評価話者(英語・男 24 / 女 45・発話 41〜46・音声区間)。教師の事後(τ = %.2f・K = %d)から性別がどれだけ読めるか = 内容の経路の性別の漏れの床" % (a.tau, a.k)}
    (out / "teacher_gender_probe.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
