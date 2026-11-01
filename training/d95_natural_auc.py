"""D95 = D94 の派生(話者を問わない自然さ): 本物 = 評価に使わない女声 --n_pool 人の 1 発話ずつ(8s)・候補 = 各組の刺激。交差検証は 5 分割(本物は話者・候補は目標で分ける)。
変換が写し合成並みに下がれば不足は話者らしさ・1.00 のままなら自然さ(ガビ・こもり・抑揚)。
以下は D94 の説明。D94(学習なし): 解析の合格 = 機械が本物と区別できないこと(二標本検定)。D93 の典型性(全体の音響統計)は変換の失敗を見逃した(変換は本物の分布の内側)。
評価の 30 目標(実女声)ごとに、本物 = cen_TTT + 登録の参照 ref_TTT の 8s 区間 2 つ(計 90)・候補の組 = その目標の刺激(写し合成 / 変換)。
特徴 = 凍結 WavLM-large の層 4/6/8 の時間平均 + 標準偏差(6144 次元・d4b と同じ)→ 標準化 → L2 ロジスティック回帰。目標で分けた 5 分割の交差検証の AUC(本物 対 候補)。
同じ目標どうしを比べる(話者の違いでは分かれない・話者が違って聞こえる変換は正しく分かれる)。
較正(耳のラベル): 合格 = d61 ORA・d68 DYN・d62 TEACHER / 失格 = d66 TAB・d73 BTAB/BW07/W10・d85_360k C05/C07。合格の線 = 合格の写し合成の AUC の最大 + 0.05。
出力 = results/conv_p0/d94.json。
    uv run python d94_two_sample.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
SETS = [("d61", "ORA", True), ("d68", "DYN", True), ("d62", "TEACHER", True), ("d66", "TAB", False), ("d73", "BTAB", False), ("d73", "BW07", False),
        ("d73", "W10", False), ("d85_360k", "C05", False), ("d85_360k", "C07", False)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="追加の候補 <dir>:<cond>(判定する新しい候補)")
    ap.add_argument("--n_pool", type=int, default=150)
    ap.add_argument("--pool", default="real", choices=("real", "tts"))
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    S = Path(a.s)
    dev = "cuda"
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()

    def load16(p: Path, off: float = 0.0) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)[int(off * sr): int((off + 8.0) * sr)]
        return resample_poly(x, 16000, sr).astype(np.float32)

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
        v = []
        for l in (4, 6, 8):
            h = hs[l][0]
            v += [h.mean(0), h.std(0)]
        return torch.cat(v).cpu().numpy()

    import random
    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / ("female-dataset" if a.pool == "real" else "data/female_tts_corpus")
    if a.pool == "tts":
        excl |= {r["pair"].split("_")[1] for r in json.loads((ROOT / "results/conv_p0/d80.json").read_text())["rows"]}
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(95).shuffle(pool)
    real_X, real_g = [], []
    for i, spk in enumerate(pool[: a.n_pool]):
        ws = sorted((base / spk).glob("*.wav")) if a.pool == "real" else sorted((base / spk).glob("t0[5-9]_neutral.wav"))
        if not ws:
            continue
        x = load16(ws[len(ws) // 2])
        if len(x) < 3 * 16000:
            continue
        real_X.append(emb(x)); real_g.append(i % 30)
    real_X, real_g = np.array(real_X), np.array(real_g)
    sets = list(SETS) + [(e.split(":")[0], e.split(":")[1], None) for e in a.extra]
    rep = {}
    for d, c, ok in sets:
        X, g = [], []
        for i_, f in enumerate(sorted((S / d).glob(f"*_{c}.wav"))):
            if f.name.startswith("m"):
                X.append(emb(load16(f))); g.append(i_ % 30)
                continue
            m = re.search(r"t(\d+)", f.name.split("_")[1] if f.name.startswith("p") else f.name)
            if not m:
                continue
            t = int(m.group(1))
            if t >= 30:
                continue
            X.append(emb(load16(f))); g.append(t)
        X, g = np.array(X), np.array(g)
        Xa = np.concatenate([real_X, X]); ya = np.r_[np.zeros(len(real_X)), np.ones(len(X))]; ga = np.r_[real_g, g]
        pr = np.zeros(len(ya))
        folds = np.arange(30) % 5
        for k in range(5):
            te = folds[ga] == k
            sc = StandardScaler().fit(Xa[~te])
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
            pr[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
        auc = float(roc_auc_score(ya, pr))
        rep[f"{d}:{c}"] = {"ear_pass": ok, "auc": round(auc, 3), "n": int(len(X))}
        print(d, c, ok, round(auc, 3), len(X), flush=True)
    passed = [v["auc"] for v in rep.values() if v["ear_pass"] is True]
    failed = [v["auc"] for v in rep.values() if v["ear_pass"] is False]
    thr = max(passed) + 0.05
    out = {"sets": rep, "thr": round(thr, 3), "ear_pass_max": max(passed), "ear_fail_min": min(failed), "separates": bool(min(failed) > thr)}
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
    (ROOT / "results/conv_p0/d95.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
