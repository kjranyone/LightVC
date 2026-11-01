"""D468(10-21 査読 E への対応): 男声の名残の対の判別(男声入力由来 対 女声入力由来・内容をそろえた組)の床 = 韻律だけで男女の入力をどこまで見分けられるか。
韻律を保つ VC では、韻律の差はどれだけ声を直しても出力に残る = 対の判別の理想値の目安。特徴(発話ごと・声の音色を含めない):
相対の高さ(有声の log f0 − 発話の有声の中央値 の 半音の標準偏差・四分位幅・|Δ| の中央値・95 パーセンタイル・上昇 / 下降の割合)・有声の割合・有声区間の数 / 秒・
音量(0.1〜4 kHz の帯のパワーの dB の標準偏差・|Δ| の中央値・2〜8 Hz の変調の割合)・無音の割合。f0 と有声は Rust の書き出し(.cond の 25・26 列)・音量は入力の音声。
判別 = ロジスティック回帰・操作者の話者でまとめた 5 分割(d326_pair と同じ)。入力そのもの と 変換の出力(既定 P43OP の d452)の両方で。
    uv run python d468_prosody_floor.py --out ../results/conv_p0/d468.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P
from d6_unit_cov import load48


def feats(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    import scipy.signal as ss
    v = c[:, 26] > 0.5
    lf = c[:, 25]
    st = 12 / np.log(2) * (lf[v] - np.median(lf[v])) if v.sum() > 20 else np.zeros(21)
    d = np.diff(12 / np.log(2) * lf)[(v[1:] & v[:-1])]
    d = d if len(d) > 5 else np.zeros(6)
    segs = np.flatnonzero(np.diff(v.astype(int)) == 1)
    sos = ss.butter(4, [100, 4000], "band", fs=48000, output="sos")
    y = ss.sosfilt(sos, x[: len(c) * 240].astype(np.float64))
    n = len(y) // 240
    e = 10 * np.log10((y[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-10)
    act = e > e.max() - 40
    ea = e[act] if act.sum() > 10 else e
    f, Pw = ss.welch(ea - ea.mean(), fs=200, nperseg=min(256, len(ea)))
    mod = Pw[(f >= 2) & (f <= 8)].sum() / (Pw[(f > 0.3) & (f <= 30)].sum() + 1e-12)
    return np.array([st.std(), np.subtract(*np.percentile(st, [75, 25])), np.median(np.abs(d)), np.percentile(np.abs(d), 95), (d > 0.3).mean(), (d < -0.3).mean(),
                     v.mean(), len(segs) / (len(v) / 200), ea.std(), np.median(np.abs(np.diff(ea))), mod, 1 - act.mean()])


def cv_auc(X: np.ndarray, y: np.ndarray, fo: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    pr = np.zeros(len(y))
    for k in range(5):
        te = fo % 5 == k
        s = StandardScaler().fit(X[~te])
        pr[te] = LogisticRegression(C=0.3, max_iter=3000).fit(s.transform(X[~te]), y[~te]).predict_proba(s.transform(X[te]))[:, 1]
    return float(roc_auc_score(y, pr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--conv", default="../results/conv_p0/d452m/conv", help="dev / devf の変換(p00* = 男声入力・p01* = 女声入力)")
    ap.add_argument("--vconv", default="../results/conv_p0/d452_vctkm/devvctkm/base", help="devvctkm の変換(p00* = 女声・p01* = 男声)")
    ap.add_argument("--perm", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(468)
    rep: dict = {}

    def run(name: str, items: list) -> None:
        X = np.array([feats(c, x) for c, x, _, _ in items])
        y = np.array([m for _, _, m, _ in items])
        us = sorted({s for *_, s in items})
        fo = np.array([us.index(s) for *_, s in items])
        obs = cv_auc(X, y, fo)
        null = [cv_auc(X, rng.permutation(y), fo) for _ in range(a.perm)]
        rep[name] = {"n": len(items), "auc": round(obs, 4), "null_mean": round(float(np.mean(null)), 4), "null_p95": round(float(np.percentile(null, 95)), 4)}
        print(name, rep[name], flush=True)

    tts_in, tts_out = [], []
    for s_, m in (("dev", 1), ("devf", 0)):
        for r in json.loads((P.H / s_ / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, _ = P.load_dump(P.H / "e2e" / f"eval_{s_}", k)
            tts_in.append((c, load48(P.H / s_ / "d56" / f"{k}_SRC.wav"), m, r["spk"]))
            yo = Path(a.conv) / f"p{r['pi'] + (0 if m else 100):04d}_t{r['ti']:03d}_CV.wav"
            if yo.exists():
                tts_out.append((c, load48(yo), m, r["spk"]))
    run("TTS_input", tts_in)
    run("TTS_output", tts_out)
    v_in, v_out = [], []
    rows = json.loads((P.H / "devvctkm" / "devset.json").read_text())["pairs"]
    for r in rows:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (P.H / "e2e" / "eval_devvctkm" / f"{k}.cond").exists():
            continue
        c, _ = P.load_dump(P.H / "e2e" / "eval_devvctkm", k)
        m = 1 if r["kind"].startswith("M") else 0
        src = P.H / "devvctkm" / "d56" / f"{k}_SRC.wav"
        if src.exists():
            v_in.append((c, load48(src), m, r["spk"]))
        yo = Path(a.vconv) / f"{k}_CV.wav"
        if yo.exists():
            v_out.append((c, load48(yo), m, r["spk"]))
    run("VCTK_input", v_in)
    run("VCTK_output", v_out)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
