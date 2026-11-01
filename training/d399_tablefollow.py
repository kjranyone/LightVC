"""D399: 生成器が目標の表(声質)にどこまで従うか — 目標ごとの帯域の釣り合い(5 帯の相対 dB)を、目標の参照の実録音と変換の出力で比べ、目標をまたいだ回帰の傾き(1 = 完全に従う・0 = 全目標で同じ)。
d396b で VCTK の目標に対し出力が 1〜3 kHz で +8 dB 明るかった(日本語では −1〜−3 dB)= 表から外れた声質を学習の分布の平均へ寄せる疑い。CPU だけ。
    uv run python d399_tablefollow.py --out ../results/conv_p0/d399.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

R = P.ROOT / "results/conv_p0"
H = P.H
BANDS = [(80, 300), (300, 1000), (1000, 3000), (3000, 8000), (8000, 20000)]
SYS = {
    "B": {"dev": f"{R}/d374m/conv/p00*_CV.wav", "dev2": f"{R}/d374/dev2/PPG27/p00*.wav", "dev3": f"{R}/d374/dev3/PPG27/p00*.wav"},
    "A": {"dev": f"{R}/d352m/conv/p00*_CV.wav", "dev2": f"{R}/d353/dev2/PPG24/p00*.wav", "dev3": f"{R}/d353/dev3/PPG24/p00*.wav"},
    "PROD": {s: f"{H}/{s}/d114/p00*_G03S10ZH.wav" for s in ("dev", "dev2", "dev3")},
    "KNN": {s: f"{H}/{s}/d135_real/p00*_KNN.wav" for s in ("dev", "dev2", "dev3")},
    "SRC": {s: f"{H}/{s}/d56/p00*_SRC.wav" for s in ("dev", "dev2", "dev3")},
}


def prof(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    f, _, Z = stft(x, sr, nperseg=2048, noverlap=2048 - 480)
    P_ = np.abs(Z) ** 2
    fe = P_.sum(0)
    keep = fe > np.percentile(fe, 50)
    e = P_[:, keep].mean(1)
    b = np.array([e[(f >= a) & (f < c)].sum() for a, c in BANDS])
    return 10 * np.log10(b / b.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="足す系: NAME=<dev の glob>|<dev2 の glob>|<dev3 の glob>")
    ap.add_argument("--only", nargs="*", default=[], help="この系だけ")
    a = ap.parse_args()
    for e in a.extra:
        nm, pats = e.split("=", 1)
        SYS[nm] = dict(zip(("dev", "dev2", "dev3"), pats.split("|")))
    if a.only:
        keep = set(a.only) | {e.split("=", 1)[0] for e in a.extra}
        for k in [k for k in SYS if k not in keep]:
            del SYS[k]
    rep: dict = {}
    refs: dict = {}
    keep: dict = {}
    for s in ("dev", "dev2", "dev3"):
        for f in sorted((H / s / "d37" / "_work").glob("ref_*.wav")):
            refs[(s, int(f.stem[4:7]))] = prof(f)
    for name, pats in SYS.items():
        X, Y = [], []
        for s, pat in pats.items():
            by: dict = {}
            for f in sorted(glob.glob(pat)):
                ti = int(Path(f).name.split("_t")[1][:3])
                by.setdefault(ti, []).append(prof(f))
            for ti, v in by.items():
                if (s, ti) in refs:
                    X.append(refs[(s, ti)])
                    Y.append(np.mean(v, 0))
        keys = [(s, ti) for s, pat in pats.items() for ti in sorted({int(Path(f).name.split("_t")[1][:3]) for f in glob.glob(pat)}) if (s, ti) in refs]
        X, Y = np.array(X), np.array(Y)
        keep[name] = dict(zip(keys, Y))
        out = {"n_targets": len(X)}
        for k, (lo, hi) in enumerate(BANDS):
            x, y = X[:, k] - X[:, k].mean(), Y[:, k] - Y[:, k].mean()
            out[f"{lo}-{hi}"] = {"slope": round(float((x * y).sum() / (x ** 2).sum()), 3), "r": round(float(np.corrcoef(x, y)[0, 1]), 3), "bias_db": round(float((Y[:, k] - X[:, k]).mean()), 2), "sd_ref": round(float(X[:, k].std()), 2), "sd_out": round(float(Y[:, k].std()), 2)}
        rep[name] = out
        print(name, json.dumps(out, ensure_ascii=False), flush=True)
    rng = np.random.default_rng(399)
    if "B" in keep:
        for name in keep:
            if name == "B":
                continue
            ks = sorted(set(keep[name]) & set(keep["B"]))
            Xr = np.array([refs[k] for k in ks]); Ya = np.array([keep[name][k] for k in ks]); Yb = np.array([keep["B"][k] for k in ks])
            def slope(ix, Y, k):
                x, y = Xr[ix, k] - Xr[ix, k].mean(), Y[ix, k] - Y[ix, k].mean()
                return float((x * y).sum() / (x ** 2).sum())
            out = {}
            for k, (lo, hi) in enumerate(BANDS):
                d0 = slope(np.arange(len(ks)), Ya, k) - slope(np.arange(len(ks)), Yb, k)
                bs = []
                for _ in range(1000):
                    ix = rng.integers(0, len(ks), len(ks))
                    bs.append(slope(ix, Ya, k) - slope(ix, Yb, k))
                out[f"{lo}-{hi}"] = {"dslope": round(d0, 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
            rep[f"{name}-B"] = out
            print(name, "− B の傾きの差(目標のブートストラップ)", json.dumps(out), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
