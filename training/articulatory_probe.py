from __future__ import annotations

import sys
import argparse
import random
import collections
from pathlib import Path

import numpy as np
import librosa
import pyworld
import torch

SR = 16000
STYLES = ["neutral", "soft", "warm", "breathy", "low_tension",
          "cute_high", "intimate_close", "young_bright"]
MOE = ["soft", "breathy", "cute_high", "intimate_close", "young_bright"]


def style_name(s: str) -> str:
    p = s.split("_", 1)
    return p[1] if len(p) > 1 else s


def formant_track(x: np.ndarray, f0: np.ndarray, order: int = 12) -> np.ndarray:
    win = int(0.025 * SR)
    hop = int(SR * 0.005)
    out = []
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win] * np.hanning(win)
        pre = np.append(fr[0], fr[1:] - 0.97 * fr[:-1])
        try:
            a = librosa.lpc(pre.astype(np.float32), order=order)
            r = np.roots(a)
            r = r[np.imag(r) > 0]
            ang = np.sort(np.arctan2(np.imag(r), np.real(r)) * SR / (2 * np.pi))
            ang = ang[ang > 90]
            out.append(ang[:3] if len(ang) >= 3 else np.pad(ang, (0, 3 - len(ang)), constant_values=np.nan))
        except Exception:
            out.append(np.array([np.nan, np.nan, np.nan]))
    return np.array(out)


def artic_feats(path: str) -> dict:
    x, _ = librosa.load(path, sr=SR, mono=True)
    x = x.astype(np.float64)
    f0, t = pyworld.harvest(x, SR, frame_period=5.0)
    ft = formant_track(x, f0)
    if len(ft) < 8:
        return {}
    f1, f2, f3 = ft[:, 0], ft[:, 1], ft[:, 2]
    m = np.isfinite(f1) & np.isfinite(f2)
    if m.sum() < 8:
        return {}
    f1v, f2v = f1[m], f2[m]
    d = {}
    d["F1"] = np.median(f1v)
    d["F2"] = np.median(f2v)
    d["F2_F1"] = np.median(f2v / (f1v + 1e-6))
    d["vowel_space"] = float(np.std(f1v) * np.std(f2v) / 1e4)
    d["F2_range"] = float(np.percentile(f2v, 90) - np.percentile(f2v, 10))
    df2 = np.abs(np.diff(f2v))
    d["artic_velocity"] = float(np.median(df2))
    d["artic_dynamic"] = float(np.percentile(df2, 90))
    return d


def cohend(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 3 or len(b) < 3:
        return np.nan
    s = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    return (a.mean() - b.mean()) / s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--per", type=int, default=45)
    args = ap.parse_args()
    random.seed(0)
    files = list(Path(args.feat).rglob("*.pt"))
    random.shuffle(files)
    by = collections.defaultdict(list)
    byspk = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        d = torch.load(f, weights_only=False)
        st = style_name(d["style"])
        if st in STYLES and len(by[st]) < args.per:
            fe = artic_feats(d["path"])
            if fe:
                by[st].append(fe)
                byspk[d["speaker"]][st].append(fe)
        if all(len(by[s]) >= args.per for s in STYLES):
            break

    keys = ["F1", "F2", "F2_F1", "vowel_space", "F2_range", "artic_velocity", "artic_dynamic"]
    print("=== (1) 萌えは構音的に固有か: Cohen's d vs neutral ===")
    print(f"{'feature':14} " + " ".join(f"{s[:6]:>7}" for s in MOE) + "  |mean_d|")
    base = {k: [u[k] for u in by['neutral'] if k in u] for k in keys}
    rows = []
    for k in keys:
        ds = [cohend([u[k] for u in by[s] if k in u], base[k]) for s in MOE]
        rows.append((np.nanmean(np.abs(ds)), k, ds))
    for md, k, ds in sorted(rows, reverse=True):
        print(f"{k:14} " + " ".join(f"{x:+7.2f}" if not np.isnan(x) else f"{'--':>7}" for x in ds) + f"  {md:.2f}")

    print("\n=== (2) 構音は話者のクセか: 話者間分散/全分散 (1に近い=clone可能な癖) ===")
    for k in keys:
        spk_means, within = [], []
        for spk, sts in byspk.items():
            vals = [u[k] for st in sts for u in sts[st] if k in u]
            if len(vals) >= 5:
                spk_means.append(np.mean(vals))
                within.append(np.var(vals))
        if len(spk_means) >= 5:
            between = np.var(spk_means)
            wmean = np.mean(within)
            icc = between / (between + wmean + 1e-9)
            print(f"  {k:14} 話者間割合 {icc:.2f}  (n_spk={len(spk_means)})")
    print("\n話者間割合が高い特徴 = 話者固有の構音癖 = clone対象。萌えdが大 = その癖が萌えを作る。")


if __name__ == "__main__":
    main()
