"""V2-0a: 実音声 same-text ペアの DTW 点検（content 特徴上・アライメント分布と除外閾値の凍結用）。

VCTK same-text（id+text 完全一致・男女双方）から M-F ペアを作り、
ContentVec(50fps) 上で DTW。経路長比・正規化コスト・有効フレーム比の分布を出す。
推論経路に DTW は入らない（学習データ前処理専用）。

    uv run python v20a_dtw.py --wav-root /tmp/opencode/vctk_probe/VCTK-Corpus/VCTK-Corpus/wav48 \
        --cross /tmp/opencode/vctk_cross.json --out ../results/v20a
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from extract_content import load_contentvec, content_of

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CV_SR = 16000
FPS = 50


def dtw_path(C: np.ndarray, step=2) -> tuple[np.ndarray, float]:
    """subtw なし・step<=2（斜め/横/縦）の対称 DTW。経路と正規化コストを返す。"""
    n, m = C.shape
    INF = np.inf
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    P = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        Ci = C[i - 1]
        for j in range(1, m + 1):
            c = Ci[j - 1]
            best, bk = INF, 0
            for di, dj, tag in ((1, 1, 1), (1, 0, 2), (0, 1, 3)):
                v = D[i - di, j - dj]
                if v < best:
                    best, bk = v, tag
            D[i, j] = best + c
            P[i, j] = bk
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        tag = P[i, j]
        if tag == 1:
            i, j = i - 1, j - 1
        elif tag == 2:
            i -= 1
        elif tag == 3:
            j -= 1
        else:
            break
    path.reverse()
    return np.array(path), float(D[n, m] / max(n, m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-root", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--out", default="../results/v20a")
    ap.add_argument("--max-texts", type=int, default=6)
    ap.add_argument("--max-pairs-per-text", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    spi = Path(a.wav_root).parent / "speaker-info.txt"
    if not spi.exists():
        spi = Path(a.wav_root).parent.parent / "speaker-info.txt"
    genders = {}
    for line in open(spi):
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            genders["p" + p[0]] = p[2]

    cross = json.load(open(a.cross))
    by_uid = defaultdict(dict)
    for key, spks in cross.items():
        uid = key.split("|")[0]
        by_uid[uid][key] = spks

    model = load_contentvec()
    import librosa

    def cv_of(spk: str, uid: str) -> np.ndarray:
        w, _ = librosa.load(Path(a.wav_root) / spk / f"{spk}_{uid}.wav", sr=CV_SR, mono=True)
        return content_of(model, w).float().numpy()

    texts = sorted(by_uid, key=lambda u: -max(len(s) for s in by_uid[u].values()))
    sel_uids = (texts[: a.max_texts // 2]
                + texts[len(texts) // 2: len(texts) // 2 + a.max_texts // 2])
    rows = []
    cache: dict[str, np.ndarray] = {}

    def feat(spk, uid):
        k = f"{spk}_{uid}"
        if k not in cache:
            cache[k] = cv_of(spk, uid)
        return cache[k]

    for uid in sel_uids:
        for key, spks in by_uid[uid].items():
            M = [s for s in spks if genders.get(s) == "M"]
            F = [s for s in spks if genders.get(s) == "F"]
            pairs = list(product(M, F))
            if len(pairs) > a.max_pairs_per_text:
                idx = np.linspace(0, len(pairs) - 1, a.max_pairs_per_text).astype(int)
                pairs = [pairs[i] for i in idx]
            for m, f in pairs:
                try:
                    cm, cf = feat(m, uid), feat(f, uid)
                except FileNotFoundError:
                    continue
                cm = cm / (np.linalg.norm(cm, axis=-1, keepdims=True) + 1e-6)
                cf = cf / (np.linalg.norm(cf, axis=-1, keepdims=True) + 1e-6)
                C = 1.0 - cm @ cf.T
                path, cost = dtw_path(C)
                len_ratio = len(path) / max(cm.shape[0], cf.shape[0])
                durs = (cm.shape[0] / FPS, cf.shape[0] / FPS)
                rows.append(dict(male=m, female=f, uid=uid,
                                 dur_m=round(durs[0], 2), dur_f=round(durs[1], 2),
                                 len_ratio=round(len_ratio, 3), cost=round(cost, 4)))
        print(f"  uid {uid}: cumulative pairs {len(rows)}", flush=True)

    with open(out / "dtw_rows.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    lr = np.array([r["len_ratio"] for r in rows])
    co = np.array([r["cost"] for r in rows])
    print(f"\n  pairs {len(rows)}")
    print(f"  len_ratio p1/p50/p99: {np.quantile(lr,0.01):.3f} / {np.quantile(lr,0.5):.3f} / {np.quantile(lr,0.99):.3f}")
    print(f"  cost       p1/p50/p99: {np.quantile(co,0.01):.4f} / {np.quantile(co,0.5):.4f} / {np.quantile(co,0.99):.4f}")
    print(f"  in [0.7,1.4]: {( (lr>=0.7)&(lr<=1.4) ).mean():.3f}   cost<=0.35: {(co<=0.35).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
