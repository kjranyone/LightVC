"""A3e の事前検査(学習なし): E1 の中の同じ文を別の声で読んだ組を、各声の統計で白色化した包絡の DTW で揃えたとき、揃いが調音の状態の対応になっているか。
揃いの質 = 揃えた相手のフレームの SPARC の EMA(物差しとしてだけ)と自分の EMA の相関(12 次元の平均)。対照 = 等速の伸縮(線形の揃い)・EMA 自身の DTW(物差しの天井・学習には使わない)。
特徴 = raw(全体の平均・標準偏差で標準化)/ cmn(声ごとの平均を引く)/ white(声ごとの平均・共分散で白色化 = A3c の差し替えの統計)。組は学習の声どうしだけ(検証 30 声を除く)。
    cd training && uv run --no-sync python snv_diag/a3e_pair_dtw.py
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_snv_a2 import FEAT, ROOT, VAL_VOICES, playing
from train_snv_a3 import envelope, load_corpus, sqrtm_pair

DEV = "cuda"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(a.shape[1])]))


def main() -> int:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    voices = load_corpus()
    val = set(sorted(voices)[-VAL_VOICES:])
    sex = {v["voice"]: v["sex"] for v in man["voices"]}
    idx = {}
    for v in man["voices"]:
        if v["voice"] not in voices or v["voice"] in val:
            continue
        k = 0
        for f in v["files"]:
            if (FEAT / f"mel/e1/{v['voice']}/{f['t']}.npy").exists() and (FEAT / f"sparc/e1/{v['voice']}/{f['t']}.npz").exists() and (FEAT / f"wav44/e1/{v['voice']}/{f['t']}.npy").exists():
                idx[(v["voice"], f["t"])] = (k, f["text"])
                k += 1
    by = defaultdict(list)
    for key, (_, text) in idx.items():
        by[text].append(key)
    pairs = [(a, b) for us in by.values() for a, b in itertools.combinations(us, 2) if a[0] != b[0]]
    names = sorted({k[0] for p in pairs for k in p})
    print("pairs", len(pairs), "cross-sex", sum(sex[a[0]] != sex[b[0]] for a, b in pairs), "voices", len(names), flush=True)

    env, ema, st = {}, {}, {}
    with torch.no_grad():
        for n in names:
            es = [envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(DEV))[0].T for _, _, w in voices[n]]
            X = torch.cat(es)
            cs, ci = sqrtm_pair(torch.cov(X.T))
            st[n] = (X.mean(0), ci)
            env[n] = es
    allx = torch.cat([torch.cat(env[n]) for n in names])
    gmu, gsd = allx.mean(0), allx.std(0)
    for key, (k, _) in idx.items():
        if key[0] in env:
            n_mel = voices[key[0]][k][0].shape[1]
            ema[key] = playing(FEAT / f"sparc/e1/{key[0]}/{key[1]}.npz", n_mel)[:, :12]

    def feat(key, kind: str) -> np.ndarray:
        x = env[key[0]][idx[key][0]]
        if kind == "raw":
            y = (x - gmu) / gsd
        elif kind == "cmn":
            y = (x - st[key[0]][0]) / gsd
        else:
            y = (x - st[key[0]][0]) @ st[key[0]][1]
        return y.cpu().numpy().astype(np.float64)

    res = defaultdict(lambda: defaultdict(list))
    for a, b in pairs:
        ea, eb = ema[a], ema[b]
        cross = "cross" if sex[a[0]] != sex[b[0]] else "same"
        lin = np.minimum(np.round(np.arange(len(ea)) * len(eb) / len(ea)).astype(int), len(eb) - 1)
        res["linear"][cross].append(corr(ea, eb[lin]))
        for kind in ("raw", "cmn", "white", "ema_ceiling"):
            if kind == "ema_ceiling":
                X, Y = ((ea - ea.mean(0)) / (ea.std(0) + 1e-6)).T, ((eb - eb.mean(0)) / (eb.std(0) + 1e-6)).T
            else:
                X, Y = feat(a, kind).T, feat(b, kind).T
            _, wp = librosa.sequence.dtw(X=X, Y=Y, metric="cosine" if kind != "ema_ceiling" else "euclidean")
            wp = wp[::-1]
            res[kind][cross].append(corr(ea[wp[:, 0]], eb[wp[:, 1]]))
    out = {k: {c: round(float(np.mean(v)), 3) for c, v in d.items()} | {"n": {c: len(v) for c, v in d.items()}} for k, d in res.items()}
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
