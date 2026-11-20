"""段 P30(paper_snv §5 M33)の準備(学習なし): 段 D(data/snv_d・同じ文を 8 声が読む)の学習の声どうしの、同じ文の発話の時刻の対応を作る。
対応 = 発話ごとに標準化した c0〜c12 と有声の旗の上の DTW(帯 0.2・段 P29 a3d の p29_oracle.py と同じ)。源 A のフレーム i に、相手 B のフレーム j(i)。
組 = 文ごとに、その文を読んだ学習の声を輪に並べて隣どうし(A → 次の声)。符号の学習の正例の位置にだけ使う(音響の回帰の対にしない・当てる先にしない)。
出力 = data/snv_p_feat_dalign/align.npz(鍵 = "A の声|A の発話|B の声|B の発話"・値 = int16 の j)と manifest.json。
    cd training && uv run --no-sync python snv_p30_align.py [procs=10] [split=results/snv_p1_n/split.json]
"""
from __future__ import annotations

import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_p_feat"
OUT = ROOT / "data/snv_p_feat_dalign"
BAND = 0.2


def feat(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(p)
    f0, env = d["f0"], d["env"]
    y = env[:13].astype(np.float32)
    v = (f0 > 0).astype(np.float32)
    return np.concatenate([(y - y.mean(1, keepdims=True)) / np.maximum(y.std(1, keepdims=True), 1e-6), 2.0 * v[None]], 0), v


def align(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    _, wp = librosa.sequence.dtw(X=fa, Y=fb, metric="euclidean", global_constraints=True, band_rad=BAND)
    wp = wp[::-1]
    j = np.zeros(fa.shape[1], np.int64)
    for i in range(len(j)):
        q = wp[wp[:, 0] == i, 1]
        j[i] = int(np.median(q)) if len(q) else j[i - 1]
    return j


def job(k: tuple[str, str, str, str]) -> tuple[str, np.ndarray, float, float]:
    u, su, v, sv = k
    (fa, va), (fb, vb) = feat(FEAT / f"d/{u}/{su}.npz"), feat(FEAT / f"d/{v}/{sv}.npz")
    j = align(fa, fb)
    return "|".join(k), j.astype(np.int16), float(np.mean(vb[j][va > 0] == 0)), fa.shape[1] / fb.shape[1]


def pairs(split: Path) -> list[tuple[str, str, str, str]]:
    tr = set(json.loads(split.read_text())["train"])
    by: dict = {}
    for v in json.loads((ROOT / "data/snv_d/manifest.json").read_text())["voices"]:
        if v["voice"] in tr:
            for x in v["files"]:
                by.setdefault(x["content"], []).append((v["voice"], x["s"]))
    out = []
    for w in sorted(by):
        ps = by[w]
        if len(ps) >= 2:
            out += [(*ps[i], *ps[(i + 1) % len(ps)]) for i in range(len(ps) if len(ps) > 2 else 1)]
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    split = ROOT / opt.get("split", "results/snv_p1_n/split.json")
    P = pairs(split)
    print("組", len(P), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(int(opt.get("procs", 10))) as pool:
        res = pool.map(job, P, chunksize=32)
    np.savez_compressed(OUT / "align.npz", **{k: j for k, j, _, _ in res})
    man = {"date": subprocess.check_output(["date", "-u", "+%FT%TZ"]).decode().strip(), "source": "data/snv_p_feat/d(段 D・学習の声だけ)", "split": str(split.relative_to(ROOT)), "n_pairs": len(res),
           "method": "発話ごとに標準化した c0〜c12 + 有声の旗 × 2 の上の DTW(librosa・帯 0.2)。A のフレームごとに B のフレームの位置", "use": "符号の学習の正例の位置にだけ使う(paper_snv §5 M33)。音響の回帰の対・当てる先には使わない",
           "voicing_mismatch_mean": round(float(np.mean([m for _, _, m, _ in res])), 4), "len_ratio_p5_p50_p95": [round(float(np.percentile([r for _, _, _, r in res], q)), 3) for q in (5, 50, 95)]}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(man, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
