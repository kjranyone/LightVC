"""段 P113(results/snv_p113/prereg.yaml)の準備(学習なし): E1-NV(音素の無い発声)で、同じ種類・同じ発声の文を演じた学習の声どうしの時刻の対応を作る(snv_p30_align.py と同じ DTW)。
組 = (種類, 文)ごとに、学習の声を輪に並べて隣どうし(A → 次の声)。符号の学習の正例の位置にだけ使う(音響の回帰の対にしない・当てる先にしない)。
出力 = data/snv_p_feat_nvalign/align.npz(鍵 = "A の声|A の発話|B の声|B の発話")と manifest.json(対ごとの DTW の代価つき)。
    cd training && uv run --no-sync python snv_p113_nvalign.py [procs=10] [check=1]
"""
from __future__ import annotations

import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np

import snv_p30_align as A

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_p_feat/e1nv"
OUT = ROOT / "data/snv_p_feat_nvalign"


def cost(job: tuple) -> tuple:
    a, ua, b, ub = job
    fa, _ = A.feat(FEAT / a / f"{ua}.npz")
    fb, _ = A.feat(FEAT / b / f"{ub}.npz")
    D, wp = librosa.sequence.dtw(X=fa, Y=fb, metric="euclidean", global_constraints=True, band_rad=A.BAND)
    j = A.align(fa, fb)
    return job, j.astype(np.int16), float(D[-1, -1] / len(wp)), fa.shape[1], fb.shape[1]


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tr = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["train"]
    man = [x for x in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if x["voice"] in set(tr) and (FEAT / x["voice"] / f"{x['cat']}_{x['k']}.npz").exists()]
    groups: dict = {}
    for x in man:
        groups.setdefault((x["cat"], x["text_id"]), []).append((x["voice"], f"{x['cat']}_{x['k']}"))
    rng = random.Random(0)
    jobs, ctrl = [], []
    for (c, t), vs in sorted(groups.items()):
        vs = sorted(vs)
        rng.shuffle(vs)
        jobs += [(vs[i][0], vs[i][1], vs[(i + 1) % len(vs)][0], vs[(i + 1) % len(vs)][1]) for i in range(len(vs)) if len(vs) > 1]
        other = [q for (c2, t2), qs in groups.items() if c2 == c and t2 != t for q in qs]
        ctrl += [(vs[i][0], vs[i][1], *rng.choice(other)) for i in range(0, len(vs), 8)]
    print("組", len(jobs), "(種類, 文)の数", len(groups), "対照(同じ種類・別の文)", len(ctrl), flush=True)
    with Pool(int(opt.get("procs", 10))) as pool:
        res = pool.map(cost, jobs, chunksize=16)
        cres = pool.map(cost, ctrl, chunksize=16) if opt.get("check", "1") == "1" else []
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "align.npz", **{"|".join(j): al for j, al, _, _, _ in res})
    cs, cc = np.array([r[2] for r in res]), np.array([r[2] for r in cres]) if cres else np.zeros(1)
    lr = np.array([r[4] / r[3] for r in res])
    bycat = {c: [round(float(np.mean([r[2] for r in res if r[0][1].startswith(c)])), 3), round(float(np.mean([r[2] for r in cres if r[0][1].startswith(c)])), 3) if cres else None] for c in sorted({k[0] for k in groups})}
    info = {"stage": "P113", "pairs": len(res), "cost_same_text_mean": round(float(cs.mean()), 3), "cost_other_text_mean": round(float(cc.mean()), 3), "cost_same_text_p10_p50_p90": np.round(np.percentile(cs, [10, 50, 90]), 3).tolist(), "len_ratio_p10_p50_p90": np.round(np.percentile(lr, [10, 50, 90]), 2).tolist(), "by_cat_same_vs_other": bycat, "auc_same_lower": round(float((cs[:, None] < cc[None, :]).mean()), 3) if cres else None}
    (OUT / "manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
