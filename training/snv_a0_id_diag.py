"""A0 の ID 不合格の切り分け(診断・合否は変えない): 物差しの分解能か、声そのものの近さか。
S1 = A0 の Irodori v4.1 女声(登録 3・検査 素 3)・S2 = 同じ声の v4 Large の元(pair_tts_v4l の t01〜t03 登録・t04〜t06 検査)・S3 = 本物の日本語の女声 24 人(female-dataset・登録 3・検査 3)。
    cd training && uv run --no-sync python snv_a0_id_diag.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from snv_a0_measure import ROOT, SR, Embedders, load, to16, eer

OUT = ROOT / "results/snv_a0"
PAIR = ROOT / "data/pair_tts_v4l"


def real_female(n: int, seed: int) -> dict:
    rows = list(csv.DictReader(open(ROOT / "data/kansei_vc/manifests/canonical_utterances.tsv"), delimiter="\t"))
    by = defaultdict(list)
    for r in rows:
        if r["source_type"] == "real_female" and 3.0 <= float(r["duration_sec"] or 0) <= 12.0:
            by[r["speaker_id"]].append(ROOT / "training" / r["wav_path"])
    spk = sorted(s for s, v in by.items() if len(v) >= 6)
    rng = random.Random(seed)
    pick = rng.sample(spk, n)
    return {s: sorted(by[s])[:6] for s in pick}


def evaluate(emb: Embedders, sets: dict) -> dict:
    E, T = {}, []
    for v, (enr, tests) in sets.items():
        ys = [load(p) for p in enr]
        E[v] = emb(np.concatenate([to16(y, sr) for y, sr in ys]))
        for p in tests:
            y, sr = load(p)
            T.append((v, emb(to16(y, sr))))
    out = {}
    for m in ("ecapa", "wavlm_sv"):
        top, pos, neg, conf = [], [], [], Counter()
        for v, e in T:
            sc = {u: float(e[m] @ E[u][m]) for u in E}
            best = max(sc, key=sc.get)
            top.append(best == v)
            if best != v:
                conf[(v, best)] += 1
            pos.append(sc[v])
            neg += [s for u, s in sc.items() if u != v]
        ee = np.array([[float(E[a][m] @ E[b][m]) for b in E] for a in E])
        off = ee[~np.eye(len(E), dtype=bool)]
        out[m] = {"top1": float(np.mean(top)), "eer": eer(np.array(pos), np.array(neg)), "n": len(T),
                  "own_mean": float(np.mean(pos)), "other_mean": float(np.mean(neg)),
                  "enroll_offdiag_p50_p90_max": [float(np.percentile(off, 50)), float(np.percentile(off, 90)), float(off.max())],
                  "confusions": [[a, b, c] for (a, b), c in conf.most_common(12)]}
    return out


def main() -> int:
    its = json.loads((OUT / "wav/items.json").read_text())
    fv = sorted({it["voice"] for it in its if it["sex"] == "F"})
    s1 = {v: ([OUT / f"wav/{v}_{j}.wav" for j in (0, 1, 2)], [OUT / f"wav/{v}_{j}.wav" for j in (3, 4, 5)]) for v in fv}
    s2 = {v: ([PAIR / v / f"t{k:02d}.wav" for k in (1, 2, 3)], [PAIR / v / f"t{k:02d}.wav" for k in (4, 5, 6)]) for v in fv}
    rf = real_female(24, 0)
    s3 = {s: (ps[:3], ps[3:6]) for s, ps in rf.items()}
    emb = Embedders()
    res = {"note": "診断。A0 の合否(prereg)は変えない", "S1_irodori_v41": evaluate(emb, s1), "S2_irodori_v4L_same_voices": evaluate(emb, s2),
           "S3_real_female_24": evaluate(emb, s3), "S3_speakers": list(rf)}
    man = json.loads((PAIR / "manifest.json").read_text())
    res["S1_confusion_captions"] = [[a, b, n, man["speakers"][a]["caption"][:40], man["speakers"][b]["caption"][:40]] for a, b, n in res["S1_irodori_v41"]["ecapa"]["confusions"]]
    (OUT / "a0_id_diag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k in ("S1_irodori_v41", "S2_irodori_v4L_same_voices", "S3_real_female_24"):
        print(k, {m: {x: res[k][m][x] for x in ("top1", "eer", "own_mean", "other_mean", "enroll_offdiag_p50_p90_max")} for m in res[k]})
    for row in res["S1_confusion_captions"]:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
