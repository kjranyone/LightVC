"""A0c(A0b の不合格の分岐・学習なし): 事前登録 = results/snv_a0c/prereg.yaml。
ID = ecapa+lda(A0b の選択)を約 15 s の検査で・INT = 雑音ボコーダと包絡の時間のぼかしで large-v3 の CER と avg_logprob を較正。
    cd training && uv run --no-sync python snv_a0c_measure.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyworld as pw
import torch
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from snv_a0_measure import DEV, ROOT, SR, Embedders, cer, eer, kana, load, match_rms, to16
from snv_a0b_measure import fit_lda, proj, splits

OUT = ROOT / "results/snv_a0c"
A0 = ROOT / "results/snv_a0"
CHANNELS = (16, 8, 4, 2)
SMEARS = (40, 100, 200)


def vocode(y16: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    edges = np.geomspace(80, 7000, n + 1)
    rng = np.random.default_rng(seed)
    env_lp = butter(4, 50, fs=SR, output="sos")
    out = np.zeros_like(y16, dtype=np.float64)
    for lo, hi in zip(edges[:-1], edges[1:]):
        bp = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
        xb = sosfiltfilt(bp, y16.astype(np.float64))
        env = np.maximum(sosfiltfilt(env_lp, np.abs(xb)), 0.0)
        nb = sosfiltfilt(bp, rng.standard_normal(len(y16)))
        ob = env * nb
        ob *= np.sqrt(np.mean(xb ** 2) / max(np.mean(ob ** 2), 1e-12))
        out += ob
    return match_rms(out, y16.astype(np.float64)).astype(np.float32)


def smear_job(args: tuple) -> dict:
    y, sr, ws = args
    f0, t = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=1000.0, frame_period=5.0)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    out = {}
    for w in ws:
        lsp = uniform_filter1d(np.log(sp + 1e-16), size=max(1, w // 5), axis=0, mode="nearest")
        out[w] = match_rms(pw.synthesize(f0, np.exp(lsp), ap, sr, 5.0), y)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"prereg": "results/snv_a0c/prereg.yaml"}
    sp = splits()
    pool = sp["pool"]
    emb = Embedders()
    cache: dict = {}

    def e_files(ps: list) -> np.ndarray:
        k = "+".join(map(str, ps))
        if k not in cache:
            cache[k] = emb(np.concatenate([to16(*load(p)) for p in ps]))["ecapa"]
        return cache[k]

    X, y = [], []
    for i, s in enumerate(sp["lda"]):
        for p in pool[s]:
            X.append(e_files([p]))
            y.append(i)
    T = fit_lda(np.stack(X), np.array(y))
    print("lda fitted", flush=True)

    def id_eval(sets: dict, t: tuple | None) -> dict:
        E = {v: proj(e_files(enr), t) for v, (enr, _) in sets.items()}
        top, pos, neg = [], [], []
        for v, (_, tests) in sets.items():
            for grp in tests:
                e = proj(e_files(grp), t)
                sc = {u: float(e @ E[u]) for u in E}
                top.append(max(sc, key=sc.get) == v)
                pos.append(sc[v])
                neg += [s for u, s in sc.items() if u != v]
        return {"top1": float(np.mean(top)), "eer": eer(np.array(pos), np.array(neg)), "n": len(top),
                "own_mean": float(np.mean(pos)), "other_mean": float(np.mean(neg))}

    its = json.loads((A0 / "wav/items.json").read_text())
    fv = sorted({it["voice"] for it in its if it["sex"] == "F"})
    w = lambda v, j: A0 / f"wav/{v}_{j}.wav"
    s1 = {v: ([w(v, j) for j in (0, 1, 2)], [[w(v, j) for j in g] for g in ((3, 4, 5), (8, 9, 10), (11, 12, 13))]) for v in fv}
    s3 = {s: (pool[s][:3], [pool[s][3:6]]) for s in sp["S3"]}
    res["ID"] = {"S1_15s_ecapa+lda": id_eval(s1, T), "S1_15s_ecapa": id_eval(s1, None), "S3_15s_ecapa+lda": id_eval(s3, T), "S3_15s_ecapa": id_eval(s3, None)}
    print("ID", json.dumps(res["ID"], indent=1), flush=True)
    del emb
    torch.cuda.empty_cache()

    ftest = [it for it in its if it["sex"] == "F" and it["role"] == "test"]
    raw = [load(w(it["voice"], it["j"])) for it in ftest]
    with ProcessPoolExecutor(10) as ex:
        sm = list(ex.map(smear_job, [(yy, sr, SMEARS) for yy, sr in raw]))
    print("smear done", flush=True)
    conds = {"clean": [to16(yy, sr) for yy, sr in raw]}
    for n in CHANNELS:
        conds[f"voc{n}"] = [vocode(to16(yy, sr), n) for yy, sr in raw]
    for ww in SMEARS:
        conds[f"smear{ww}"] = [to16(d[ww], sr) for d, (_, sr) in zip(sm, raw)]

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")
    per = {}
    for c, ys in conds.items():
        cs, lp, hyps = [], [], []
        for it, y16 in zip(ftest, ys):
            segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
            segs = list(segs)
            hyp = "".join(s.text for s in segs)
            hyps.append(hyp)
            cs.append(cer(kana(it["text"]), kana(hyp)))
            lp.append(float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0)
        per[c] = {"cer": np.array(cs), "lp": np.array(lp), "hyp": hyps[:3]}
        print("INT", c, f"cer_median {np.median(cs):.3f} cer_mean {np.mean(cs):.3f} lp_mean {np.mean(lp):.4f}", hyps[0], flush=True)
    base = per["clean"]
    res["INT"] = {c: {"cer_median": float(np.median(v["cer"])), "cer_mean": float(np.mean(v["cer"])), "lp_mean": float(np.mean(v["lp"])),
                      "win_clean_cer": float(np.mean(base["cer"] < v["cer"])), "win_clean_lp": float(np.mean(base["lp"] > v["lp"])), "hyp_examples": v["hyp"]}
                  for c, v in per.items()}
    (OUT / "a0c_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
