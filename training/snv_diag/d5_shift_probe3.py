"""段 D5 の補足 3(results/snv_d5/prereg.yaml supplement_3・学習なし): 有声の判定の細かい切れが変換の内容を壊すか。有声だけを特徴の段で動かし、包絡と周期性は新しい高さで作り直す。
iro(検証の Irodori の男声 6 声 × t03・t04・登録 t05〜t08): clean / frag(有声のフレームに 10〜40 ms の無声の切れ目を入れ、有声の区間の長さの中央値を約 70 ms に)。
cv(devA の先頭 n 人 × s0・登録 e0〜e7): clean / smooth(40 ms 以下の無声の切れ目を高さの補間で埋め、20 ms 以下の有声の島を消す)/ self(構造 = 自分の登録)/ toM(目標 = 検証の Irodori の男声 2 声)。
jv(JVNV の M1・M2 の regular_01・登録 regular_02): clean / chain0・chain1(録音の鎖 v2)/ frag。
A0F の目標 2 声・k 0。CER = large-v3 の仮名で、加工しない元の音の書き起こし 対 出力。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_shift_probe3.py snv_i10_c_pr snv_i12_combo snv_p8_wst [n=24] [out=results/snv_d5/shift_probe3.json]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import snv_d5_chain as CH
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
SR = P.SR
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
PLAN = {"iro": ("clean", "frag"), "cv": ("clean", "smooth", "self", "toM"), "jv": ("clean", "chain0", "chain1", "frag")}
BASE = {"iro": ("clean", "frag"), "cv": ("clean", "smooth"), "jv": ("clean", "chain0", "chain1", "frag")}


def runs(v: np.ndarray) -> list:
    e = np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


def frag(f0: np.ndarray, g: np.random.Generator) -> np.ndarray:
    f = f0.copy()
    for a, b in runs(f0 > 0):
        t = a
        while True:
            t += int(np.clip(round(float(g.lognormal(np.log(14), 0.7))), 3, 80))
            gap = int(g.integers(2, 9))
            if t + gap >= b - 2:
                break
            f[t:t + gap] = 0.0
            t += gap
    return f


def smooth(f0: np.ndarray) -> np.ndarray:
    f = f0.copy()
    for a, b in runs(f0 <= 0):
        if a > 0 and b < len(f) and b - a <= 8:
            f[a:b] = np.exp(np.interp(np.arange(a, b), [a - 1, b], np.log([f[a - 1], f[b]])))
    for a, b in runs(f > 0):
        if b - a <= 4:
            f[a:b] = 0.0
    return f


def seg_med(f0s: list) -> float:
    return float(np.median([b - a for f in f0s for a, b in runs(f > 0)]) * 5)


def work(job: tuple) -> tuple:
    import librosa
    key, path, cond = job
    torch.set_num_threads(1)
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    if cond.startswith("chain"):
        x = CH.apply(x, CH.room(f"probe3_{key.split('|')[1]}", int(cond[5:])), np.random.default_rng(CH.seed_of(f"probe3|{key}|{cond}")))
    x = x[: P.n_frames(len(x)) * P.H]
    f = P.analyze(x)
    f0, env, per = f["f0"], f["env"], f["per"]
    if cond in ("frag", "smooth"):
        f0 = (frag(f0, np.random.default_rng(CH.seed_of(f"frag|{key}"))) if cond == "frag" else smooth(f0)).astype(np.float32)
        env = P.envelope(x, f0.astype(np.float64))
        per = P.periodicity(torch.from_numpy(x), torch.from_numpy(f0)).numpy()
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return (key, cond), {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": x}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a] or ["snv_i10_c_pr", "snv_i12_combo", "snv_p8_wst"]
    n = int(opt.get("n", 24))
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    vm = [x for x in split["val"] if x.startswith("em")]
    items = [{"set": "iro", "spk": v, "src": [f"e1|{v}|{t}" for t in ("t03", "t04")], "enr": [f"e1|{v}|{t}" for t in ("t05", "t06", "t07", "t08")]} for v in vm]
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    items += [{"set": "cv", "spk": s["speaker"], "src": [f"cv|{s['speaker']}|s0"], "enr": [f"cv|{s['speaker']}|e{i}" for i in range(8)]} for s in [x for x in man["speakers"] if x["split"] == "devA"][:n]]
    items += [{"set": "jv", "spk": sp, "src": [f"jv|{sp}|{e}_regular_01" for e in EMOS], "enr": [f"jv|{sp}|{e}_regular_02" for e in EMOS]} for sp in ("M1", "M2")]

    def path(k: str) -> str:
        kind, a, b = k.split("|")
        if kind == "e1":
            return str(ROOT / f"data/snv_e1/{a}/{b}.wav")
        if kind == "cv":
            return str(T.FEAT / f"cv/{a}_{b}.wav")
        return str(ROOT / f"data/jvnv/jvnv_v1/{a}/{b.split('_')[0]}/regular/{a}_{b}.wav")

    jobs = [(k, path(k), c) for it in items for k in it["src"] + it["enr"] for c in BASE[it["set"]]]
    U: dict = {}
    with ProcessPoolExecutor(11) as ex:
        for key, u in ex.map(work, jobs, chunksize=2):
            U[key] = u
    res: dict = {"prereg": "results/snv_d5/prereg.yaml supplement_3", "n_cv": n, "systems": {}, "rows": [],
                 "voiced_seg_median_ms": {f"{st}/{c}": seg_med([U[(k, c)]["f0"] for it in items if it["set"] == st for k in it["src"] + it["enr"]]) for st, cs in BASE.items() for c in cs}}
    print("特徴", len(U), "有声の区間の長さの中央値(ms)", res["voiced_seg_median_ms"], flush=True)
    import snv_p1_eval as EV
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    DEV = EV.DEV
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    w16 = lambda w: to16(w.astype(np.float64), SR).astype(np.float32)

    def tr(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    mrefs = {v: [EV.utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)] for v in vm[:2]}
    ref = {k: tr(w16(U[(k, "clean")]["wav"])) for it in items for k in it["src"]}
    for tag in tags:
        s = EV.System(tag, gen)
        o: dict = {}
        for it in items:
            for cond in PLAN[it["set"]]:
                fc = cond if cond in BASE[it["set"]] else "clean"
                enr = [U[(k, fc)] for k in it["enr"]]
                for k in it["src"]:
                    if cond == "self":
                        Ss = [(None, s.structure(f"self|{it['spk']}|{fc}", enr))]
                    elif cond == "toM":
                        Ss = [(v, s.structure(f"m|{v}", mrefs[v])) for v in mrefs]
                    else:
                        Ss = [(v, s.structure(f"t|{v}", trefs[v])) for v in tgts]
                    for v, S in Ss:
                        e = cer(ref[k], tr(w16(s.convert(U[(k, fc)], enr, S, 0))))
                        res["rows"].append({"sys": tag, "set": it["set"], "cond": cond, "spk": it["spk"], "src": k, "tgt": v, "cer": round(e, 4)})
        for st, conds in PLAN.items():
            for cond in conds:
                cs = [r["cer"] for r in res["rows"] if r["sys"] == tag and r["set"] == st and r["cond"] == cond]
                o[f"{st}/{cond}"] = {"median": round(float(np.median(cs)), 4), "mean": round(float(np.mean(cs)), 4), "n": len(cs)}
        res["systems"][tag] = o
        print(tag, json.dumps({c: d["median"] for c, d in o.items()}, ensure_ascii=False), flush=True)
        del s
        torch.cuda.empty_cache()
    (ROOT / opt.get("out", "results/snv_d5/shift_probe3.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
