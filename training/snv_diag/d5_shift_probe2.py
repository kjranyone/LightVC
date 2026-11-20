"""段 D5 の補足 2(results/snv_d5/prereg.yaml supplement_2・学習なし): 高さの外れが壊すのは推論の E の側か C の側か、本物の声を族の高さへ動かすと戻るか。
iro: 検証の Irodori の男声 6 声 × t03・t04(登録 t05〜t08)。条件 wcopy / wlow(高さ × 0.7)/ wlow_E(E が読む包絡・周期性・有声だけ wlow・C と描画器の高さは wcopy)/ wlow_C(その逆)。
cv: 開発の組(cv の devA の先頭 n 人 × s0・登録 e0〜e7)。条件 clean / wcopy / whigh(高さ × 1.4)/ whigh_E / whigh_C。
→ A0F の目標 2 声・k 0。CER = large-v3 の仮名で、加工しない元の音の書き起こし 対 出力。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_shift_probe2.py snv_i10_c_pr snv_i12_combo snv_p8_wst [n=24] [out=results/snv_d5/shift_probe2.json]
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
import snv_d5_world as W
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
SR = P.SR
RATIO = {"clean": None, "wcopy": 1.0, "wlow": 0.7, "whigh": 1.4}
PLAN = {"iro": ("wcopy", "wlow", "wlow_E", "wlow_C"), "cv": ("clean", "wcopy", "whigh", "whigh_E", "whigh_C")}


def work(job: tuple) -> tuple:
    import librosa
    key, path, cond = job
    torch.set_num_threads(1)
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    if RATIO[cond] is not None:
        x = W.shift(x, SR, RATIO[cond])
    f = P.analyze(x[: P.n_frames(len(x)) * P.H])
    f0, env, per = f["f0"], f["env"], f["per"]
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return (key, cond), {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": x}


def mix(e: dict, c: dict) -> dict:
    n = min(len(e["v"]), len(c["v"]))
    v = e["v"][:n] * c["v"][:n]
    y = e["y"][:, :n].copy()
    y[25:29] *= v[None]
    return {"y": y, "v": v, "lf": c["lf"][:n] * v, "f0": c["f0"][:n] * v, "vs": e["vs"], "vn": e["vn"], "wav": e["wav"]}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a] or ["snv_i10_c_pr", "snv_i12_combo", "snv_p8_wst"]
    n = int(opt.get("n", 24))
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    items = []
    for v in [x for x in split["val"] if x.startswith("em")]:
        items.append({"set": "iro", "spk": v, "src": [f"e1|{v}|{t}" for t in ("t03", "t04")], "enr": [f"e1|{v}|{t}" for t in ("t05", "t06", "t07", "t08")]})
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    for s in [x for x in man["speakers"] if x["split"] == "devA"][:n]:
        items.append({"set": "cv", "spk": s["speaker"], "src": [f"cv|{s['speaker']}|s0"], "enr": [f"cv|{s['speaker']}|e{i}" for i in range(8)]})
    path = lambda k: str(ROOT / f"data/snv_e1/{k.split('|')[1]}/{k.split('|')[2]}.wav") if k.startswith("e1|") else str(T.FEAT / f"cv/{k.split('|')[1]}_{k.split('|')[2]}.wav")
    base = {"iro": ("clean", "wcopy", "wlow"), "cv": ("clean", "wcopy", "whigh")}
    jobs = [(k, path(k), c) for it in items for k in it["src"] + it["enr"] for c in base[it["set"]]]
    U: dict = {}
    with ProcessPoolExecutor(11) as ex:
        for key, u in ex.map(work, jobs, chunksize=2):
            U[key] = u
    print("特徴", len(U), flush=True)

    def get(k: str, cond: str) -> dict:
        if cond.endswith("_E"):
            return mix(U[(k, cond[:-2])], U[(k, "wcopy")])
        if cond.endswith("_C"):
            return mix(U[(k, "wcopy")], U[(k, cond[:-2])])
        return U[(k, cond)]

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
    ref = {k: tr(w16(U[(k, "clean")]["wav"])) for it in items for k in it["src"]}
    res: dict = {"prereg": "results/snv_d5/prereg.yaml supplement_2", "n_cv": n, "systems": {}, "rows": []}
    for tag in tags:
        s = EV.System(tag, gen)
        o: dict = {}
        for it in items:
            for cond in PLAN[it["set"]]:
                ec = cond if not cond.endswith("_C") else "wcopy"
                enr = [get(k, ec if not ec.endswith("_E") else ec[:-2]) for k in it["enr"]]
                for k in it["src"]:
                    for tg in tgts:
                        e = cer(ref[k], tr(w16(s.convert(get(k, cond), enr, s.structure(f"t|{tg}", trefs[tg]), 0))))
                        res["rows"].append({"sys": tag, "set": it["set"], "cond": cond, "spk": it["spk"], "src": k, "tgt": tg, "cer": round(e, 4)})
        for st, conds in PLAN.items():
            for cond in conds:
                cs = [r["cer"] for r in res["rows"] if r["sys"] == tag and r["set"] == st and r["cond"] == cond]
                o[f"{st}/{cond}"] = {"median": round(float(np.median(cs)), 4), "mean": round(float(np.mean(cs)), 4), "n": len(cs)}
        res["systems"][tag] = o
        print(tag, json.dumps({c: d["median"] for c, d in o.items()}, ensure_ascii=False), flush=True)
        del s
        torch.cuda.empty_cache()
    (ROOT / opt.get("out", "results/snv_d5/shift_probe2.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
