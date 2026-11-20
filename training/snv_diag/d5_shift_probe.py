"""段 D5 の補足 1(results/snv_d5/prereg.yaml supplement_1・学習なし): 族の中の声に本物の条件を 1 つずつ足して、台帳の軸(高さ・録音・速さ)が変換の内容をどれだけ壊すかを測る。
源 = 検証の Irodori の男声 6 声 × t03・t04(登録 = t05〜t08 に同じ加工)→ A0F の目標 2 声・k 0。CER = large-v3 の仮名で、加工しない元の音の書き起こし 対 出力。
条件: clean / wcopy(WORLD の分析合成だけ)/ wlow(高さ × 0.7)/ wlow2(× 0.6)/ chain(録音の鎖 v1 の場 3 つ)/ wlow+chain / fast(WSOLA 1.25 倍速)/ wlow+chain+fast。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_shift_probe.py snv_i10_c_pr snv_i12_combo snv_p8_wst snv_p3_x [out=results/snv_d5/shift_probe.json]
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
import snv_d5_world as W
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
SR = P.SR
CONDS = ("clean", "wcopy", "wlow", "wlow2", "chain0", "chain1", "chain2", "wlow+chain0", "wlow+chain1", "wlow+chain2", "fast", "wlow+chain0+fast")


def process(x: np.ndarray, cond: str, voice: str, key: str) -> np.ndarray:
    from i15_rate_probe import wsola
    for op in cond.split("+"):
        if op == "wcopy":
            x = W.shift(x, SR, 1.0)
        elif op == "wlow":
            x = W.shift(x, SR, 0.7)
        elif op == "wlow2":
            x = W.shift(x, SR, 0.6)
        elif op.startswith("chain"):
            x = CH.apply(x, CH.room(f"probe_{voice}", int(op[5:])), np.random.default_rng(CH.seed_of(f"probe|{key}|{cond}")))
        elif op == "fast":
            x = wsola(x.astype(np.float64), 1.25).astype(np.float32)
    return x.astype(np.float32)


def work(job: tuple) -> tuple:
    import librosa
    voice, t, cond = job
    torch.set_num_threads(1)
    x = process(librosa.load(str(ROOT / f"data/snv_e1/{voice}/{t}.wav"), sr=SR, mono=True)[0].astype(np.float32), cond, voice, f"{voice}/{t}")
    f = P.analyze(x[: P.n_frames(len(x)) * P.H])
    f0, env, per = f["f0"], f["env"], f["per"]
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return (voice, t, cond), {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": x}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a] or ["snv_i10_c_pr", "snv_i12_combo", "snv_p8_wst", "snv_p3_x"]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    voices = [v for v in split["val"] if v.startswith("em")]
    jobs = [(v, t, c) for v in voices for t in ("t03", "t04", "t05", "t06", "t07", "t08") for c in CONDS]
    U: dict = {}
    with ProcessPoolExecutor(11) as ex:
        for key, u in ex.map(work, jobs, chunksize=2):
            U[key] = u
    print("特徴", len(U), flush=True)
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
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    ref = {(v, t): tr(w16(U[(v, t, "clean")]["wav"])) for v in voices for t in ("t03", "t04")}
    f0m = {c: round(float(np.median(np.concatenate([U[(v, t, c)]["f0"][U[(v, t, c)]["f0"] > 0] for v in voices for t in ("t03", "t04")]))), 1) for c in CONDS}
    res: dict = {"prereg": "results/snv_d5/prereg.yaml supplement_1", "voices": voices, "f0_median_hz": f0m, "systems": {}, "rows": []}
    for tag in tags:
        s = EV.System(tag, gen)
        o = {}
        for c in CONDS:
            cs = []
            for v in voices:
                enr = [U[(v, t, c)] for t in ("t05", "t06", "t07", "t08")]
                for t in ("t03", "t04"):
                    for tg in tgts:
                        e = cer(ref[(v, t)], tr(w16(s.convert(U[(v, t, c)], enr, s.structure(f"t|{tg}", trefs[tg]), 0))))
                        cs.append(e)
                        res["rows"].append({"sys": tag, "cond": c, "voice": v, "t": t, "tgt": tg, "cer": round(e, 4)})
            o[c] = {"median": round(float(np.median(cs)), 4), "mean": round(float(np.mean(cs)), 4), "n": len(cs)}
        g = lambda names: round(float(np.median([r["cer"] for r in res["rows"] if r["sys"] == tag and r["cond"] in names])), 4)
        o["chain_all"], o["wlow+chain_all"] = {"median": g(("chain0", "chain1", "chain2"))}, {"median": g(("wlow+chain0", "wlow+chain1", "wlow+chain2"))}
        res["systems"][tag] = o
        print(tag, json.dumps({c: d["median"] for c, d in o.items()}, ensure_ascii=False), flush=True)
        del s
        torch.cuda.empty_cache()
    (ROOT / opt.get("out", "results/snv_d5/shift_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
