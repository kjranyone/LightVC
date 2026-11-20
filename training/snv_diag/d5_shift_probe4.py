"""段 D5 の補足 4(results/snv_d5/prereg.yaml supplement_4・学習なし): 開発の組の本物の声を Irodori の符号器と透かしに通すと、変換の内容が戻るか。
cv の devA の先頭 n 人の s0 と登録 e0〜e7(snv_diag/i15_codec_pass.py groups cv_dev → data/snv_i15_codec/codec_wm/cv_dev)。操作者の側(源と登録)だけを通した音の特徴に替える。
条件: clean / codecwm・目標 = A0F の 2 声(k 0)と self(構造 = 自分の登録)。CER = large-v3 の仮名で、元の音の書き起こし 対 出力。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_shift_probe4.py snv_i10_c_pr snv_i12_combo snv_p8_wst [n=24] [out=results/snv_d5/shift_probe4.json]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
SR = P.SR


def work(job: tuple) -> tuple:
    import librosa
    key, path = job
    torch.set_num_threads(1)
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    x = x[: P.n_frames(len(x)) * P.H]
    f = P.analyze(x)
    f0, env, per = f["f0"], f["env"], f["per"]
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return key, {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": x}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a] or ["snv_i10_c_pr", "snv_i12_combo", "snv_p8_wst"]
    n = int(opt.get("n", 24))
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    spk = [s["speaker"] for s in man["speakers"] if s["split"] == "devA"][:n]
    names = ["s0"] + [f"e{i}" for i in range(8)]
    path = {"clean": lambda k: str(T.FEAT / f"cv/{k}.wav"), "codecwm": lambda k: str(ROOT / f"data/snv_i15_codec/codec_wm/cv_dev/{k}.wav")}
    jobs = [((c, f"{s}_{nm}"), path[c](f"{s}_{nm}")) for c in path for s in spk for nm in names]
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
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    ref = {s: tr(w16(U[("clean", f"{s}_s0")]["wav"])) for s in spk}
    res: dict = {"prereg": "results/snv_d5/prereg.yaml supplement_4", "n_cv": n, "systems": {}, "rows": []}
    for tag in tags:
        sy = EV.System(tag, gen)
        o = {}
        for c in path:
            for s in spk:
                enr = [U[(c, f"{s}_e{i}")] for i in range(8)]
                for v, S in [(v, sy.structure(f"t|{v}", trefs[v])) for v in tgts] + [(None, sy.structure(f"self|{s}|{c}", enr))]:
                    e = cer(ref[s], tr(w16(sy.convert(U[(c, f"{s}_s0")], enr, S, 0))))
                    res["rows"].append({"sys": tag, "cond": c if v is not None else f"{c}_self", "spk": s, "tgt": v, "cer": round(e, 4)})
        for c in ("clean", "codecwm", "clean_self", "codecwm_self"):
            cs = [r["cer"] for r in res["rows"] if r["sys"] == tag and r["cond"] == c]
            o[c] = {"median": round(float(np.median(cs)), 4), "mean": round(float(np.mean(cs)), 4), "n": len(cs)}
        res["systems"][tag] = o
        print(tag, json.dumps({c: d["median"] for c, d in o.items()}, ensure_ascii=False), flush=True)
        del sy
        torch.cuda.empty_cache()
    (ROOT / opt.get("out", "results/snv_d5/shift_probe4.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
