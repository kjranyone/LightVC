"""腕 64 の合成の男声のコーパス(data/male_tts_v4l_train)の台帳と教師の f0。学習の女声と同じ作り方: 高域の検査(prep_rvoc_f0.band_db・同じ閾)→ harvest + stonemask(5 ms)= A
→ f0hi.teacher_f0(A + DIO48 の検査で 1 kHz 超を置換)。出力 = data/rvoc_f0hi_male/<spk>/<stem>.npy・data/male_tts_v4l_train/f0_manifest.json(rvoc_f0hi の台帳と同じ形)。
    OMP_NUM_THREADS=1 uv run python prep_male_train_f0.py --procs 11
本物の男声(E-C1r・VCTK の評価の話者を除く): --wavs <wav の一覧の json> --out ../data/rvoc_f0hi_male_vctk --manifest ../data/male_vctk_train/f0_manifest.json --tag vctk_male
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import prep_rvoc_f0 as PR

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/male_tts_v4l_train"
OUT = ROOT / "data/rvoc_f0hi_male"
TAG = "tts_male_v4l"


def work(job: tuple) -> dict:
    wav, out, tag = job
    import pyworld
    import soundfile as sf
    from scipy.signal import resample_poly
    import f0hi as FH
    spk = Path(wav).parent.name
    rec = {"src": tag, "spk": spk, "wav": wav}
    try:
        info = sf.info(wav)
        rec.update(sr=int(info.samplerate), dur=round(float(info.duration), 3))
        if info.samplerate < PR.SR_MIN:
            return {**rec, "keep": False, "why": "sr"}
        x, sr = sf.read(wav, dtype="float64", always_2d=True)
        x = x.mean(1)
        b1, b2 = PR.band_db(x, sr)
        rec.update(b1=round(b1, 1), b2=round(b2, 1))
        if b1 < PR.B1_MIN or b2 < PR.B2_MIN:
            return {**rec, "keep": False, "why": "hf"}
        dst = Path(out) / spk / (Path(wav).stem + ".npy")
        g = math.gcd(sr, 16000)
        x16 = resample_poly(x, 16000 // g, sr // g)
        A, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
        A = pyworld.stonemask(x16, A, t, 16000).astype(np.float32)
        x48 = x if sr == 48000 else resample_poly(x, 48000 // math.gcd(sr, 48000), sr // math.gcd(sr, 48000))
        f, hi = FH.teacher_f0(x48.astype(np.float32), len(A), A)
        assert np.array_equal(f[~hi], A[~hi])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, f.astype(np.float32))
        v = f[f > 0]
        return {**rec, "keep": True, "f0": str(dst.relative_to(ROOT)), "hi_frames": int(hi.sum()), "f0_med": round(float(np.median(v)), 1) if len(v) else 0.0,
                "vfrac": round(float((f > 0).mean()), 3)}
    except Exception as e:
        return {**rec, "keep": False, "why": f"err {type(e).__name__} {e}"[:200]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--wavs", default="", help="wav の一覧(json のリスト)・既定 = data/male_tts_v4l_train の mt*/u*.wav")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--manifest", default=str(SRC / "f0_manifest.json"))
    ap.add_argument("--tag", default=TAG)
    a = ap.parse_args()
    wavs = json.loads(Path(a.wavs).read_text()) if a.wavs else sorted(str(p) for p in SRC.glob("mt*/u*.wav"))
    with Pool(a.procs) as pl:
        rows = pl.map(work, [(w, str(Path(a.out).resolve()), a.tag) for w in wavs], chunksize=8)
    kept = [r for r in rows if r["keep"]]
    why: dict = {}
    for r in rows:
        if not r["keep"]:
            why[r["why"].split()[0]] = why.get(r["why"].split()[0], 0) + 1
    rep = {"files": len(rows), "kept": len(kept), "spk": len({r["spk"] for r in kept}), "hours": round(sum(r["dur"] for r in kept) / 3600, 2), "dropped": why,
           "f0_med_of_files": round(float(np.median([r["f0_med"] for r in kept if r["f0_med"] > 0])), 1) if kept else 0.0}
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(a.manifest).write_text(json.dumps({"rule": "prep_male_train_f0.py", "sr_min": PR.SR_MIN, "b1_min": PR.B1_MIN, "b2_min": PR.B2_MIN,
                                                       "f0_teacher": "f0hi.teacher_f0 (A = harvest ceil 1000 + B = DIO48 gated)", **rep, "rows": rows}, ensure_ascii=False))
    print(rep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
