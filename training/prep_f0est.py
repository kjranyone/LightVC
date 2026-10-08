"""F1(因果な f0・有声推定器・converter.md §3b)の台帳と harvest + stonemask の f0(200fps・フレーム k の中心 = 48kHz の 240k・f0_floor 60 = 女声の教師 data/rvoc_f0 と同じ)。

学習: VCTK の評価に使わない男声(差し替えラダーの男 24 人以外)の全発話(共通文 1〜24 を除く)+ その声を元にした JA TTS 男声。
評価: VCTK の評価男声 24 人の発話 41〜59(学習に入れない・そのクローンの JA TTS 男声も学習から除く)。
女声は data/rvoc_f0(出力部と共通・高域の選別あり)+ その選別で落ちた女声(f0 には帯域は無関係 = フルデータ)をここで作る(female_extra)。

    OMP_NUM_THREADS=1 uv run python prep_f0est.py --ladder <scratchpad>/r4_spk --procs 8
出力: data/f0est/<src>/<spk>/<stem>.npy・data/f0est/manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "f0est"
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
TTS_M = ROOT / "data/male_tts_corpus"


def work(job: tuple) -> dict:
    import pyworld
    import soundfile as sf
    from scipy.signal import resample_poly
    src, spk, wav, split = job
    rec = {"src": src, "spk": spk, "wav": str(wav), "split": split}
    try:
        x, sr = sf.read(str(wav), dtype="float64", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        rec.update(sr=int(sr), dur=round(len(x) / sr, 3))
        dst = OUT / src / spk / (Path(wav).stem + ".npy")
        if not dst.exists():
            g = math.gcd(sr, 16000)
            x16 = resample_poly(x, 16000 // g, sr // g)
            f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
            f0 = pyworld.stonemask(x16, f0, t, 16000)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, f0.astype(np.float32))
        return {**rec, "f0": str(dst.relative_to(ROOT)), "ok": True}
    except Exception as e:
        return {**rec, "ok": False, "why": f"err {type(e).__name__}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    ev_m = set(J["males"])
    info = [l.split() for l in (VC / "speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    males = sorted(f"p{r[0]}" for r in info if len(r) > 2 and r[2] == "M")
    tr_m = [s for s in males if s not in ev_m]
    jobs = []
    for s in tr_m:
        for w in sorted((VC / "wav48" / s).glob("*.wav")):
            if int(w.stem.split("_")[1]) > 24:
                jobs.append(("vctk_m", s, w, "train"))
        d = TTS_M / f"male_{s}"
        if d.exists():
            for w in sorted(d.glob("*.wav")):
                jobs.append(("tts_m", f"male_{s}", w, "train"))
    for s in sorted(ev_m):
        for u in range(41, 60):
            w = VC / "wav48" / s / f"{s}_{u:03d}.wav"
            if w.exists():
                jobs.append(("vctk_m", s, w, "eval"))
    rv = json.loads((ROOT / "data/rvoc_f0/manifest.json").read_text())
    for r in rv["rows"]:
        if not r["keep"]:
            src = "female_extra_" + ("real" if r["src"] == "real_female" else "tts")
            jobs.append((src, r["spk"], Path(r["wav"]), "train"))
    print("train male speakers", len(tr_m), "jobs", len(jobs), flush=True)
    t0 = time.time()
    with Pool(a.procs) as pool:
        rows = list(pool.imap_unordered(work, jobs, chunksize=8))
    rows.sort(key=lambda r: r["wav"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps({"train_male_speakers": tr_m, "eval_male_speakers": sorted(ev_m), "rows": rows}, ensure_ascii=False))
    for sp in ("train", "eval"):
        for src in ("vctk_m", "tts_m", "female_extra_real", "female_extra_tts"):
            sel = [r for r in rows if r["split"] == sp and r["src"] == src and r["ok"]]
            print(sp, src, "files", len(sel), "hours", round(sum(r["dur"] for r in sel) / 3600, 2), flush=True)
    print("done", round((time.time() - t0) / 60, 1), "min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
