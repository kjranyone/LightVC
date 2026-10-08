"""rvoc の学習データ台帳と精密な f0(current/renderer.md §3)。女声フルコーパスの train 全量を走査し、
高域を持つファイルだけ(標本化 ≥ 44.1kHz・11.5〜15.5kHz ≥ −40dB・16.5〜20kHz ≥ −65dB、どちらも 2〜6kHz 比)に
harvest + stonemask の f0 を 200fps(フレーム k = 48kHz のサンプル 240k を中心)で前計算する。

    OMP_NUM_THREADS=1 uv run python prep_rvoc_f0.py --procs 12
出力: data/rvoc_f0/<src>/<spk>/<stem>.npy(float32)・data/rvoc_f0/manifest.json
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
OUT = ROOT / "data" / "rvoc_f0"
SR_MIN = 44100
B1_MIN = -40.0
B2_MIN = -65.0


def band_db(x: np.ndarray, sr: int) -> tuple[float, float]:
    from scipy.signal import welch
    f, p = welch(x, sr, nperseg=2048)
    ref = p[(f >= 2000) & (f < 6000)].mean() + 1e-30
    b1 = p[(f >= 11500) & (f < 15500)].mean() / ref
    b2 = p[(f >= 16500) & (f < 20000)].mean() / ref
    return float(10 * np.log10(b1 + 1e-30)), float(10 * np.log10(b2 + 1e-30))


def work(job: tuple) -> dict:
    import pyworld
    import soundfile as sf
    from scipy.signal import resample_poly
    src, spk, wav = job
    rec = {"src": src, "spk": spk, "wav": str(wav)}
    try:
        info = sf.info(str(wav))
        rec.update(sr=int(info.samplerate), dur=round(float(info.duration), 3))
        if info.samplerate < SR_MIN:
            return {**rec, "keep": False, "why": "sr"}
        x, sr = sf.read(str(wav), dtype="float64", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        b1, b2 = band_db(x, sr)
        rec.update(b1=round(b1, 1), b2=round(b2, 1))
        if b1 < B1_MIN or b2 < B2_MIN:
            return {**rec, "keep": False, "why": "hf"}
        dst = OUT / src / spk / (Path(wav).stem + ".npy")
        if not dst.exists():
            g = math.gcd(sr, 16000)
            x16 = resample_poly(x, 16000 // g, sr // g)
            f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
            f0 = pyworld.stonemask(x16, f0, t, 16000)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, f0.astype(np.float32))
        return {**rec, "keep": True, "f0": str(dst.relative_to(ROOT))}
    except Exception as e:
        return {**rec, "keep": False, "why": f"err {type(e).__name__}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from train_ddsp_vc import index
    spk, tr, _ = index()
    jobs = [(k.split("/")[0], k.split("/")[1], w) for k in tr if not k.endswith("/unknown") for _, w, _ in spk[k]]
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    print("files", len(jobs), flush=True)
    rows = []
    t0 = time.time()
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=8)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(jobs)}  {el / 60:.1f} min  eta {el / (i + 1) * (len(jobs) - i - 1) / 60:.1f} min  keep {np.mean([q['keep'] for q in rows]):.3f}", flush=True)
    rows.sort(key=lambda r: r["wav"])
    OUT.mkdir(parents=True, exist_ok=True)
    name = "manifest.json" if not a.limit else "manifest_limit.json"
    (OUT / name).write_text(json.dumps({"sr_min": SR_MIN, "b1_min": B1_MIN, "b2_min": B2_MIN, "rows": rows}, ensure_ascii=False))
    for g in ("real_female", "tts"):
        sel = [r for r in rows if r["src"].startswith(g)]
        kept = [r for r in sel if r["keep"]]
        why = {}
        for r in sel:
            if not r["keep"]:
                why[r["why"]] = why.get(r["why"], 0) + 1
        print(g, "files", len(sel), "kept", len(kept), f"({len(kept) / max(1, len(sel)):.3f})", "hours", round(sum(r["dur"] for r in kept) / 3600, 1), "dropped", why, flush=True)
    print("done", round((time.time() - t0) / 60, 1), "min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
