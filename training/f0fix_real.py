"""female_real_feat の f0 再計算オーバーレイ。

監査(diag_cfm_audit + corpus scan)で female_real_feat の保存 f0 が約85%の
ファイルで元音声と無相関(canonical: content cos=1.0 / energy corr=1.0 は健全)、
female_tts_feat・male_feat は同規約で corr≈1 と判明したため、検証済み規約
(librosa 44.1k load -> harvest+stonemask, HOP512)で f0 のみを再計算して
data/female_real_f0fix/<spk>/<stem>.pt へ書く。元キャッシュは変更しない。

    uv run python f0fix_real.py --workers 10
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pyworld
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/female_real_feat"
OUT = ROOT / "data/female_real_f0fix"
HOP = 512


def fix_one(args: tuple[str, str, str]) -> tuple[str, str]:
    src_path, out_path, wav_path = args
    try:
        d = torch.load(src_path, map_location="cpu", weights_only=False)
        w44, _ = librosa.load(wav_path, sr=44100, mono=True)
        n = len(w44) // HOP
        w64 = w44.astype(np.float64)
        f0, t = pyworld.harvest(w64, 44100, f0_floor=65, f0_ceil=1000,
                                frame_period=HOP / 44100 * 1000)
        f0 = pyworld.stonemask(w64, f0, t, 44100).astype(np.float32)
        f0 = f0[:n] if len(f0) >= n else np.pad(f0, (0, n - len(f0)))
        torch.save({"f0": torch.from_numpy(f0), "src": src_path}, out_path)
        return (out_path, "")
    except Exception as e:  # noqa: BLE001
        return (out_path, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()

    jobs = []
    for spk in sorted(SRC.iterdir()):
        if not spk.is_dir():
            continue
        od = OUT / spk.name
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(spk.glob("*.pt")):
            outf = od / f.name
            if outf.exists():
                continue
            d = torch.load(f, map_location="cpu", weights_only=False)
            jobs.append((str(f), str(outf), d["path"]))
    print(f"  jobs {len(jobs)}", flush=True)
    t0 = time.time()
    errs = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (_, msg) in enumerate(ex.map(fix_one, jobs, chunksize=16)):
            if msg:
                errs += 1
                print(f"  ERR {msg}", flush=True)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(jobs)} ({time.time()-t0:.0f}s, err {errs})",
                      flush=True)
    summary = {"done": len(jobs) - errs, "errors": errs,
               "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "_fix_summary.json").write_text(json.dumps(summary))
    print(json.dumps(summary), flush=True)
    return 0 if errs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
