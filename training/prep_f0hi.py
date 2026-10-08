"""高い f0(1kHz 超)に対応した教師の f0 を renderer の学習データに反映する(current/f0_range.md §3-2)。
data/rvoc_f0/manifest.json の keep な行ごとに、既存の f0(A・harvest 上限 1000Hz)をそのまま使い、f0hi の B(DIO 48kHz + 検査)を採ったフレームだけ置換した npy を
data/rvoc_f0hi/<src>/<spk>/<stem>.npy に書く(B が無いファイルは書かず、台帳の f0 は元の npy を指したまま = **data/rvoc_f0 を消さない**)。置換の外が A とビット一致することをファイルごとに assert する。エラーがあれば errors.json に出し終了コード 1。台帳 data/rvoc_f0hi/manifest.json は manifest.json と同じ形式 + 各行に hi_frames。
    OMP_NUM_THREADS=1 uv run python prep_f0hi.py --procs 11 [--limit 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import gcd
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "rvoc_f0" / "manifest.json"
OUT = ROOT / "data" / "rvoc_f0hi"
KEEP = "keep"


def work(row: dict) -> dict:
    import soundfile as sf
    from scipy.signal import resample_poly
    import f0hi as H
    if not row.get(KEEP):
        return row
    try:
        A = np.load(ROOT / row["f0"]).astype(np.float32)
        x, sr = sf.read(row["wav"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 48000:
            g = gcd(sr, 48000)
            x = resample_poly(x, 48000 // g, sr // g).astype(np.float32)
        n = len(A)
        f, hi = H.teacher_f0(x, n, A)
        assert np.array_equal(f[~hi], A[~hi]), "置換の外が A とビット一致しない"
        assert len(f) == len(A)
        out = dict(row)
        out["hi_frames"] = int(hi.sum())
        if hi.any():
            dst = OUT / row["src"] / row["spk"] / (Path(row["wav"]).stem + ".npy")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp.npy")
            np.save(tmp, f.astype(np.float32))
            tmp.replace(dst)
            out["f0"] = str(dst.relative_to(ROOT))
        return out
    except Exception as e:
        return {**row, "hi_err": f"{type(e).__name__}: {e}"}


def main() -> int:
    global SRC, OUT, KEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--src", default=str(SRC), help="元の台帳(rows の各行に wav・f0)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep_key", default="keep", help="使う行の真偽のキー(f0est の台帳は ok)")
    a = ap.parse_args()
    SRC, OUT, KEEP = Path(a.src).resolve(), Path(a.out).resolve(), a.keep_key
    m = json.loads(SRC.read_text())
    rows = m["rows"]
    keep = [r for r in rows if r.get(KEEP)]
    if a.limit:
        keep = keep[::max(1, len(keep) // a.limit)][:a.limit]
    print("files", len(keep), flush=True)
    t0 = time.time()
    done = []
    with Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap(work, keep, chunksize=8)):
            done.append(r)
            if (i + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(keep)}  {el / 60:.1f} min  eta {el / (i + 1) * (len(keep) - i - 1) / 60:.1f} min  files_with_hi {sum(1 for q in done if q.get('hi_frames'))}", flush=True)
    byw = {r["wav"]: r for r in done}
    out_rows = [byw.get(r["wav"], r) if r.get(KEEP) else r for r in rows] if not a.limit else done
    OUT.mkdir(parents=True, exist_ok=True)
    name = "manifest.json" if not a.limit else "manifest_limit.json"
    (OUT / name).write_text(json.dumps({**{k: v for k, v in m.items() if k != "rows"}, "f0_teacher": "f0hi.teacher_f0 (A = harvest ceil 1000 + B = DIO48 gated)", "rows": out_rows}, ensure_ascii=False))
    for g in sorted({r["src"] for r in done}):
        sel = [r for r in done if r["src"] == g]
        print(g, "files", len(sel), "with_hi", sum(1 for r in sel if r.get("hi_frames")), "hi_frames", sum(r.get("hi_frames", 0) for r in sel),
              "errors", sum(1 for r in sel if r.get("hi_err")), flush=True)
    errs = [r for r in done if r.get("hi_err")]
    (OUT / "errors.json").write_text(json.dumps([{"wav": r["wav"], "err": r["hi_err"]} for r in errs], ensure_ascii=False, indent=1))
    n_hi = sum(1 for r in done if r.get("hi_frames"))
    print("files_with_hi", n_hi, "hi_frames_total", sum(r.get("hi_frames", 0) for r in done), "errors", len(errs), flush=True)
    print("done", round((time.time() - t0) / 60, 1), "min", flush=True)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
