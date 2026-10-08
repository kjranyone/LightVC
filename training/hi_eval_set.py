"""1kHz 超の f0 を含む評価区間(held 話者 = 学習に使っていない 21 話者の全発話から走査・current/f0_range.md §3-4)。
各話者の発話を最大 --per_spk 本走査し、f0hi.teacher_f0 で置換フレームが --min_hi 以上ある発話から、置換フレームの中心に 2 秒の窓を取って保存する。
出力: data/hi_eval/<spk>_<stem>_<k>.npz(x: float32 48kHz・f0: 新しい教師・f0_old: A のみ)・data/hi_eval/index.json
    OMP_NUM_THREADS=1 uv run python hi_eval_set.py --per_spk 150 --procs 11
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hi_eval"
WIN = 2.0


def scan(job):
    import f0hi as H
    from train_ddsp_vc import load48
    spk, stem, path, min_hi = job
    try:
        x = load48(path).astype(np.float32)[:12 * 48000]
        n = len(x) // H.HOP
        A = H.harvest(x, H.CEIL_LO)
        f, rep = H.teacher_f0(x, n, A)
        if rep.sum() < min_hi:
            return []
        idx = np.where(rep)[0]
        c = int(np.median(idx))
        half = int(WIN / 2 / 0.005)
        a = max(0, min(c - half, n - 2 * half))
        sl = slice(a, a + 2 * half)
        A = np.pad(A[:n], (0, max(0, n - len(A))))
        name = f"{spk}_{stem}"
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"{name}.npz", x=x[a * H.HOP:(a + 2 * half) * H.HOP], f0=f[sl], f0_old=A[sl], src=path, start_frame=a)
        return [{"name": name, "spk": spk, "path": path, "start_frame": int(a), "hi_frames": int(rep[sl].sum()), "f0_hi_median": float(np.median(f[sl][rep[sl]]))}]
    except Exception as e:
        return [{"name": f"{spk}_{stem}", "err": f"{type(e).__name__}: {e}"}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_spk", type=int, default=150)
    ap.add_argument("--min_hi", type=int, default=20)
    ap.add_argument("--procs", type=int, default=11)
    a = ap.parse_args()
    from s0_artic import build_index
    pairs, lats, held = build_index(0)
    jobs = []
    for s in sorted(held):
        wavs = sorted((ROOT / "female-dataset" / s).glob("*.wav"))[:a.per_spk] if (ROOT / "female-dataset" / s).is_dir() else []
        if not wavs:
            files = sorted(p for p in pairs if p.parent.name == s)[:a.per_spk]
            wavs = [Path(torch.load(f, map_location="cpu", weights_only=False)["path"]) for f in files]
        for w in wavs:
            jobs.append((s, w.stem, str(w), a.min_hi))
    print("held speakers", len(held), "files to scan", len(jobs), flush=True)
    with Pool(a.procs) as pool:
        res = [r for rr in pool.imap(scan, jobs, chunksize=4) for r in rr]
    ok = [r for r in res if "err" not in r]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(json.dumps({"min_hi": a.min_hi, "items": ok, "errors": [r for r in res if "err" in r]}, ensure_ascii=False, indent=1))
    print("segments", len(ok), "speakers with hi", len({r["spk"] for r in ok}), "errors", len(res) - len(ok), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
