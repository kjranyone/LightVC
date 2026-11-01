"""段 1 の基準(current/ml_method.md §3・current/e2e_ppg.md)のデータ: female-dataset の評価外の実女声 n 人 × 3 クリップを 48 kHz に再標本化して Rust の新しい前段(--causal-ct)に通し、条件(31 行・学習の正解)と C1 の事後(200・学習の入力)を書き出す(描画は省く)。
    uv run python d274_data.py --s <dev の親> --out <dev の親>/e2e --n_fem 600
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
SELF = ["--mix", "1", "--causal-ct", "--no-f0-shift", "--f4", "results/f4_1/export", "--rvoc", "results/rvoc3hi/export"]


def job(j: tuple) -> None:
    cmd, cond, front = j
    if Path(cond).exists():
        return
    tmp = str(cond) + ".wav"
    env = {"PATH": "/usr/bin:/bin", "LIGHTVC_COND_ONLY": "1", "LIGHTVC_DUMP_COND": str(cond), "LIGHTVC_DUMP_FRONT": str(front)}
    r = subprocess.run([str(c) for c in cmd] + ["-o", tmp], cwd=ROOT, env=env, capture_output=True)
    Path(tmp).unlink(missing_ok=True)
    if r.returncode != 0:
        Path(cond).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_fem", type=int, default=600)
    a = ap.parse_args()
    from d6_unit_cov import load48
    S, O = Path(a.s), Path(a.out)
    for d in ("fem", "male", "eval", "wav"):
        (O / d).mkdir(parents=True, exist_ok=True)
    excl = set((S / "excl.txt").read_text().split())
    for f in list(S.glob("*/devset.json")) + [S / "eval_targets.json"]:
        j = json.loads(f.read_text())
        excl |= set(j["targets"])
        excl |= {p["spk"] for p in j.get("pairs", []) if isinstance(p, dict) and "spk" in p}
    excl |= {r["id"] for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"] if r["grp"] == "REAL_POOL"}
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(273).shuffle(pool)
    jobs, meta = [], {"fem": [], "male": [], "eval": []}
    for spk in pool:
        if len({m["spk"] for m in meta["fem"]}) >= a.n_fem:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 5:
            continue
        picks = [w for w in (ws[len(ws) // 2], ws[len(ws) // 3], ws[(2 * len(ws)) // 3]) if sf.info(str(w)).duration >= 3.0]
        if len(picks) < 3:
            continue
        for w in picks:
            k = f"{spk}_{w.stem}"
            wp = O / "wav" / f"{k}.wav"
            if not wp.exists():
                sf.write(wp, load48(w)[: 20 * 48000].astype(np.float32), 48000)
            jobs.append(([BIN, "artic"] + SELF + ["-i", wp], O / "fem" / f"{k}.cond", O / "fem" / f"{k}.front"))
            meta["fem"].append({"key": k, "spk": spk})
    (O / "meta.json").write_text(json.dumps(meta, indent=1))
    print("jobs", len(jobs), {k: len(v) for k, v in meta.items()}, flush=True)
    with Pool(10) as pl:
        for i, _ in enumerate(pl.imap_unordered(job, jobs, chunksize=4)):
            if (i + 1) % 200 == 0:
                print("done", i + 1, flush=True)
    ok = {k: sum((O / k / f"{m['key']}.cond").exists() for m in v) for k, v in meta.items()}
    print("written", ok, flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
