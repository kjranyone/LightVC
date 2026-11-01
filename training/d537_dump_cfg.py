"""D537: d467_steady と同じ組・同じ前置きの入力(<set>/d467/<k>.wav)を、Rust の追加のフラグつきで書き出す(<H>/e2e/eval_<set>_pre_<cfg>)。d536_render_dump.py で描画する。
    uv run python d537_dump_cfg.py --sets devreal dev2 dev3 devvctkm --cfg V70 --flags=--voicing-hyst+0.7,0.7
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--flags", required=True, help="+ で区切った追加の引数")
    a = ap.parse_args()
    fl = [x for x in a.flags.split("+") if x]
    for s in a.sets:
        S = P.H / s
        ed = P.H / "e2e" / f"eval_{s}_pre_{a.cfg}"
        ed.mkdir(parents=True, exist_ok=True)
        pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"] if r.get("kind", "FF") != "AA" and (S / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav").exists()]
        if s == "devvctkm":
            pairs = [r for r in pairs if r["kind"].startswith("M")]
        jobs = []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            w = S / "d467" / f"{k}.wav"
            assert w.exists(), w
            jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + fl + ["-i", w], ed / f"{k}.cond", ed / f"{k}.front"))
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        print("dumped", s, a.cfg, sum((ed / f"p{r['pi']:04d}_t{r['ti']:03d}.cond").exists() for r in pairs), "/", len(pairs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
