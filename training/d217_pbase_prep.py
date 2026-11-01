"""D217 の前計算(results/diag_rr18p/prereg.yaml): クローンの交差ペアの学習の土台を製品の経路にするための Rust の書き出し。
源(男声 TTS・data/realx/wav の 1,154 本)ごとに LIGHTVC_DUMP_FRONT(C1 の logit・F3 の包絡)を 1 回・操作者(男声 TTS 18 声)の登録を項目の番号の偶奇で 2 組
(項目は自分の源を含まない組を使う)・目標(1,154 人)の登録 = 記憶と同じ参照のクリップ(ref_stems)をつないだ音声。土台の式は pbase.py(Rust と一致 2.9e-6)。
出力 = data/realx/pbase/{front/<stem>.f32, op/<spk>_<h>/, en/<target>/, ref/}・manifest.json。
    uv run python d217_pbase_prep.py --workers 5
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
OUT = ROOT / "data/realx/pbase"
FILES = ("items_clone.json", "items_clone_full.json", "items_clone_multi.json", "items_clone_scale.json", "items_clone_full2.json")
ART = ["--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3", "--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5",
       "--resid", "0.3", "--color", "--soft", "1"]


def run(j: tuple) -> str:
    cmd, env, done = j
    if Path(done).exists():
        return "skip"
    e = {"PATH": "/usr/bin:/bin", **env}
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=e, capture_output=True)
    if r.returncode != 0:
        return f"FAIL {done} {r.stderr[-300:]!r}"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    for d in ("front", "op", "en", "ref", "tmp"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    items = sum((json.loads((ROOT / "data/realx" / f).read_text()) for f in FILES if (ROOT / "data/realx" / f).exists()), [])
    src = {}
    for it in items:
        s = f"p{it['item']:04d}_t{it['ti']:03d}_SRC"
        assert src.setdefault(s, it["spk"]) == it["spk"]
    refs = {}
    for it in items:
        assert refs.setdefault(it["target"], tuple(it["ref_stems"])) == tuple(it["ref_stems"]), it["target"]
    spk = sorted(set(src.values()))
    print("src", len(src), "spk", len(spk), "targets", len(refs), flush=True)
    jobs = []
    for s_ in spk:
        for h in (0, 1):
            w = OUT / "ref" / f"op_{s_}_{h}.wav"
            if not w.exists():
                xs, tot = [], 0
                for st in sorted(k for k, v in src.items() if v == s_ and int(k[1:5]) % 2 == h):
                    x = load48(ROOT / "data/realx/wav" / f"{st}.wav"); xs.append(x); tot += len(x)
                    if tot >= 40 * 48000:
                        break
                sf.write(w, np.concatenate(xs), 48000)
            jobs.append(([BIN, "artic-enroll", "-r", w, "-o", OUT / "op" / f"{s_}_{h}"], {}, str(OUT / "op" / f"{s_}_{h}" / "enroll.json")))
    with Pool(a.workers) as pool:
        print("op", [r for r in pool.imap_unordered(run, jobs) if r.startswith("FAIL")], flush=True)
    jobs = []
    for st, s_ in sorted(src.items()):
        o = OUT / "front" / f"{st}.f32"
        jobs.append(([BIN, "artic", "--enroll", OUT / "op" / f"{s_}_0", "--operator-enroll", OUT / "op" / f"{s_}_1", *ART, "-i", ROOT / "data/realx/wav" / f"{st}.wav", "-o", OUT / "tmp" / f"{st}.wav"],
                     {"LIGHTVC_COND_ONLY": "1", "LIGHTVC_DUMP_FRONT": str(o)}, str(o)))
    with Pool(a.workers) as pool:
        bad = [r for r in pool.imap_unordered(run, jobs, chunksize=4) if r.startswith("FAIL")]
    print("front", len(jobs), "fail", bad[:3], flush=True)
    for f in (OUT / "tmp").glob("*.wav"):
        f.unlink()
    jobs = []
    for A, stems in sorted(refs.items()):
        w = OUT / "ref" / f"{A}.wav"
        if not w.exists():
            sf.write(w, np.concatenate([load48(ROOT / "female-dataset" / A / f"{x}.wav") for x in stems]), 48000)
        jobs.append(([BIN, "artic-enroll", "-r", w, "-o", OUT / "en" / A], {}, str(OUT / "en" / A / "enroll.json")))
    with Pool(a.workers) as pool:
        bad = [r for r in pool.imap_unordered(run, jobs, chunksize=4) if r.startswith("FAIL")]
    print("en", len(jobs), "fail", bad[:3], flush=True)
    man = {"src_spk": src, "targets": sorted(refs), "op_half": "項目 i は op/<spk>_<1 − i%2>(自分の源を含まない組)", "alpha": 1.3, "gamma": 0.3, "beta": 1.0}
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print("PBASEDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
