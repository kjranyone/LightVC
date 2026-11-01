"""D506 の組 devff5(女声 → 女声の関門 v3 の最終の組の候補): どの学習の台帳(rvoc_f0hi・rvoc_f0・f0est・f0est_hi)にも、どの評価の組(scratch の devset.json の操作者と目標)にも、
v1 の本物の群(d237)にも入っていない female-dataset の実女声のうち、44.1 kHz 以上のファイルの合計が 28 s 以上の話者。目標 = <dev> の 30 人(登録・参照はそのまま)。
1 人の操作者から 8 s の入力のクリップを最大 3 本(真ん中のファイルから順に・互いに重ならない)・操作者の参照 = 残りのファイルの 20〜22 s(artic-enroll)。1 人 6 組(クリップ数で目標を割り振る・Random(5061) の巡回)。
    uv run python d506_ffset5.py --s <dev の親> --ff <dev の親>/devff5
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-500:]}")


def man(p: Path) -> set:
    j = json.loads(p.read_text())
    rows = j["rows"] if isinstance(j, dict) and "rows" in j else j
    return {r.get("spk") for r in rows if isinstance(r, dict)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--ff", required=True)
    ap.add_argument("--dev", default="dev")
    ap.add_argument("--per_op", type=int, default=6)
    a = ap.parse_args()
    S, F = Path(a.s), Path(a.ff)
    dev = S / a.dev
    base = ROOT / "female-dataset"
    trained = set()
    for m in ("rvoc_f0hi", "rvoc_f0", "f0est", "f0est_hi"):
        trained |= man(ROOT / f"data/{m}/manifest.json")
    used = set()
    for f in S.glob("*/devset.json"):
        j = json.loads(f.read_text())
        used |= {r["spk"] for r in j.get("pairs", []) if "spk" in r}
        used |= {t for t in j.get("targets", []) if isinstance(t, str)}
    pool_real = {r["id"] for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]}
    cand = sorted({p.name for p in base.iterdir() if p.is_dir()} - trained - used - pool_real)
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (F / d).mkdir(parents=True, exist_ok=True)
    for f in sorted((dev / "d37/_work").glob("ref_*.wav")):
        shutil.copy(f, F / "d37/_work" / f.name)
    for e in sorted((dev / "d85_work").glob("en_*")):
        if not (F / "d85_work" / e.name).exists():
            shutil.copytree(e, F / "d85_work" / e.name)
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    tcycle = list(range(len(targets)))
    random.Random(5061).shuffle(tcycle)
    tq = iter(tcycle * 20)
    log, pi, ops = [], 0, []
    for spk in cand:
        ws = [w for w in sorted((base / spk).glob("*.wav")) if sf.info(str(w)).samplerate >= 44100]
        if not ws:
            continue
        durs = [sf.info(str(w)).duration for w in ws]
        if sum(durs) < 28.0:
            continue
        mid = len(ws) // 2
        order = list(range(mid, len(ws))) + list(range(0, mid))
        nclip = int(min(3, max(1, (sum(durs) - 22.0) // 8.0)))
        clips, usedk = [], set()
        for _ in range(nclip):
            src, tl = [], 0.0
            for k in order:
                if k in usedk or tl >= 8.0:
                    continue
                src.append(k); tl += durs[k]
            if tl < 8.0:
                break
            usedk |= set(src)
            clips.append(src)
        rest = [k for k in range(len(ws)) if k not in usedk]
        ref, rl = [], 0.0
        for k in rest:
            if rl >= 22.0:
                break
            ref.append(k); rl += durs[k]
        if not clips or rl < 20.0:
            if clips and rl < 20.0 and len(clips) > 1:
                clips = clips[:-1]
                rest = [k for k in range(len(ws)) if k not in set().union(*map(set, clips))]
                ref, rl = [], 0.0
                for k in rest:
                    if rl >= 22.0:
                        break
                    ref.append(k); rl += durs[k]
            if not clips or rl < 20.0:
                continue
        xr = np.concatenate([load48(ws[k]) for k in ref]).astype(np.float32)
        ops.append(spk)
        en0 = None
        per_clip = [a.per_op // len(clips) + (1 if i < a.per_op % len(clips) else 0) for i in range(len(clips))]
        for ci, src in enumerate(clips):
            xs = np.concatenate([load48(ws[k]) for k in src])[: 8 * 48000].astype(np.float32)
            for _ in range(per_clip[ci]):
                ti = next(tq)
                sf.write(F / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", xs, 48000)
                opw = F / "d56/_work" / f"op_{pi:04d}.wav"
                sf.write(opw, xr, 48000)
                opd = F / "d85_work" / f"op_{pi:04d}"
                if not (opd / "enroll.json").exists():
                    if en0 is None:
                        run([BIN, "artic-enroll", "-r", opw, "-o", opd])
                        en0 = opd
                    else:
                        shutil.copytree(en0, opd)
                log.append({"pi": pi, "ti": ti, "kind": "FF", "spk": spk, "clip": ci, "src": [ws[k].name for k in src], "ref": [ws[k].name for k in ref], "target": targets[ti]})
                pi += 1
    (F / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "female_to_female"}, ensure_ascii=False, indent=1))
    print("candidates", len(cand), "FF pairs", len(log), "operators", len(ops), "targets", len({r["ti"] for r in log}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
