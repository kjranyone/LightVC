"""D479 の組 devff4(gate_ff_v2 の最終の組・results/conv_p0/gate_ff_v2.yaml)。目標 = <dev> の 30 人(d37・d85_work/en_TTT をそのまま)。
操作者 = 自己再構成の保留の実女声(train_ppgvoc.e2e_held・学習の行から除外済み)のうち 参照 20 s 以上・44.1 kHz 以上の話者。1 人の入力のクリップ 8 s(真ん中から)を 3 人の目標へ
(目標は Random(4791) の巡回で偏らせない)・操作者の登録 = 残りのクリップ 22 s(artic-enroll)。devff / devff2 / devff3 の操作者・全ての目標とは重ならないことを確かめる。
    uv run python d479_ffset4.py --s <dev の親> --ff <dev の親>/devff4 --dev dev
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--ff", required=True)
    ap.add_argument("--dev", default="dev")
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    import train_ppgvoc as PP
    S, F = Path(a.s), Path(a.ff)
    dev = S / a.dev
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (F / d).mkdir(parents=True, exist_ok=True)
    for f in sorted((dev / "d37/_work").glob("ref_*.wav")):
        shutil.copy(f, F / "d37/_work" / f.name)
    for e in sorted((dev / "d85_work").glob("en_*")):
        if not (F / "d85_work" / e.name).exists():
            shutil.copytree(e, F / "d85_work" / e.name)
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    alltg = set()
    for f in list(S.glob("*/devset.json")):
        alltg |= set(json.loads(f.read_text())["targets"])
    held = [h[0] for h in PP.e2e_held()]
    used_ops = set()
    for d_ in ("devff", "devff2", "devff3"):
        used_ops |= {p["spk"] for p in json.loads((S / d_ / "devset.json").read_text())["pairs"]}
    trained = {r["spk"] for r in json.loads((ROOT / "data/rvoc_f0hi/manifest.json").read_text())["rows"]}
    rows_tr, _, _ = PP.load_rows(set(held), True)
    trained |= {r[4] for r in rows_tr if len(r) > 4}
    base = ROOT / "female-dataset"
    pool = [h for h in held if h not in alltg and h not in used_ops]
    print("held", len(held), "pool", len(pool), "in training rows", len(set(pool) & trained), flush=True)
    assert not (set(pool) & trained), "保留の話者が学習の行にある"
    random.Random(479).shuffle(pool)
    tcycle = list(range(len(targets)))
    random.Random(4791).shuffle(tcycle)
    tq = iter(tcycle * 8)
    log, pi = [], 0
    for spk in pool:
        ws = sorted((base / spk).glob("*.wav"))
        if not ws:
            continue
        durs = [sf.info(str(w)).duration for w in ws]
        if sum(durs) < 32.0 or min(sf.info(str(w)).samplerate for w in ws[:3]) < 44100:
            continue
        mid = len(ws) // 2
        order = list(range(mid, len(ws))) + list(range(0, mid))
        src, tl = [], 0.0
        for k in order:
            if tl >= 8.0:
                break
            src.append(k); tl += durs[k]
        rest = [k for k in range(len(ws)) if k not in src]
        ref, rl = [], 0.0
        for k in rest:
            if rl >= 22.0:
                break
            ref.append(k); rl += durs[k]
        if tl < 8.0 or rl < 20.0:
            continue
        xs = np.concatenate([load48(ws[k]) for k in src])[: 8 * 48000].astype(np.float32)
        xr = np.concatenate([load48(ws[k]) for k in ref]).astype(np.float32)
        en0 = None
        for _ in range(a.per):
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
            log.append({"pi": pi, "ti": ti, "kind": "FF", "spk": spk, "src": [ws[k].name for k in src], "ref": [ws[k].name for k in ref], "target": targets[ti]})
            pi += 1
    (F / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "female_to_female"}, ensure_ascii=False, indent=1))
    print("FF pairs", len(log), "operators", len({r["spk"] for r in log}), "targets", len({r["ti"] for r in log}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
