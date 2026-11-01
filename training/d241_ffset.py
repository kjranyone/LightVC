"""D241 の組(results/conv_p0/d241_prereg.yaml): 女声 → 女声の変換の組 <ff> を dev と同じ構成で作る(男声の話し方・TTS の入力を除いて、系が作る声だけを測る)。
目標 = dev の 30 人(d37・d85_work/en_TTT をそのまま使う)。操作者 = female-dataset の実女声(全ての目標・D237 の REAL_POOL 150・D238 の当て 400 を除く・Random(241))を目標ごとに 2 人
(元の声 8 秒 = 操作者の真ん中のクリップから・操作者の参照 22 秒以上 = 残りから)= p0000〜p0059。自己変換 = 目標本人の参照に使っていない末尾のクリップ 8 秒・操作者の登録 = 目標の登録 = p0100〜p0129。
    uv run python d241_ffset.py --s <dev の親> --ff <dev の親>/devff
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
    a = ap.parse_args()
    S, F = Path(a.s), Path(a.ff)
    dev = S / "dev"
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (F / d).mkdir(parents=True, exist_ok=True)
    for f in sorted((dev / "d37").glob("cen_*.wav")):
        shutil.copy(f, F / "d37" / f.name)
    for f in sorted((dev / "d37/_work").glob("ref_*.wav")):
        shutil.copy(f, F / "d37/_work" / f.name)
    for e in sorted((dev / "d85_work").glob("en_*")):
        if not (F / "d85_work" / e.name).exists():
            shutil.copytree(e, F / "d85_work" / e.name)
    shutil.copy(dev / "excl.txt", F / "excl.txt")
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    excl = set((S / "excl.txt").read_text().split())
    for f in list(S.glob("*/devset.json")) + [S / "eval_targets.json"]:
        excl |= set(json.loads(f.read_text())["targets"])
    excl |= {r["id"] for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"] if r["grp"] == "REAL_POOL"}
    base = ROOT / "female-dataset"
    fit238 = []
    pool238 = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(238).shuffle(pool238)
    for spk in pool238:
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or sf.info(str(ws[len(ws) // 2])).duration < 3:
            continue
        fit238.append(spk)
        if len(fit238) == 400:
            break
    excl |= set(fit238)
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(241).shuffle(pool)
    log = []
    pi = 0
    it = iter(pool)
    for ti in range(len(targets)):
        for _ in range(2):
            while True:
                spk = next(it)
                ws = sorted((base / spk).glob("*.wav"))
                durs = [sf.info(str(w)).duration for w in ws]
                if sum(durs) >= 32.0 and min(sf.info(str(w)).samplerate for w in ws[:3]) >= 44100:
                    break
            mid = len(ws) // 2
            order = list(range(mid, len(ws))) + list(range(0, mid))
            src, tl = [], 0.0
            for k in order:
                if tl >= 8.0:
                    break
                src.append(k); tl += durs[k]
            if tl < 8.0:
                continue
            rest = [k for k in range(len(ws)) if k not in src]
            ref, rl = [], 0.0
            for k in rest:
                if rl >= 22.0:
                    break
                ref.append(k); rl += durs[k]
            if rl < 20.0:
                continue
            sf.write(F / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", np.concatenate([load48(ws[k]) for k in src])[: 8 * 48000].astype(np.float32), 48000)
            opw = F / "d56/_work" / f"op_{pi:04d}.wav"
            sf.write(opw, np.concatenate([load48(ws[k]) for k in ref]).astype(np.float32), 48000)
            opd = F / "d85_work" / f"op_{pi:04d}"
            if not (opd / "enroll.json").exists():
                run([BIN, "artic-enroll", "-r", opw, "-o", opd])
            log.append({"pi": pi, "ti": ti, "kind": "FF", "spk": spk, "src": [ws[k].name for k in src], "ref": [ws[k].name for k in ref], "target": targets[ti]})
            pi += 1
    for ti, t in enumerate(targets):
        ws = sorted((base / t).glob("*.wav"))
        durs = [sf.info(str(w)).duration for w in ws]
        nref, acc = 0, 0.0
        while acc < 20.0 and nref < len(ws):
            acc += durs[nref]; nref += 1
        src, tl = [], 0.0
        for k in range(len(ws) - 1, nref - 1, -1):
            if tl >= 8.0:
                break
            src.append(k); tl += durs[k]
        if tl < 8.0:
            continue
        p = 100 + ti
        sf.write(F / "d56" / f"p{p:04d}_t{ti:03d}_SRC.wav", np.concatenate([load48(ws[k]) for k in sorted(src)])[: 8 * 48000].astype(np.float32), 48000)
        shutil.copy(F / "d37/_work" / f"ref_{ti:03d}.wav", F / "d56/_work" / f"op_{p:04d}.wav")
        if not (F / "d85_work" / f"op_{p:04d}").exists():
            shutil.copytree(F / "d85_work" / f"en_{ti:03d}", F / "d85_work" / f"op_{p:04d}")
        log.append({"pi": p, "ti": ti, "kind": "AA", "spk": t, "src": [ws[k].name for k in sorted(src)], "target": t})
    ops = {r["spk"] for r in log if r["kind"] == "FF"}
    assert not (ops & excl), "操作者が除外の集合と重なる"
    (F / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "female_to_female"}, ensure_ascii=False, indent=1))
    print("FF", sum(r["kind"] == "FF" for r in log), "AA", sum(r["kind"] == "AA" for r in log), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
