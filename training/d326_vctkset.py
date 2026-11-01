"""D326 の組(results/conv_p0/d326_prereg.yaml): 目標 = dev の 30 人・操作者 = VCTK の実録音(英語)の女声(学習に未使用の 30 人・p0000〜p0059・目標ごとに 2 本)と
男声(実録音が学習の交差の標本に入っていない 17 人・p0100〜p0159)。発話ごとに前後の無音を削ってつなぐ・元の声 8 秒・操作者の参照 = 話者ごとに末尾の発話から 24 秒以上(元の声と重ならない)。
    uv run python d326_vctkset.py --s <dev の親> --out <dev の親>/devvctk
--matched: 内容をそろえる(D326c)= 元の声を全話者が同じ文を読む 001〜024 から作り、同じ番号 j の女声と男声は同じ文の区間(始まり 001・005・009・013・017・021 の 6 区間)を読む。
    uv run python d326_vctkset.py --s <dev の親> --out <dev の親>/devvctkm --matched
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
V = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
CONDGAN = Path("/home/kojirotanaka/lightvc_scratch_20261013/condgan/meta.json")


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-400:]}")


def trim(x: np.ndarray, sr: int = 48000, db: float = 35.0, pad: float = 0.1) -> np.ndarray:
    h = int(0.02 * sr)
    n = len(x) // h
    if n < 3:
        return x
    e = 10 * np.log10((x[: n * h].reshape(n, h) ** 2).mean(1) + 1e-12)
    on = np.where(e > e.max() - db)[0]
    a, b = max(0, on[0] * h - int(pad * sr)), min(len(x), (on[-1] + 1) * h + int(pad * sr))
    return x[a:b]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matched", action="store_true")
    a = ap.parse_args()
    S, O = Path(a.s), Path(a.out)
    dev = S / "dev"
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (O / d).mkdir(parents=True, exist_ok=True)
    for f in sorted((dev / "d37").glob("cen_*.wav")):
        shutil.copy(f, O / "d37" / f.name)
    for f in sorted((dev / "d37/_work").glob("ref_*.wav")):
        shutil.copy(f, O / "d37/_work" / f.name)
    for e in sorted((dev / "d85_work").glob("en_*")):
        if not (O / "d85_work" / e.name).exists():
            shutil.copytree(e, O / "d85_work" / e.name)
    shutil.copy(dev / "excl.txt", O / "excl.txt")
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    rows = [l.split() for l in (V / "speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    gender = {("p" + r[0]) if not r[0].startswith("p") else r[0]: r[2] for r in rows}
    used = {x["spk"] for x in json.loads(CONDGAN.read_text())["male"]}
    have = {d.name for d in (V / "wav48").iterdir() if d.is_dir()}
    msp = sorted(s for s, g in gender.items() if g == "M" and s not in used and s in have)
    fsp = sorted(s for s, g in gender.items() if g == "F" and s in have)
    random.Random(326).shuffle(fsp)
    fsp = fsp[:30]
    print("VCTK 男声", len(msp), msp, "| 女声", len(fsp), flush=True)
    cache: dict = {}
    cursor: dict = {}

    def utt(spk: str) -> list:
        if spk not in cache:
            ws = sorted((V / "wav48" / spk).glob("*.wav"))
            ref, tl = [], 0.0
            while tl < 24.0 and ws:
                w = ws.pop()
                x = trim(load48(w))
                ref.insert(0, x)
                tl += len(x) / 48000
            cache[spk] = (ws, np.concatenate(ref))
            cursor[spk] = 0
        return cache[spk][0]

    starts = [1, 5, 9, 13, 17, 21]
    block, taken = {}, set()
    for j in range(60):
        f_, m_ = fsp[j % len(fsp)], msp[j % len(msp)]
        for d in range(len(starts)):
            b = (j + d) % len(starts)
            if (f_, b) not in taken and (m_, b) not in taken:
                break
        taken |= {(f_, b), (m_, b)}
        block[j] = b
    log = []
    for kind, spks, p0 in (("FE", fsp, 0), ("ME", msp, 100)):
        for j in range(60):
            ti, spk = j // 2, spks[j % len(spks)]
            ws = utt(spk)
            src, tl = [], 0.0
            if a.matched:
                shared = {w.stem.split("_")[1]: w for w in ws if w.stem.split("_")[1] <= "024"}
                k = starts[block[j]]
                while tl < 8.0 and f"{k:03d}" in shared:
                    x = trim(load48(shared[f"{k:03d}"]))
                    src.append(x)
                    tl += len(x) / 48000
                    k += 1
                if tl < 8.0:
                    print("skip", kind, spk, j, round(tl, 1), flush=True)
                    continue
            while tl < 8.0:
                x = trim(load48(ws[cursor[spk]]))
                cursor[spk] += 1
                src.append(x)
                tl += len(x) / 48000
            pi = p0 + j
            sf.write(O / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", np.concatenate(src)[: 8 * 48000].astype(np.float32), 48000)
            spd = O / "d85_work" / f"spk_{spk}"
            if not (spd / "enroll.json").exists():
                opw = O / "d56/_work" / f"ref_{spk}.wav"
                sf.write(opw, cache[spk][1].astype(np.float32), 48000)
                run([BIN, "artic-enroll", "-r", opw, "-o", spd])
            opd = O / "d85_work" / f"op_{pi:04d}"
            if not opd.exists():
                opd.symlink_to(spd.name)
            log.append({"pi": pi, "ti": ti, "kind": kind, "spk": spk, "n_src_utt": len(src), "block": starts[block[j]] if a.matched else None, "target": targets[ti]})
    (O / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "vctk_real_to_female"}, ensure_ascii=False, indent=1))
    print("FE", sum(r["kind"] == "FE" for r in log), "ME", sum(r["kind"] == "ME" for r in log), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
