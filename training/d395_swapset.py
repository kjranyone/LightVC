"""D395 の組(results/conv_p0/d395_prereg.yaml): VCTK の女声 F(目標・実録音)40 人 × 2 区間。区間 = 全話者共通の文(001〜024)の連続で F が 8 s 以上。
同じ文の区間を別の女声 F2 と男声 M(学習の交差の標本に入っていない 17 人を巡回)も読む。F の登録 = F の共通でない発話の末尾 24 s(Rust artic-enroll)。
F は自分自身の条件(--enroll と --operator-enroll に F の登録)で、F2 と M は事後だけを使うので F の登録で書き出す(事後は登録に依らないことを 1 組で検査)。
    uv run python d395_swapset.py --out <dev の親>/d395
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN, job
from d274_evaldump import COM
from d326_vctkset import CONDGAN, V, run, trim
from d6_unit_cov import load48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_f", type=int, default=40)
    a = ap.parse_args()
    O = Path(a.out)
    for d in ("wav", "en", "dump", "ref"):
        (O / d).mkdir(parents=True, exist_ok=True)
    rows = [l.split() for l in (V / "speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    gender = {("p" + r[0]) if not r[0].startswith("p") else r[0]: r[2] for r in rows}
    used = {x["spk"] for x in json.loads(CONDGAN.read_text())["male"]}
    have = {d.name for d in (V / "wav48").iterdir() if d.is_dir()}
    msp = sorted(s for s, g in gender.items() if g == "M" and s not in used and s in have)
    fsp = sorted(s for s, g in gender.items() if g == "F" and s in have)
    rng = random.Random(395)
    rng.shuffle(fsp)
    tgt, part = fsp[: a.n_f], fsp[a.n_f:] + fsp[: a.n_f]
    shared = lambda s: {w.stem.split("_")[1]: w for w in (V / "wav48" / s).glob("*.wav") if w.stem.split("_")[1] <= "024"}
    cache: dict = {}

    def sh(s: str) -> dict:
        if s not in cache:
            cache[s] = shared(s)
        return cache[s]

    triples, mi = [], 0
    for i, f in enumerate(tgt):
        starts = [1, 5, 9, 13, 17]
        rng.shuffle(starts)
        got = 0
        for k0 in starts:
            if got == 2:
                break
            f2 = next(p for p in part[i + 1:] + part if p != f and all(f"{k:03d}" in sh(p) for k in range(k0, k0 + 3)))
            m = msp[mi % len(msp)]
            ids, tl = [], 0.0
            k = k0
            while tl < 8.0 and k <= 24:
                s_ = f"{k:03d}"
                if s_ not in sh(f) or s_ not in sh(f2) or s_ not in sh(m):
                    break
                tl += len(trim(load48(sh(f)[s_]))) / 48000
                ids.append(s_)
                k += 1
            if tl < 8.0:
                continue
            mi += 1
            got += 1
            key = f"t{i:03d}_b{k0:02d}"
            for role, s in (("F", f), ("F2", f2), ("M", m)):
                x = np.concatenate([trim(load48(sh(s)[q])) for q in ids]).astype(np.float32)
                sf.write(O / "wav" / f"{key}_{role}.wav", x, 48000)
            triples.append({"key": key, "ti": i, "F": f, "F2": f2, "M": m, "ids": ids})
        en = O / "en" / f
        if not (en / "enroll.json").exists():
            ws = sorted(w for w in (V / "wav48" / f).glob("*.wav") if w.stem.split("_")[1] > "024")
            ref, tl = [], 0.0
            while tl < 24.0 and ws:
                x = trim(load48(ws.pop()))
                ref.insert(0, x)
                tl += len(x) / 48000
            rw = O / "ref" / f"ref_{f}.wav"
            sf.write(rw, np.concatenate(ref).astype(np.float32), 48000)
            run([BIN, "artic-enroll", "-r", rw, "-o", en])
    print("triples", len(triples), "targets", len({t["F"] for t in triples}), "males", len({t["M"] for t in triples}), flush=True)
    jobs = []
    for t in triples:
        en = O / "en" / t["F"]
        for role in ("F", "F2", "M"):
            w = O / "wav" / f"{t['key']}_{role}.wav"
            jobs.append(([BIN, "artic", "--enroll", en, "--operator-enroll", en, "--med-from-operator"] + COM + ["-i", w], O / "dump" / f"{t['key']}_{role}.cond", O / "dump" / f"{t['key']}_{role}.front"))
    t0 = triples[0]
    other = O / "en" / triples[-1]["F"]
    jobs.append(([BIN, "artic", "--enroll", other, "--operator-enroll", other, "--med-from-operator"] + COM + ["-i", O / "wav" / f"{t0['key']}_M.wav"], O / "dump" / "chk_M.cond", O / "dump" / "chk_M.front"))
    with Pool(6) as pl:
        list(pl.imap_unordered(job, jobs))
    a_ = np.fromfile(O / "dump" / f"{t0['key']}_M.front", "<f4").reshape(-1, 224)[:, :200]
    b_ = np.fromfile(O / "dump" / "chk_M.front", "<f4").reshape(-1, 224)[:, :200]
    n = min(len(a_), len(b_))
    print("事後は登録に依らない: max|差|", float(np.abs(a_[:n] - b_[:n]).max()), "フレーム", len(a_), len(b_), flush=True)
    (O / "set.json").write_text(json.dumps({"triples": triples, "msp": msp}, indent=1))
    print("dumped", sum(Path(c).exists() for _, c, _ in jobs), "/", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
