"""cond_gan の診断(diag)のデータ(current/cond_gan.md・results/diag_condgan1/prereg.yaml): 新しい前段(--causal-ct)の条件(31 行)と C1 の事後(200)を Rust で書き出す(描画は省く)。
女声(本物)= female-dataset の評価外 600 人 × 2 クリップ・男声(入力)= male_tts_v4l の m030〜m047(評価の組の操作者 m000〜m029 を除く)+ VCTK の男声 30 人(3 文ずつつないだ 10 本)。
評価 = dev の男声 → 女声 30 組(操作者の包絡のまま・高さと周期性を目標へ = 学習の入力と同じ形)。
    uv run python d273_data.py --s <dev の親> --out <dev の親>/condgan
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
        if len(meta["fem"]) >= 2 * a.n_fem:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 4 or sf.info(str(ws[0])).samplerate < 44100:
            continue
        for w in (ws[len(ws) // 2], ws[len(ws) // 3]):
            if sf.info(str(w)).duration < 3.0:
                continue
            k = f"{spk}_{w.stem}"
            jobs.append(([BIN, "artic"] + SELF + ["-i", w], O / "fem" / f"{k}.cond", O / "fem" / f"{k}.front"))
            meta["fem"].append({"key": k, "spk": spk})
    mt = ROOT / "data/male_tts_v4l"
    for si in range(30, 48):
        spk = f"m{si:03d}"
        ws = sorted((mt / spk).glob("t*.wav"))
        for i in range(0, len(ws) - 1, 2):
            k = f"{spk}_{i:02d}"
            wp = O / "wav" / f"{k}.wav"
            if not wp.exists():
                sf.write(wp, np.concatenate([load48(w) for w in ws[i:i + 2]]).astype(np.float32), 48000)
            jobs.append(([BIN, "artic"] + SELF + ["-i", wp], O / "male" / f"{k}.cond", O / "male" / f"{k}.front"))
            meta["male"].append({"key": k, "spk": spk, "src": "tts"})
    vc = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
    males = [l.split()[0] for l in (vc / "speaker-info.txt").read_text().splitlines()[1:] if len(l.split()) > 2 and l.split()[2] == "M"]
    for sid in males[:30]:
        ws = sorted((vc / "wav48" / f"p{sid}").glob("*.wav"))[:30]
        for i in range(0, len(ws) - 2, 3):
            k = f"p{sid}_{i:02d}"
            wp = O / "wav" / f"{k}.wav"
            if not wp.exists():
                sf.write(wp, np.concatenate([load48(w) for w in ws[i:i + 3]]).astype(np.float32), 48000)
            jobs.append(([BIN, "artic"] + SELF + ["-i", wp], O / "male" / f"{k}.cond", O / "male" / f"{k}.front"))
            meta["male"].append({"key": k, "spk": f"p{sid}", "src": "vctk"})
    D = S / "dev"
    for f in sorted((D / "d56").glob("p00*_SRC.wav"))[:30]:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        cmd = [BIN, "artic", "--enroll", D / "d85_work" / f"en_{ti:03d}", "--operator-enroll", D / "d85_work" / f"op_{pi:04d}", "--mix", "1", "--med-from-operator",
               "--causal-ct", "--per-p75", "--f4", "results/f4_1/export", "--rvoc", "results/rvoc3hi/export", "-i", f]
        jobs.append((cmd, O / "eval" / f"{k}.cond", O / "eval" / f"{k}.front"))
        meta["eval"].append({"key": k, "pi": pi, "ti": ti})
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
