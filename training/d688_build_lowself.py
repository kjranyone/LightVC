"""D688(amendment_1): 低い声の実女声(音声の dio + stonemask の有声の中央値 170〜230 Hz)の自己変換の組 d688_lowself2(1 回目の d688_lowself は特徴の f0 のオクターブ誤りで無効)。生成器の学習の話者・官能指数 v1 の学習の本物を除く。
d681_build_self.py と同じ型: 目標の登録 = 本人の名前順の録音を 20 s まで・入力 = 本人の別の録音 8 s・前置き = 登録の録音の末尾 12 s。規則は results/conv_p0/d688/prereg.yaml。
    uv run python d688_build_lowself.py
"""
from __future__ import annotations

import json
import random
import subprocess
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

import train_ppgvoc as TP

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
S = H / "d688_lowself2"
FD = ROOT / "female-dataset"
SR = 48000
BIN = ROOT / "target/release/lightvc-app"


def load48(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float64", always_2d=True)
    x = x.mean(1)
    return x if sr == SR else soxr.resample(x, sr, SR)


def f0_audio(ws: list) -> float:
    import pyworld as pw
    v = []
    for w in ws:
        x, sr = sf.read(str(w), dtype="float64", always_2d=True)
        x = np.ascontiguousarray(soxr.resample(x.mean(1), sr, 16000))
        f, t = pw.dio(x, 16000, f0_floor=70.0, f0_ceil=1000.0, frame_period=5.0)
        f = pw.stonemask(x, f, t, 16000)
        v.append(f[f > 0])
    v = np.concatenate(v)
    return float(np.median(v)) if len(v) > 100 else float("nan")


def enroll(ref: Path, out: Path, rr: bool) -> None:
    if (out / "enroll.bin").exists() and (not rr or (out / "rr_mem" / "mem_v1.f32").exists()):
        return
    cmd = [str(BIN), "artic-enroll", "-r", str(ref), "-o", str(out)] + (["--rr", "results/d587_rbar/export"] if rr else [])
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr[-300:]


def main() -> int:
    _, train_spk, _ = TP.load_rows({h[0] for h in TP.e2e_held()}, True)
    det = {Path(r["path"]).parent.name for r in json.loads((ROOT / "results/conv_p0/d678/lists.json").read_text())["real_train"]}
    excl = set(train_spk) | det
    cand = []
    for d in sorted(FD.iterdir()):
        if not d.is_dir() or d.name in excl:
            continue
        ws = sorted(d.glob("*.wav"))
        dur = [sf.info(str(w)).duration for w in ws]
        if sum(dur) < 34.0:
            continue
        cand.append((d.name, ws[:2]))
    with Pool(12) as pl:
        med = pl.map(f0_audio, [w for _, w in cand])
    cand = [(s, m) for (s, _), m in zip(cand, med) if np.isfinite(m) and 170.0 <= m <= 230.0]
    rng = random.Random(688)
    rng.shuffle(cand)
    pick = sorted(cand[:30])
    for d in ("d37/_work", "d56/_work", "d85_work", "rr_en"):
        (S / d).mkdir(parents=True, exist_ok=True)
    pairs = []
    for t, (spk, f0m) in enumerate(pick):
        ws = sorted((FD / spk).glob("*.wav"))
        ref, used, tot = [], 0, 0.0
        for w in ws:
            if tot >= 20.0:
                break
            x = load48(w)
            ref.append(x)
            tot += len(x) / SR
            used += 1
        src, tl = [], 0.0
        for w in ws[used:]:
            if tl >= 8.0:
                break
            x = load48(w)
            src.append(x)
            tl += len(x) / SR
        if tl < 8.0:
            continue
        key = f"p{t:04d}_t{t:03d}"
        rp = S / "d37/_work" / f"ref_{t:03d}.wav"
        sf.write(rp, np.concatenate(ref).astype(np.float32), SR)
        op = S / "d56/_work" / f"op_{t:04d}.wav"
        if not op.exists():
            op.symlink_to(rp)
        enroll(rp, S / "rr_en" / f"en_{t:03d}", True)
        enroll(rp, S / "d85_work" / f"op_{t:04d}", False)
        x = np.concatenate(src)[: 8 * SR]
        sf.write(S / "d56" / f"{key}_SRC.wav", x.astype(np.float32), SR)
        pre = np.concatenate(ref)[-12 * SR:]
        sf.write(S / "d56/_work" / f"pre_{key}.wav", np.concatenate([np.pad(pre, (12 * SR - len(pre), 0)), x]).astype(np.float32), SR)
        pairs.append({"pi": t, "ti": t, "kind": "LOWSELF", "spk": spk, "f0med_audio": round(f0m, 1), "n_ref_files": used})
    (S / "devset.json").write_text(json.dumps({"targets": [p["spk"] for p in pairs], "pairs": pairs, "kind": "LOWSELF"}, indent=1))
    rep = {"n_candidates": len(cand), "n_pairs": len(pairs), "f0med_audio_median": round(float(np.median([p["f0med_audio"] for p in pairs])), 1)}
    print(rep, flush=True)
    (ROOT / "results/conv_p0/d688/build2.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
