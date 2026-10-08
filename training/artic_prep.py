"""構音の逆推定 段 2 の学習データ(current/artic_inv.md §4): 実音声の区間と、WORLD で f0 だけ変えた再合成の対。

区間ごと(2.0s・48kHz)に 3 信号: real(実音声)・w0(WORLD 再合成・f0 そのまま)・ws(WORLD 再合成・f0 × 2^(s/12)・s ~ U(±1, ±6))。
w0 と ws は同じスペクトル包絡(cheaptrick)と非周期性(d4c)= 声道は同じで f0 だけ違う = 構音の座標が一致すべき対。
信号ごとに保存: f0(harvest・100fps・損失側の倍音の位置)・yin(生の因果 YIN・推定器の入力 = 推論と同じ・240 hop を 2 つおきに間引き = 100fps)・
倍音の振幅 am [T, 64](dB・artic_fit.harmonic_obs)・因果 log-mel [128, T](窓 1024 左寄せ・hop 480)。
データ: 女声 = train_ddsp_vc.index の train 話者・男声 = train_zsvc.male_index(vctk=True) の train。発話統計は使わない。

    uv run python artic_prep.py --n 20000 --procs 8 --out ../data/artic_inv
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

SR = 48000
HOP = 480
SEG = 2 * SR
T = SEG // HOP
WIN = 1024


def causal_logmel(x: np.ndarray, fb: np.ndarray) -> np.ndarray:
    pad = np.concatenate([np.zeros(WIN - HOP), x])
    fr = np.lib.stride_tricks.sliding_window_view(pad, WIN)[::HOP][:T] * np.hanning(WIN)
    mag = np.abs(np.fft.rfft(fr, n=2048, axis=1))
    return np.log(np.maximum(mag @ fb.T, 1e-5)).T.astype(np.float16)


def analyze(x: np.ndarray, fb: np.ndarray, f0: np.ndarray) -> dict:
    """f0 は与える(real は WORLD 分析の harvest・w0 は同じ・ws は × 比 = 再合成に使った値そのもの)。"""
    import artic_dsp as D
    import artic_fit as AF
    f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T]
    _, am, mk = AF.harmonic_obs(x, f0)
    am = np.where(mk, am, -120.0)
    y, _ = D.causal_yin(x, voi_max=0.45)
    y = y[: 2 * T: 2] if len(y) >= 2 * T else np.pad(y[::2], (0, T - len(y[::2])))
    return {"f0": f0.astype(np.float32), "yin": y.astype(np.float32), "am": am.astype(np.float16), "mel": causal_logmel(x, fb)}


def world_pair(x: np.ndarray, st: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import librosa
    import pyworld
    x16 = librosa.resample(x, orig_sr=SR, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=900, frame_period=HOP / SR * 1000)
    f0 = pyworld.stonemask(x16, f0, t, 16000)
    sp = pyworld.cheaptrick(x16, f0, t, 16000)
    ap = pyworld.d4c(x16, f0, t, 16000)
    outs = []
    for s in (0.0, st):
        y16 = pyworld.synthesize(f0 * 2 ** (s / 12), sp, ap, 16000, HOP / SR * 1000)
        y = librosa.resample(y16, orig_sr=16000, target_sr=SR)
        y = np.pad(y, (0, max(0, SEG - len(y))))[:SEG]
        outs.append(y)
    return outs[0], outs[1], f0


def worker(args: tuple) -> int:
    wid, jobs, out = args
    import nvoc as N
    from train_ddsp_vc import load48
    fb = N.mel_fb().numpy().astype(np.float64)
    rng = random.Random(wid * 7919)
    done = 0
    for i, (path, gender) in jobs:
        dst = out / f"{i:06d}.npz"
        if dst.exists():
            continue
        try:
            x = load48(path).astype(np.float64)
            if len(x) < SEG + SR // 2:
                continue
            s0 = rng.randrange(SR // 4, len(x) - SEG)
            x = x[s0:s0 + SEG]
            if np.sqrt((x ** 2).mean()) < 3e-3:
                continue
            x = x / max(1.0, np.abs(x).max() / 0.95)
            st = rng.choice((-1, 1)) * rng.uniform(1.0, 6.0)
            w0, ws, f0 = world_pair(x, st)
            r, a0, a1 = analyze(x, fb, f0), analyze(w0, fb, f0), analyze(ws, fb, f0 * 2 ** (st / 12))
            if (r["f0"] > 0).mean() < 0.15:
                continue
            np.savez_compressed(dst, gender=np.int8(gender), st=np.float32(st),
                                **{f"{k}_{n}": v for n, d in (("real", r), ("w0", a0), ("ws", a1)) for k, v in d.items()})
            done += 1
        except Exception as e:
            print("skip", path, e, flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data/artic_inv"))
    ap.add_argument("--male_frac", type=float, default=0.4)
    a = ap.parse_args()
    from train_ddsp_vc import index
    from train_zsvc import male_index
    spk, tr, _ = index()
    fem = [w for k in tr for _, w, _ in spk[k] if not k.endswith("/unknown")]
    mtr, _ = male_index(vctk=True)
    mal = [w for _, w in mtr]
    rng = random.Random(0)
    nm = int(a.n * a.male_frac)
    picks = [(p, 0) for p in rng.sample(fem, a.n - nm)] + [(p, 1) for p in rng.choices(mal, k=nm)]
    rng.shuffle(picks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(picks))
    chunks = [(w, jobs[w::a.procs], out) for w in range(a.procs)]
    print("female", len(fem), "male", len(mal), "jobs", len(jobs), flush=True)
    from multiprocessing import Pool
    with Pool(a.procs) as pool:
        n = sum(pool.map(worker, chunks))
    print("done", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
