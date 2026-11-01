"""学習の話者ごとの参照の長時間平均スペクトル(LTAS・帯域の相対 dB)を、表と同じ参照のファイル(table_files・20 s)から前計算する(話者差の帯域の一致の損失・D399 の表への追従の弱さ)。
帯域 = BANDS(12 帯・80 Hz〜20 kHz)・各帯のパワーを全フレームで合計 → 全帯の合計で割る → 10 log10。学習の損失と同じ関数(ltas_np / train_ppgvoc.ltas_torch)。
    uv run python build_ltas.py --out ../data/rvoc_tables_real_ltas.npz --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))

BANDS = (80, 200, 400, 700, 1000, 1500, 2200, 3200, 4500, 6500, 9000, 13000, 20000)
NFFT, HOPL = 2048, 480


def ltas_np(x: np.ndarray, sr: int = 48000) -> np.ndarray:
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    w = np.hanning(NFFT)
    n = 1 + max(0, (len(x) - NFFT) // HOPL)
    P = np.zeros(len(f))
    for i in range(n):
        seg = x[i * HOPL:i * HOPL + NFFT]
        if len(seg) < NFFT:
            break
        P += np.abs(np.fft.rfft(seg * w)) ** 2
    b = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in zip(BANDS[:-1], BANDS[1:])])
    return (10 * np.log10(b / b.sum() + 1e-12)).astype(np.float32)


def one(args: tuple) -> tuple:
    spk, files = args
    xs = []
    for wav in sorted(files):
        x, sr = sf.read(wav, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != 48000:
            x = resample_poly(x, 48000, sr).astype(np.float32)
        xs.append(x)
    return spk, ltas_np(np.concatenate(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import train_ppgvoc as P
    P.TAB = P.ROOT / "data/rvoc_tables_real"
    rows = json.loads(P.MANIFEST.read_text())["rows"]
    by: dict = {}
    for r in rows:
        if r.get("f0") and r["src"].startswith("real"):
            by.setdefault(r["spk"], []).append((r["wav"], r["src"], r["dur"]))
    _, spks, counts = P.load_rows({h[0] for h in P.e2e_held()}, True)
    jobs = [(s, P.table_files(by[s])[0]) for s in spks]
    with Pool(a.workers) as pl:
        res = dict(pl.imap_unordered(one, jobs, chunksize=8))
    np.savez(a.out, bands=np.array(BANDS), **res)
    M = np.stack(list(res.values()))
    print("speakers", len(res), "band sd across speakers (dB)", np.round(M.std(0), 2).tolist(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
