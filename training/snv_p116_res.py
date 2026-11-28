"""段 P116(results/snv_p116/prereg.yaml)の準備(学習なし): 「鳴りだけの信号」= 音から、前段の包絡(25 行)で声道の色と音量を抜いた残差(スペクトルを平らにした波形)。STFT(2048・hop 240)の上で、フレームごとの包絡の逆数を掛けて戻す。
出力 = data/snv_p116_res/<e1|e1nv>/<声>/<発話>.flac(48 kHz・PCM_16・尖頭値 0.5 に正規化した後の倍率は台帳に残さない = 学習では区間ごとに二乗平均で正規化する)。
    cd training && uv run --no-sync python snv_p116_res.py [procs=10]
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import pae as PA
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p116_res"
NFFT = 2048


def whiten(x: np.ndarray, env: np.ndarray) -> np.ndarray:
    fb, dct = PA._np_consts()
    e = env.astype(np.float64).copy()
    e = np.concatenate([e[:, :1], e[:, :1], e], 1)
    e = (e[:, 2:] + e[:, 1:-1] + e[:, :-2]) / 3.0
    g_mel = np.exp(np.clip(-(dct[:25].T @ e), -12.0, 12.0))
    den = fb.sum(0)[:, None]
    g = np.where(den > 1e-9, (fb.T @ g_mel) / np.maximum(den, 1e-9), 0.0)
    X = librosa.stft(x.astype(np.float32), n_fft=NFFT, hop_length=P.H, window="hann", center=True)
    m = min(X.shape[1], g.shape[1])
    Y = np.zeros_like(X)
    Y[:, :m] = X[:, :m] * g[:, :m]
    return librosa.istft(Y, hop_length=P.H, window="hann", center=True, length=len(x)).astype(np.float32)


def work(job: tuple) -> int:
    w, f, dst = (Path(q) for q in job)
    if dst.exists():
        return 0
    x, sr = sf.read(str(w), dtype="float32")
    assert sr == P.SR
    r = whiten(x, np.load(f)["env"])
    r = r / (np.abs(r).max() + 1e-9) * 0.5
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.flac")
    sf.write(str(tmp), r, P.SR, subtype="PCM_16", format="FLAC")
    tmp.rename(dst)
    return 1


def main() -> int:
    from train_snv_p97 import items_for
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    sp = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    jobs = [(str(w), str(f), str(OUT / f.relative_to(T.FEAT).with_suffix(".flac"))) for w, f in items_for(sp["train"] + sp["val"])]
    with Pool(int(opt.get("procs", 10))) as pool:
        n = sum(pool.imap_unordered(work, jobs, chunksize=16))
    (OUT / "manifest.json").write_text(json.dumps({"stage": "P116", "what": "前段の包絡で声道の色と音量を抜いた残差(STFT 2048 / hop 240・包絡は 3 点でならす)", "n": len(jobs)}, ensure_ascii=False))
    print("done", n, "/", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
