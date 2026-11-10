"""A7a(synth_nam_vc §5): 演奏の声門の次元(信号処理・因果)。44.1 kHz・hop 512・左寄せの窓 2048(先読み 0)で [T, 9] を作る。
次元 = tilt・hf・cpp・h1h2・shr・jit・shim・mod・flat(results/snv_a7a/prereg.yaml)。高さは SPARC の pitch(mel のフレームへ補間した値)を受け取る。
出力 = data/snv_e1_feat/glot/e1/{voice}/{t}.npy・glot/e1nv/{voice}/{cat}_{k}.npy(float32・[T, 9]・T = mel のフレーム数)。
eval = A7 の評価の音: glot/n0r・glot/a1diag(D0)・glot/a1src/tref_{v}(登録 = wav44/a0enroll と同じ音)・glot/a1src/{源}(本物の男声の 10 s を元の mp3 から切り直す・T = 862)。
A8a = glot/a1src/{uirou,namikawa}_enroll(本物の男声の操作者の登録 = 先頭 30 s・T = 2584・試験の区間と重ならない)も eval で作る。
    cd training && uv run --no-sync python snv_glot.py test
    cd training && uv run --no-sync python snv_glot.py
    cd training && uv run --no-sync python snv_glot.py eval
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_e1_feat"
SR = 44100
HOP = 512
WIN = 2048
SUB = 128
SRC_T = int(10 * SR / HOP) + 1
ENR_T = int(30 * SR / HOP) + 1
G_DIM = 9
NAMES = ("tilt", "hf", "cpp", "h1h2", "shr", "jit", "shim", "mod", "flat")
EPS = 1e-10
RUN = 6
MOD_BAND = (8.0, 32.0)
MOD_AVG = 0.1
_F = np.fft.rfftfreq(WIN, 1 / SR)
_W = np.hanning(WIN + 1)[:WIN].astype(np.float64)
_EDGES = np.geomspace(100.0, 11000.0, 25)
_LO = np.searchsorted(_F, _EDGES[:-1])
_HI = np.maximum(np.searchsorted(_F, _EDGES[1:]), _LO + 1)
_XC = np.log2(np.sqrt(_EDGES[:-1] * _EDGES[1:]))
_XC = _XC - _XC.mean()
_Q = np.arange(WIN // 2)
_QLO, _QHI = int(SR / 1000), int(SR / 60)
_MB = butter(2, MOD_BAND, btype="band", fs=SR / SUB)


def frames(y: np.ndarray, n: int) -> np.ndarray:
    yp = np.concatenate([np.zeros(WIN - HOP), y.astype(np.float64), np.zeros(max(0, n * HOP - len(y)))])
    idx = np.arange(n)[:, None] * HOP + np.arange(WIN)[None]
    return yp[idx] * _W


def running_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([np.zeros(k), x]))
    cnt = np.minimum(np.arange(1, len(x) + 1), k)
    return (c[k:] - c[: len(x)]) / cnt


def peak_db(L: np.ndarray, f0: np.ndarray, k: float) -> np.ndarray:
    half = np.maximum(0.25 * f0, 1.5 * _F[1])[:, None]
    m = np.abs(_F[None] - k * f0[:, None]) <= half
    return np.where(m, L, -np.inf).max(1)


def mod_track(y: np.ndarray, n: int) -> np.ndarray:
    e = running_mean(np.concatenate([y.astype(np.float64), np.zeros(max(0, n * HOP - len(y)))]) ** 2, 4 * SUB)
    le = np.log(e[SUB - 1:: SUB] + EPS)
    bp = lfilter(*_MB, le)
    ms = running_mean(bp ** 2, int(round(MOD_AVG * SR / SUB)))
    j = (np.arange(n) + 1) * (HOP // SUB) - 1
    return np.log(ms[j] + 1e-6)


def glot(y: np.ndarray, f0: np.ndarray, n: int) -> np.ndarray:
    f0 = np.clip(np.asarray(f0, dtype=np.float64)[:n], 40.0, 2000.0)
    X = np.fft.rfft(frames(y, n), axis=1)
    P = np.abs(X) ** 2 + EPS
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(P, 1)], 1)
    lb = np.log(cs[:, _HI] - cs[:, _LO] + EPS)
    tilt = (lb - lb.mean(1, keepdims=True)) @ _XC / (_XC @ _XC)
    hf = np.log(P[:, (_F >= 4000) & (_F < 11000)].sum(1)) - np.log(P[:, (_F >= 50) & (_F < 4000)].sum(1))
    Ld = 10 * np.log10(P)
    C = 20 * np.log10(np.abs(np.fft.irfft(Ld, n=WIN, axis=1)[:, : WIN // 2]) + EPS)
    q = _Q[_QLO:].astype(np.float64)
    Cr = C[:, _QLO:]
    qc = q - q.mean()
    b = (Cr - Cr.mean(1, keepdims=True)) @ qc / (qc @ qc)
    line = Cr.mean(1, keepdims=True) + b[:, None] * qc[None]
    cpp = (Cr - line)[:, : _QHI - _QLO + 1].max(1)
    h1, h2 = peak_db(Ld, f0, 1.0), peak_db(Ld, f0, 2.0)
    s1, s3 = peak_db(Ld, f0, 0.5), peak_db(Ld, f0, 1.5)
    lf, le = np.log(f0), np.log(P.sum(1))
    jit = running_mean(np.abs(np.diff(lf, prepend=lf[0])), RUN)
    shim = running_mean(np.abs(np.diff(le, prepend=le[0])), RUN)
    Ph = P[:, (_F >= 3000) & (_F < 8000)]
    flat = np.log(Ph).mean(1) - np.log(Ph.mean(1))
    out = np.stack([tilt, hf, cpp, h1 - h2, 0.5 * (s1 + s3) - 0.5 * (h1 + h2), jit, shim, mod_track(y, n), flat], 1)
    return out.astype(np.float32)


def f0_of(npz: Path, n: int) -> np.ndarray:
    from train_snv_a2 import playing
    return np.exp(playing(npz, n)[:, 12])


def test() -> int:
    import librosa
    y = librosa.load(str(ROOT / "data/snv_e1/ef000/t00.wav"), sr=SR, mono=True)[0]
    n = np.load(FEAT / "mel/e1/ef000/t00.npy").shape[1]
    f0 = f0_of(FEAT / "sparc/e1/ef000/t00.npz", n)
    g = glot(y, f0, n)
    assert g.shape == (n, G_DIM) and np.isfinite(g).all(), g.shape
    z = librosa.load(str(ROOT / "data/snv_e1/ef001/t03.wav"), sr=SR, mono=True)[0]
    for s in (n * HOP // 3, n * HOP // 2 + 77, 2 * n * HOP // 3):
        y2 = y.copy()
        y2[s:] = z[: len(y) - s] if len(z) >= len(y) - s else np.resize(z, len(y) - s)
        keep = (np.arange(n) + 1) * HOP <= s
        g2 = glot(y2, np.where(keep, f0, f0 * 1.3), n)
        d_keep = float(np.abs(g2[keep] - g[keep]).max())
        d_after = np.abs(g2[~keep] - g[~keep]).max(0)
        print("edit at", s, "frames kept", int(keep.sum()), "max diff before", d_keep, "min over dims of max diff after", float(d_after.min()), flush=True)
        assert d_keep == 0.0, d_keep
        assert (d_after > 1e-3).all(), dict(zip(NAMES, d_after.tolist()))
    print("per-dim mean", dict(zip(NAMES, np.round(g.mean(0), 3).tolist())), flush=True)
    print("future invariance PASS (real speech, 3 edit points, all dims move after the edit)", flush=True)
    return 0


def job(args: tuple) -> str:
    wav, mel, sp, out = args
    if out.exists():
        return out.name
    import librosa
    n = np.load(mel).shape[1]
    y = librosa.load(str(wav), sr=SR, mono=True)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, glot(y, f0_of(sp, n), n))
    return out.name


def eval_sources() -> int:
    import librosa
    import soundfile as sf
    from snv_a1_convert import segments
    for name, mp3 in (("uirou", ROOT / "外郎売.mp3"), ("namikawa", ROOT / "namikawa.mp3")):
        enr, tests, sr = segments(mp3)
        for i, seg, n in [("enroll", enr, ENR_T)] + [(i, seg, SRC_T) for i, seg in enumerate(tests)]:
            out = FEAT / f"glot/a1src/{name}_{i}.npy"
            y = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=SR)
            ref16 = sf.read(str(ROOT / f"results/snv_a1/src/{name}_{i}.wav"))[0]
            y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)[: len(ref16)]
            r = float(np.corrcoef(y16, ref16[: len(y16)])[0, 1])
            assert r > 0.99, (name, i, r)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, glot(y, f0_of(FEAT / f"sparc/a1src/{name}_{i}.npz", n), n))
            print(name, i, "corr with the 16 kHz source", round(r, 5), flush=True)
    return 0


def eval_jobs() -> list:
    js = []
    for p in sorted((FEAT / "wav44").glob("n0r__*.wav")):
        n = p.stem[5:]
        js.append((p, FEAT / f"mel/n0r/{n}.npy", FEAT / f"sparc/n0r/{n}.npz", FEAT / f"glot/n0r/{n}.npy"))
    for p in sorted((FEAT / "wav44").glob("a1diag__D0_*.wav")):
        n = p.stem[8:]
        js.append((p, FEAT / f"mel/a1diag/{n}.npy", FEAT / f"sparc/a1diag/{n}.npz", FEAT / f"glot/a1diag/{n}.npy"))
    for p in sorted((FEAT / "wav44").glob("a0enroll__*.wav")):
        v = p.stem[10:]
        if (FEAT / f"sparc/a1src/tref_{v}.npz").exists():
            js.append((p, FEAT / f"mel/a0enroll/{v}.npy", FEAT / f"sparc/a1src/tref_{v}.npz", FEAT / f"glot/a1src/tref_{v}.npy"))
    return js


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        return test()
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        js = eval_jobs()
        print("eval files", len(js), flush=True)
        with ProcessPoolExecutor(12, initializer=_init) as ex:
            list(ex.map(job, js, chunksize=8))
        return eval_sources()
    js = []
    for p in sorted((ROOT / "data/snv_e1").glob("e*/t*.wav")):
        rel = f"{p.parent.name}/{p.stem}"
        mel, sp = FEAT / f"mel/e1/{rel}.npy", FEAT / f"sparc/e1/{rel}.npz"
        if mel.exists() and sp.exists():
            js.append((p, mel, sp, FEAT / f"glot/e1/{rel}.npy"))
    for p in sorted((ROOT / "data/snv_e1nv").glob("e*/*_*.wav")):
        rel = f"{p.parent.name}/{p.stem}"
        mel, sp = FEAT / f"mel/e1nv/{rel}.npy", FEAT / f"sparc/e1nv/{rel}.npz"
        if mel.exists() and sp.exists():
            js.append((p, mel, sp, FEAT / f"glot/e1nv/{rel}.npy"))
    print("files", len(js), flush=True)
    with ProcessPoolExecutor(12, initializer=_init) as ex:
        for i, _ in enumerate(ex.map(job, js, chunksize=16)):
            if i % 2000 == 0:
                print(i, flush=True)
    print("done", len(js), flush=True)
    return 0


def _init() -> None:
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path.insert(0, str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    raise SystemExit(main())
