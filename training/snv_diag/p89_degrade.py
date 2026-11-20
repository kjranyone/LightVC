"""段 P89 step a(results/snv_p89/prereg.yaml・学習なし・ASR なし): 族の男声の元の音に加工を 1 つずつ当て、包絡のフレームごとの揺れの比(c1〜c8 の隣のフレームの差 ÷ 8 フレームの動き)がどこまで上がるかを出す。
    cd training && SNV_DEV=cuda uv run --no-sync python snv_diag/p89_degrade.py [out=results/snv_p89/degrade.json]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as I13
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = EV.ROOT
SR = P.SR


def ratio_of(u: dict) -> float:
    y, v = u["y"], u["v"] > 0.5
    idx = np.flatnonzero(np.convolve(v.astype(int), np.ones(9, int), "valid") == 9)
    if len(idx) < 20:
        return float("nan")
    return float(np.abs(y[1:9, idx + 1] - y[1:9, idx]).mean() / (np.abs(y[1:9, idx + 8] - y[1:9, idx]).mean() / 8))


def ff(x: np.ndarray, sr: int, args: list[str], ext: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        a, b, c = f"{d}/a.wav", f"{d}/b.{ext}", f"{d}/c.wav"
        sf.write(a, x, sr)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, *args, b], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", b, "-ar", str(sr), "-ac", "1", c], check=True)
        y = sf.read(c)[0].astype(np.float64)
    n = min(len(x), len(y))
    return np.pad(y[:n], (0, len(x) - n))


def mp3(x: np.ndarray, kb: int) -> np.ndarray:
    return resample_poly(ff(resample_poly(x, 2, 3), 32000, ["-c:a", "libmp3lame", "-b:a", f"{kb}k"], "mp3"), 3, 2)[: len(x)]


def specsub(x: np.ndarray, over: float, floor: float) -> np.ndarray:
    S = librosa.stft(x, n_fft=960, hop_length=480)
    mag, ph = np.abs(S), np.angle(S)
    nz = np.percentile(mag, 10, axis=1, keepdims=True)
    return librosa.istft(np.maximum(mag - over * nz, floor * mag) * np.exp(1j * ph), hop_length=480, length=len(x))


def agc(x: np.ndarray) -> np.ndarray:
    e, out = 1e-3, np.empty_like(x)
    aa, ar = np.exp(-1 / (0.005 * SR)), np.exp(-1 / (0.05 * SR))
    env = np.empty_like(x)
    for i, s in enumerate(np.abs(x)):
        e = (aa if s > e else ar) * e + (1 - (aa if s > e else ar)) * s
        env[i] = e
    g = np.minimum(1.0, (env / 0.05 + 1e-9) ** (-0.75))
    return x * g / max(np.abs(x * g).max(), 1e-9) * np.abs(x).max()


def noise(x: np.ndarray, snr: float, seed: int = 0) -> np.ndarray:
    n = np.random.default_rng(seed).standard_normal(len(x))
    return x + n * np.sqrt((x ** 2).mean() / 10 ** (snr / 10))


FORMS = {
    "clean": lambda x: x,
    "mp3_32k": lambda x: mp3(x, 32), "mp3_48k": lambda x: mp3(x, 48), "mp3_64k": lambda x: mp3(x, 64),
    "opus12_mp3": lambda x: mp3(ff(x, SR, ["-c:a", "libopus", "-b:a", "12k"], "ogg"), 48), "opus24_mp3": lambda x: mp3(ff(x, SR, ["-c:a", "libopus", "-b:a", "24k"], "ogg"), 48),
    "aac32_mp3": lambda x: mp3(ff(x, SR, ["-c:a", "aac", "-b:a", "32k"], "m4a"), 48),
    "noise30": lambda x: noise(x, 30), "noise20": lambda x: noise(x, 20),
    "ns_floor05": lambda x: specsub(noise(x, 20), 2.0, 0.05), "ns_floor20": lambda x: specsub(noise(x, 20), 2.0, 0.2),
    "agc": agc, "clip": lambda x: np.tanh(3 * x / max(np.abs(x).max(), 1e-9)) * np.abs(x).max(),
    "q8": lambda x: np.round(x / max(np.abs(x).max(), 1e-9) * 127) / 127 * np.abs(x).max(), "rs16k": lambda x: resample_poly(resample_poly(x, 1, 3), 3, 1)[: len(x)],
}


def job(a: tuple) -> dict:
    torch.set_num_threads(1)
    src, path, wj = a
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float64)
    out = {"src": src}
    for nm, f in FORMS.items():
        try:
            out[nm] = ratio_of(I13.utt_of(np.asarray(f(x), np.float64)))
        except Exception as e:
            out[nm] = float("nan")
            out[nm + "_err"] = str(e)[:80]
    if wj is not None:
        out["irreg"] = ratio_of(I13.utt_of(np.asarray(wj, np.float64)))
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = ROOT / opt.get("out", "results/snv_p89/degrade.json")
    man = json.loads((ROOT / "data/snv_p73/sets.json").read_text())["d_clean"]
    items = [(w["wav"], str(ROOT / w["wav"])) for v in man for w in v["src"]]
    me = types.SimpleNamespace(gen=EV.load_gen())
    g = np.random.default_rng(0)
    jobs = []
    for src, path in items:
        u = I13.utt_of(librosa.load(path, sr=SR, mono=True)[0].astype(np.float64))
        f0 = u["f0"].copy()
        v = f0 > 0
        f0[v] = f0[v] * (1 + 0.03 * g.standard_normal(int(v.sum())))
        sub = v & (g.random(len(f0)) < 0.3)
        f0[sub] = f0[sub] / 2
        u2 = u | {"f0": f0.astype(np.float32), "lf": np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)}
        jobs.append((src, path, EV.System.render(me, u2, torch.from_numpy(u["y"])[None].to(EV.DEV), 0)))
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(job, jobs))
    keys = list(FORMS) + ["irreg"]
    res = {k: {"中央値": round(float(np.nanmedian([r[k] for r in rows])), 3), "上の 1/4": round(float(np.nanpercentile([r[k] for r in rows], 75)), 3), "欠け": int(np.isnan([r[k] for r in rows]).sum())} for k in keys}
    out.write_text(json.dumps({"prereg": "results/snv_p89/prereg.yaml(step_a)", "n": len(rows), "揺れの比": res, "rows": rows}, ensure_ascii=False, indent=1))
    for k, v in res.items():
        print(f"{k:12s}", v, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
