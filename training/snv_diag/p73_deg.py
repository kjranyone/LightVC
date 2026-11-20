"""段 P73 step a 補足 6(results/snv_p73/prereg.yaml・学習なし): 族の声に録音の成分を当てた音(入力側だけ・EMA はきれいな音のもの)。
声 = E1 の検証の男声 6 + 学習の男声 12・発話 = t00〜t02(登録)・t05・t06(源)。条件: n40・n30・n20・n10 = ピンクの雑音(声の区間の強さに対する SNR)/ rv3・rv6 = 合成の残響(RT60 0.3 s・DRR 6 dB / 0.6 s・0 dB)/
mp48・mp24 = 32 kHz の MP3(48・24 kbps・ffmpeg の libmp3lame)/ mpn30 = n30 → mp48 / all = rv3 → n30 → mp48。出力 = data/snv_p73/deg/<条件>/<声>/<発話>.wav と data/snv_p73/sets.json の組 d_<条件>。
    cd training && uv run --no-sync python snv_diag/p73_deg.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_d5_chain as CH

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/snv_p73"
SP = "data/snv_e1_feat/sparc"
TS = ("t00", "t01", "t02", "t05", "t06")
CONDS = ("n40", "n30", "n20", "n10", "rv3", "rv6", "mp48", "mp24", "mpn30", "all")


def active_power(x: np.ndarray, sr: int) -> float:
    n = sr // 50
    p = (x[: len(x) // n * n].reshape(-1, n) ** 2).mean(1)
    return float(p[p > np.percentile(p, 95) * 10 ** -3.5].mean())


def noise(x: np.ndarray, sr: int, snr: float, g: np.random.Generator) -> np.ndarray:
    return (x + CH.colored(len(x), 1.0, g) * np.sqrt(active_power(x, sr) / 10 ** (snr / 10))).astype(np.float32)


def reverb(x: np.ndarray, rt60: float, drr: float, seed: int) -> np.ndarray:
    return fftconvolve(x, CH.rir({"rt60": rt60, "drr_db": drr, "rir_seed": seed}))[: len(x)].astype(np.float32)


def mp3(x: np.ndarray, sr: int, kbps: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        a, b, c = f"{td}/a.wav", f"{td}/b.mp3", f"{td}/c.wav"
        sf.write(a, x, sr)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a, "-ar", "32000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{kbps}k", b], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", b, "-ar", str(sr), c], check=True)
        y = sf.read(c, dtype="float32")[0]
    d = int(np.argmax(np.correlate(y[: sr * 2], x[: sr], "valid"))) if len(x) > sr * 2 else 0
    y = y[d:]
    return np.pad(y, (0, max(0, len(x) - len(y))))[: len(x)]


def main() -> int:
    assert CH.SR == 48000
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    voices = [v for v in split["val"] if v.startswith("em")] + [v for v in split["train"] if v.startswith("em")][:12]
    sets = json.loads((D / "sets.json").read_text())
    for c in CONDS:
        for v in voices:
            for t in TS:
                q = D / f"deg/{c}/{v}/{t}.wav"
                if q.exists():
                    continue
                q.parent.mkdir(parents=True, exist_ok=True)
                x, sr = sf.read(ROOT / f"data/snv_e1/{v}/{t}.wav", dtype="float32")
                assert sr == CH.SR
                g = np.random.default_rng(CH.seed_of(f"p73deg|{c}|{v}|{t}"))
                if c in ("rv3", "all"):
                    x = reverb(x, 0.3, 6.0, CH.seed_of(f"p73rir|{v}"))
                if c == "rv6":
                    x = reverb(x, 0.6, 0.0, CH.seed_of(f"p73rir|{v}"))
                if c[0] == "n" or c in ("mpn30", "all"):
                    x = noise(x, sr, float(c[1:]) if c[0] == "n" else 30.0, g)
                if c.startswith("mp") or c == "all":
                    x = mp3(x, sr, 24 if c == "mp24" else 48)
                sf.write(q, x, sr)
        w = lambda v, t: {"wav": f"data/snv_p73/deg/{c}/{v}/{t}.wav", "ema": f"{SP}/e1/{v}/{t}.npz"}
        sets[f"d_{c}"] = [{"voice": v, "src": [w(v, "t05"), w(v, "t06")], "enr": [w(v, t) for t in ("t00", "t01", "t02")]} for v in voices]
        print(c, flush=True)
    w = lambda v, t: {"wav": f"data/snv_e1/{v}/{t}.wav", "ema": f"{SP}/e1/{v}/{t}.npz"}
    sets["d_clean"] = [{"voice": v, "src": [w(v, "t05"), w(v, "t06")], "enr": [w(v, t) for t in ("t00", "t01", "t02")]} for v in voices]
    (D / "sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
