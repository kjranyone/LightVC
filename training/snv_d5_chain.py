"""段 D5(paper_snv rev41 §5 M1 rev41 (b)・§6 段 D5 lever_b): 入力側の録音の鎖。きれいな Irodori の音に、息の雑音(高域・声の包絡に従う)→ 部屋(近い反射 + 合成の残響)→ 床(色つきの雑音)と電源の唸り → マイクの色(低域の切り・近接の持ち上げ・傾き・高域の切り)→ 機器の雑音 → 音量 を掛ける。
v2 = v1 に、近い反射(0.3〜6 ms・スペクトルの細かい山谷)と息の雑音(高域の周期性)を足した(v1 は生成の痕跡が動かす包絡の c13〜c24 と高域の周期性の大きさに届かなかった・results/snv_d5/sig_probe.json)。
当てる先はきれいな音のまま(入力側だけ)。音声・本物の部屋の応答は使わない。値の域は開発の組(cv の devA)の台帳の分布(results/snv_d5/ledger.json)から決めた。
場 = 声と版 k で決まる(同じ場の発話は同じ部屋・同じマイク・同じ床の大きさ)。雑音の実現は発話ごと。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_d5_chain.py [ver=v2] [out=data/snv_d5_f2<ver>] [utts=t03,t04,t05] [k=0] [selftest]
"""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 48000
VER = "v2"


def seed_of(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def room(voice: str, k: int) -> dict:
    g = np.random.default_rng(seed_of(f"d5chain|{VER}|{voice}|{k}"))
    lu = lambda lo, hi: float(np.exp(g.uniform(np.log(lo), np.log(hi))))
    r = {"rt60": float(g.uniform(0.1, 0.4)) if g.random() < 0.5 else 0.0, "drr_db": float(g.uniform(3, 18)), "rir_seed": seed_of(f"d5rir|{voice}|{k}"),
         "floor_db": float(g.uniform(-62, -18)) if g.random() < 0.9 else None, "beta": float(g.uniform(0.5, 2.5)),
         "hum_db": float(g.uniform(-55, -30)) if g.random() < 0.15 else None, "hum_hz": float(g.choice([50.0, 60.0])), "hum_h": [float(x) for x in g.uniform(0.1, 1.0, 3)],
         "hpf_hz": lu(20, 120) if g.random() < 0.7 else None, "shelf_db": float(g.uniform(0, 6)) if g.random() < 0.3 else 0.0, "shelf_hz": lu(100, 250),
         "tilt_db_oct": float(g.uniform(-1.5, 1.5)), "lpf_hz": lu(6000, 20000) if g.random() < 0.3 else None,
         "hiss_db": float(g.uniform(-70, -40)) if g.random() < 0.5 else None, "gain_db": float(g.uniform(-12, 4))}
    if VER != "v1":
        r["refl"] = [[lu(0.3, 6.0), float(g.uniform(-15, -4)) , float(g.choice([-1.0, 1.0]))] for _ in range(int(g.integers(1, 4)))] if g.random() < 0.7 else []
        r["asp_db"] = float(g.uniform(-30, -10)) if g.random() < 0.5 else None
        r["asp_hz"] = lu(1500, 4000)
    return r


def rir(r: dict) -> np.ndarray:
    g = np.random.default_rng(r["rir_seed"])
    L = int(r["rt60"] * 1.5 * SR)
    tt = np.arange(L) / SR
    tail = g.standard_normal(L) * np.exp(-6.9 * tt / r["rt60"])
    tail[: int(0.003 * SR)] = 0.0
    tail *= 10 ** (-r["drr_db"] / 20) / (np.sqrt((tail ** 2).sum()) + 1e-12)
    tail[0] += 1.0
    return tail


def early(r: dict) -> np.ndarray | None:
    if not r.get("refl"):
        return None
    h = np.zeros(int(0.0065 * SR) + 1)
    h[0] = 1.0
    for ms, db, sg in r["refl"]:
        h[int(round(ms * 1e-3 * SR))] += sg * 10 ** (db / 20)
    return h


def aspirate(y: np.ndarray, r: dict, g: np.random.Generator) -> np.ndarray:
    n = len(y)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    hp = (f / r["asp_hz"]) ** 2 / np.sqrt(1 + (f / r["asp_hz"]) ** 4)
    yh = np.fft.irfft(np.fft.rfft(y) * hp, n)
    lp = 1.0 / np.sqrt(1 + (f / 50.0) ** 4)
    e = np.sqrt(np.maximum(np.fft.irfft(np.fft.rfft(yh ** 2) * lp, n), 0.0))
    w = np.fft.irfft(np.fft.rfft(g.standard_normal(n)) * hp, n)
    w = w / (np.sqrt((w ** 2).mean()) + 1e-12) * e
    return y + w * 10 ** (r["asp_db"] / 20)


def colored(n: int, beta: float, g: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(g.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X *= np.maximum(f, 20.0) ** (-beta / 2)
    x = np.fft.irfft(X, n)
    return x / (np.sqrt((x ** 2).mean()) + 1e-12)


def color(y: np.ndarray, r: dict) -> np.ndarray:
    n = len(y)
    f = np.maximum(np.fft.rfftfreq(n, 1.0 / SR), 1.0)
    H = 10 ** (r["tilt_db_oct"] * np.log2(f / 1000.0) / 20)
    if r["hpf_hz"]:
        H = H * (f / r["hpf_hz"]) ** 2 / np.sqrt(1 + (f / r["hpf_hz"]) ** 4)
    if r["shelf_db"]:
        H = H * 10 ** (r["shelf_db"] / 20 / (1 + (f / r["shelf_hz"]) ** 2))
    if r["lpf_hz"]:
        H = H / np.sqrt(1 + (f / r["lpf_hz"]) ** 8)
    return np.fft.irfft(np.fft.rfft(y) * H, n)


def apply(x: np.ndarray, r: dict, g: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float64)
    n = len(y)
    rms = np.sqrt((y ** 2).mean()) + 1e-12
    if r.get("asp_db") is not None:
        y = aspirate(y, r, g)
    he = early(r)
    if he is not None:
        z = np.fft.irfft(np.fft.rfft(y, n + len(he)) * np.fft.rfft(he, n + len(he)), n + len(he))[:n]
        y = z * (rms / (np.sqrt((z ** 2).mean()) + 1e-12))
    if r["rt60"] > 0:
        h = rir(r)
        z = np.fft.irfft(np.fft.rfft(y, n + len(h)) * np.fft.rfft(h, n + len(h)), n + len(h))[:n]
        y = z * (rms / (np.sqrt((z ** 2).mean()) + 1e-12))
    if r["floor_db"] is not None:
        y = y + colored(n, r["beta"], g) * rms * 10 ** (r["floor_db"] / 20)
    if r["hum_db"] is not None:
        tt = np.arange(n) / SR
        ph = g.uniform(0, 2 * np.pi, 3)
        hm = sum(a * np.sin(2 * np.pi * r["hum_hz"] * (i + 1) * tt + p) for i, (a, p) in enumerate(zip(r["hum_h"], ph)))
        y = y + hm / (np.sqrt((hm ** 2).mean()) + 1e-12) * rms * 10 ** (r["hum_db"] / 20)
    y = color(y, r)
    if r["hiss_db"] is not None:
        y = y + g.standard_normal(n) * rms * 10 ** (r["hiss_db"] / 20)
    return np.clip(y * 10 ** (r["gain_db"] / 20), -1.0, 1.0).astype(np.float32)


def work(job: tuple) -> str:
    import librosa
    import soundfile as sf
    voice, t, k, out = job
    dst = Path(out) / voice / f"{t}_k{k}.wav"
    if not dst.exists():
        x = librosa.load(str(ROOT / f"data/snv_e1/{voice}/{t}.wav"), sr=SR, mono=True)[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, apply(x, room(voice, k), np.random.default_rng(seed_of(f"d5n|{VER}|{voice}|{t}|{k}"))), SR)
    return voice


def selftest() -> None:
    g = np.random.default_rng(0)
    x = (g.standard_normal(SR) * 0.05).astype(np.float32)
    r = room("x", 0) | {"rt60": 0.0, "floor_db": None, "hum_db": None, "hpf_hz": None, "shelf_db": 0.0, "tilt_db_oct": 0.0, "lpf_hz": None, "hiss_db": None, "gain_db": 0.0, "refl": [], "asp_db": None}
    assert np.allclose(apply(x, r, g), x, atol=1e-6), "恒等の場で音が変わる"
    y = apply(x, r | {"floor_db": -20.0}, g)
    d = 10 * np.log10(((y - x) ** 2).mean() / (x ** 2).mean())
    assert abs(d + 20) < 0.5, d
    print("selftest ok")


def main() -> int:
    if "selftest" in sys.argv[1:]:
        selftest()
        return 0
    global VER
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    VER = opt.get("ver", VER)
    out = ROOT / opt.get("out", f"data/snv_d5_f2{VER}")
    utts, k = opt.get("utts", "t03,t04,t05").split(","), int(opt.get("k", 0))
    voices = sorted(d.name for d in (ROOT / "data/snv_e1").glob("e[fm]*"))
    jobs = [(v, t, k, str(out)) for v in voices for t in utts]
    with ProcessPoolExecutor(11) as ex:
        for i, _ in enumerate(ex.map(work, jobs, chunksize=8)):
            if (i + 1) % 600 == 0:
                print(i + 1, "/", len(jobs), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rooms.json").write_text(json.dumps({"ver": VER, "k": k, "rooms": {v: room(v, k) for v in voices}}, indent=1))
    print("done", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
