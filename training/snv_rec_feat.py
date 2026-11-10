"""paper_snv rev21 §5 M16 の入力: 学習のコーパス(e1・d・e1nv)の各発話を録音の条件で乱した音から、前段(snv_p0_front・先読み 0)の特徴を作る(当てる先はきれいな特徴のまま = 入力側だけ)。
版 k ごとに録音の場 = 声と k で決まる(同じ場の発話は同じ音量・同じ部屋・同じ背景の種類と大きさ・雑音の実現は発話ごと):
音量 U[−12, +4] dB / 残響 確率 0.5(RT60 U[0.15, 0.8] s・直接音との比 U[−3, 10] dB・インパルス応答は場ごと)/
背景 なし 0.2・色つき雑音 0.4(パワー ∝ f^−β・β U[0, 2]・SN 比 U[5, 30] dB)・和音と低音の伴奏 0.4(SN 比 U[5, 25] dB)/ 背景も残響も無い場は色つき雑音 U[20, 35] dB。
順 = 残響 → 背景(残響の後の発話の RMS に対する SN 比)→ 音量。合成の雑音と伴奏だけ(音声は足さない)。
出力 = data/snv_p_feat_rec/{組}/{声}/{名}_k{k}.npz(f0・ap・env・per = snv_p_feat と同じ)・場の値 = data/snv_p_feat_rec/rooms.json。
fam=real(paper_snv rev24 §5 M18・出力 data/snv_p_feat_rec2): 背景 = なし 0.1 / 本物の雑音 0.3(SN 比 U[0, 30] dB)/ 本物の音楽(歌なし)0.4(U[0, 20] dB)/ 合成の色つき雑音 0.1 / 合成の伴奏 0.1。
本物の音源 = data/snv_p_feat_rec2/sources.json(MUSAN の歌と人の声の無い・CC0 / CC BY / パブリックドメインの wav・16 kHz)。場ごとに音源のファイル・発話ごとに切り出しの位置。音量と残響は M16 と同じ。
fam=op(paper_snv rev31 §5 M22・出力 data/snv_p_feat_rec_op): 操作者の音(オーナー 2026-11-09「例えば手をたたきながら笑うなど」)。場ごとに 確率 0.8 で手や物の音(MUSAN の操作者の音の学習の 97 本・data/snv_sep/operator_sources.json と合成の手拍子の列 clap_00〜23 を半々)を
1 秒あたり U[0.3, 1.5] 個、長さ 0.08〜1 s(窓の中の最も大きい 10 ms を中心に)・発話の RMS に対して U[−15, +10] dB で、発話の中の一様な位置に重ねる。床 = 色つき雑音(β U[0, 2])を発話の RMS の U[−70, −35] dB・残響 確率 0.3・音量 U[−12, +4] dB。
    cd training && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync python snv_rec_feat.py [K=2] [sets=e1,d,e1nv] [workers=18] [fam=synth|real|op] [selftest]
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

import snv_p0_front as P
import snv_p_feat as PF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p_feat_rec"
SR = P.SR
FAM = "synth"
SOURCES: dict = {}


def seed_of(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def room(voice: str, k: int) -> dict:
    g = np.random.default_rng(seed_of(f"room|{voice}|{k}"))
    r = {"gain_db": float(g.uniform(-12, 4)), "rt60": float(g.uniform(0.15, 0.8)) if g.random() < 0.5 else 0.0, "drr_db": float(g.uniform(-3, 10))}
    u = g.random()
    r["bg"] = "none" if u < 0.2 else "noise" if u < 0.6 else "music"
    r["beta"] = float(g.uniform(0, 2))
    r["snr_db"] = float(g.uniform(5, 30)) if r["bg"] == "noise" else float(g.uniform(5, 25))
    if r["bg"] == "none" and r["rt60"] == 0.0:
        r["bg"], r["snr_db"] = "noise", float(g.uniform(20, 35))
    r["rir_seed"] = seed_of(f"rir|{voice}|{k}")
    return r


def room_real(voice: str, k: int) -> dict:
    g = np.random.default_rng(seed_of(f"room2|{voice}|{k}"))
    r = {"gain_db": float(g.uniform(-12, 4)), "rt60": float(g.uniform(0.15, 0.8)) if g.random() < 0.5 else 0.0, "drr_db": float(g.uniform(-3, 10))}
    u = g.random()
    r["bg"] = "none" if u < 0.1 else "real_noise" if u < 0.4 else "real_music" if u < 0.8 else "noise" if u < 0.9 else "music"
    r["beta"] = float(g.uniform(0, 2))
    r["snr_db"] = {"real_noise": float(g.uniform(0, 30)), "real_music": float(g.uniform(0, 20)), "noise": float(g.uniform(5, 30)), "music": float(g.uniform(5, 25)), "none": 0.0}[r["bg"]]
    if r["bg"] == "none" and r["rt60"] == 0.0:
        r["bg"], r["snr_db"] = "noise", float(g.uniform(20, 35))
    if r["bg"].startswith("real_"):
        pool = SOURCES[r["bg"][5:]]
        r["src"] = pool[int(g.integers(0, len(pool)))]
    r["rir_seed"] = seed_of(f"rir2|{voice}|{k}")
    return r


def room_op(voice: str, k: int) -> dict:
    g = np.random.default_rng(seed_of(f"room_op|{voice}|{k}"))
    return {"gain_db": float(g.uniform(-12, 4)), "rt60": float(g.uniform(0.15, 0.8)) if g.random() < 0.3 else 0.0, "drr_db": float(g.uniform(-3, 10)),
            "beta": float(g.uniform(0, 2)), "floor_db": float(g.uniform(-70, -35)), "ev": bool(g.random() < 0.8), "ev_rate": float(g.uniform(0.3, 1.5)),
            "p_clap": 0.5, "rir_seed": seed_of(f"rir_op|{voice}|{k}")}


def read_any(path: str, n: int, g: np.random.Generator) -> np.ndarray:
    import math
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(path)
    m = int(n * info.samplerate / SR) + 64
    if info.frames > m:
        st = int(g.integers(0, info.frames - m))
        x = sf.read(path, start=st, stop=st + m, dtype="float64")[0]
    else:
        x = np.resize(sf.read(path, dtype="float64")[0], m)
    if x.ndim > 1:
        x = x.mean(1)
    if info.samplerate != SR:
        q = math.gcd(SR, info.samplerate)
        x = resample_poly(x, SR // q, info.samplerate // q)
    return np.resize(x, n) if len(x) < n else x[:n]


def event(path: str, g: np.random.Generator, dur: float) -> np.ndarray:
    x = read_any(path, 3 * SR, g)
    fr = (x[: len(x) // 480 * 480].reshape(-1, 480) ** 2).mean(1)
    n = int(dur * SR)
    s = int(np.clip(int(fr.argmax()) * 480 + 240 - n // 3, 0, len(x) - n))
    e = x[s:s + n].copy()
    f = max(1, min(int(0.005 * SR), n // 4))
    e[:f] *= np.linspace(0.0, 1.0, f)
    e[n - f:] *= np.linspace(1.0, 0.0, f)
    return e / (np.sqrt((e ** 2).mean()) + 1e-12)


def perturb_op(x: np.ndarray, r: dict, g: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float64)
    if r["rt60"] > 0:
        h = rir(r)
        n = len(y)
        z = np.fft.irfft(np.fft.rfft(y, n + len(h)) * np.fft.rfft(h, n + len(h)), n + len(h))[:n]
        y = z * (np.sqrt((y ** 2).mean()) / (np.sqrt((z ** 2).mean()) + 1e-12))
    rms = np.sqrt((y ** 2).mean())
    y = y + colored(len(y), r["beta"], g) * rms * 10 ** (r["floor_db"] / 20)
    if r["ev"]:
        for _ in range(max(1, int(round(r["ev_rate"] * len(y) / SR)))):
            pool = SOURCES["clap"] if g.random() < r["p_clap"] else SOURCES["op"]
            dur = float(g.uniform(0.08, 1.0))
            e = event(pool[int(g.integers(len(pool)))], g, dur)
            if len(e) >= len(y):
                continue
            on = int(g.integers(0, len(y) - len(e)))
            y[on:on + len(e)] += e * rms * 10 ** (float(g.uniform(-15, 10)) / 20)
    y = y * 10 ** (r["gain_db"] / 20)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def real_bg(n: int, path: str, g: np.random.Generator) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(path)
    m = n // 3 + 64
    if info.frames > m:
        st = int(g.integers(0, info.frames - m))
        x = sf.read(path, start=st, stop=st + m, dtype="float64")[0]
    else:
        x = np.resize(sf.read(path, dtype="float64")[0], m)
    if x.ndim > 1:
        x = x.mean(1)
    y = resample_poly(x, 3, 1)[:n] if info.samplerate == 16000 else np.resize(x, n)
    return y / (np.sqrt((y ** 2).mean()) + 1e-12)


def colored(n: int, beta: float, g: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(g.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X *= np.maximum(f, 20.0) ** (-beta / 2)
    x = np.fft.irfft(X, n)
    return x / (np.sqrt((x ** 2).mean()) + 1e-12)


def music(n: int, g: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    t0 = 0
    while t0 < n:
        L = int(g.uniform(0.4, 1.2) * SR)
        root = 110.0 * 2 ** (g.integers(0, 12) / 12)
        notes = [root * 2 ** (k / 12) for k in (0, int(g.choice([3, 4])), 7)] + [root / 2]
        tt = np.arange(L) / SR
        env = np.minimum(1.0, tt / 0.02) * np.exp(-tt / g.uniform(0.3, 1.5))
        seg = sum(sum(np.sin(2 * np.pi * f * h * tt + g.uniform(0, 2 * np.pi)) / h for h in range(1, 11) if f * h < 8000) * g.uniform(0.5, 1.5) for f in notes) * env
        e = min(n, t0 + L)
        out[t0:e] += seg[: e - t0]
        t0 += int(L * g.uniform(0.6, 1.0))
    return out / (np.sqrt((out ** 2).mean()) + 1e-12)


def rir(r: dict) -> np.ndarray:
    g = np.random.default_rng(r["rir_seed"])
    L = int(r["rt60"] * 1.5 * SR)
    tt = np.arange(L) / SR
    tail = g.standard_normal(L) * np.exp(-6.9 * tt / r["rt60"])
    tail[: int(0.003 * SR)] = 0.0
    tail *= 10 ** (-r["drr_db"] / 20) / (np.sqrt((tail ** 2).sum()) + 1e-12)
    tail[0] += 1.0
    return tail


def perturb(x: np.ndarray, r: dict, g: np.random.Generator) -> np.ndarray:
    y = x.astype(np.float64)
    if r["rt60"] > 0:
        h = rir(r)
        n = len(y)
        z = np.fft.irfft(np.fft.rfft(y, n + len(h)) * np.fft.rfft(h, n + len(h)), n + len(h))[:n]
        y = z * (np.sqrt((y ** 2).mean()) / (np.sqrt((z ** 2).mean()) + 1e-12))
    rms = np.sqrt((y ** 2).mean())
    if r["bg"] == "noise":
        y = y + colored(len(y), r["beta"], g) * rms * 10 ** (-r["snr_db"] / 20)
    elif r["bg"] == "music":
        y = y + music(len(y), g) * rms * 10 ** (-r["snr_db"] / 20)
    elif r["bg"].startswith("real_"):
        y = y + real_bg(len(y), r["src"], g) * rms * 10 ** (-r["snr_db"] / 20)
    y = y * 10 ** (r["gain_db"] / 20)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def work(job: tuple) -> str:
    key, src, k = job
    torch.set_num_threads(1)
    dst = OUT / f"{key}_k{k}.npz"
    if dst.exists():
        return key
    x = PF.load48(src)
    T = P.n_frames(len(x))
    r = {"real": room_real, "op": room_op}.get(FAM, room)(key.split("/")[1], k)
    xp = (perturb_op if FAM == "op" else perturb)(x[: T * P.H], r, np.random.default_rng(seed_of(f"utt|{key}|{k}")))
    f = P.analyze(xp)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmp, **{kk: v.astype(np.float32) for kk, v in f.items()})
    os.replace(tmp, dst)
    return key


def selftest() -> None:
    x = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
    r0 = {"gain_db": 0.0, "rt60": 0.0, "drr_db": 0.0, "bg": "noise", "beta": 1.0, "snr_db": 200.0, "rir_seed": 0}
    assert np.allclose(perturb(x, r0, np.random.default_rng(0)), x, atol=1e-6), "恒等の乱しが恒等でない"
    r1 = r0 | {"snr_db": 10.0}
    d = perturb(x, r1, np.random.default_rng(0)) - x
    snr = 10 * np.log10((x.astype(np.float64) ** 2).mean() / (d ** 2).mean())
    assert abs(snr - 10.0) < 0.2, snr
    r2 = r0 | {"gain_db": -6.0}
    assert abs(20 * np.log10(np.sqrt((perturb(x, r2, np.random.default_rng(0)) ** 2).mean() / (x ** 2).mean())) + 6.0) < 0.05
    r3 = r0 | {"rt60": 0.5, "drr_db": 3.0}
    y3 = perturb(x, r3, np.random.default_rng(0))
    assert y3.shape == x.shape and abs(np.sqrt((y3 ** 2).mean()) - np.sqrt((x ** 2).mean())) < 1e-3
    import tempfile
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        src = str(Path(td) / "src16k.wav")
        sf.write(src, (0.3 * np.sin(2 * np.pi * 440.0 * np.arange(16000 * 3) / 16000)).astype(np.float32), 16000)
        r4 = r0 | {"bg": "real_music", "snr_db": 5.0, "src": src}
        d4 = perturb(x, r4, np.random.default_rng(0)) - x
        snr4 = 10 * np.log10((x.astype(np.float64) ** 2).mean() / (d4 ** 2).mean())
        assert abs(snr4 - 5.0) < 0.2, snr4
        sp4 = np.abs(np.fft.rfft(d4))
        assert abs(np.argmax(sp4) * SR / len(d4) - 440.0) < 2.0, "本物の音源の上げ直しで高さが変わる"
    if FAM == "op":
        rs = [room_op(f"e{i}", k) for i in range(400) for k in range(2)]
        y = perturb_op(x * 3, rs[0] | {"ev": True, "ev_rate": 1.0, "rt60": 0.0, "floor_db": -60.0, "gain_db": 0.0}, np.random.default_rng(0))
        assert y.shape == x.shape and np.isfinite(y).all()
        print("selftest ok op", {"ev": sum(r["ev"] for r in rs) / len(rs), "reverb": sum(r["rt60"] > 0 for r in rs) / len(rs)}, "clap", len(SOURCES["clap"]), "op", len(SOURCES["op"]), flush=True)
        return
    fam_room = room_real if FAM == "real" else room
    rs = [fam_room(f"e{i}", k) for i in range(400) for k in range(2)]
    bgs = ("none", "noise", "music", "real_noise", "real_music") if FAM == "real" else ("none", "noise", "music")
    print("selftest ok", FAM, {b: sum(r["bg"] == b for r in rs) / len(rs) for b in bgs}, "reverb", sum(r["rt60"] > 0 for r in rs) / len(rs), flush=True)


def init_fam(fam: str) -> None:
    global FAM, OUT, SOURCES
    FAM = fam
    if fam == "real":
        OUT = ROOT / "data/snv_p_feat_rec2"
        SOURCES = json.loads((OUT / "sources.json").read_text())["pools"]
    if fam == "op":
        OUT = ROOT / "data/snv_p_feat_rec_op"
        SOURCES = {"op": json.loads((ROOT / "data/snv_sep/operator_sources.json").read_text())["pools"]["train"],
                   "clap": sorted(str(q) for q in (ROOT / "data/snv_sep/wm").glob("clap_*.wav"))[:24]}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    init_fam(opt.get("fam", "synth"))
    selftest()
    if "selftest" in sys.argv[1:]:
        return 0
    K, workers = int(opt.get("K", 2)), int(opt.get("workers", 18))
    sets = opt.get("sets", "e1,d,e1nv").split(",")
    rooms: dict = {}
    for name in sets:
        js = [(key, src, k) for key, src in PF.jobs_for(name) for k in range(K)]
        for key, _, k in js:
            rooms.setdefault(key.split("/")[1], {})[k] = {"real": room_real, "op": room_op}.get(FAM, room)(key.split("/")[1], k)
        done = 0
        with ProcessPoolExecutor(workers, initializer=init_fam, initargs=(FAM,)) as ex:
            for _ in ex.map(work, js, chunksize=4):
                done += 1
                if done % 2000 == 0:
                    print(name, done, "/", len(js), flush=True)
        print(name, "done", len(js), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rooms.json").write_text(json.dumps(rooms, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
