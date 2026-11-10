"""仮定 A3 の録音の側(学習なし): 本物の男声の崩れ(BGM の 3 区間で E → C だけ 0.84〜0.96・写しの描画は 0.00〜0.27・P3 0.08〜0.78)を、Irodori の声の発話に録音の条件を足して前段の解析からやり直すと再現するか。
条件 = clean / pink20・pink10(1/f の雑音・発話の RMS に対する SN 比 dB)/ music10(和音と低音の倍音の合成の伴奏・SN 比 10 dB)/ rev(合成の残響 RT60 0.5 s・直接音との比 3 dB + pink30)/ gainN(音量 N dB)。
M16 の族の外(rev21): babble10(検証の別の Irodori の声 3 人の話し声の和・SN 比 10 dB)/ rev12(RT60 1.2 s・直接音との比 0 dB + pink30)/ hum15(50 Hz の倍音 40 本 + 同じパワーの 1/f の雑音・SN 比 15 dB)。
同じ条件を源と登録(E の参照・自己の構造)の両方に(雑音は別の実現)・目標の構造は元のまま。系 = snv_p1_eval.System(既定 snv_p3_x・snv_i8_m13)・自己の reamp と A0F の目標 2 声(k 0)。
CER = 乱した音の写しの描画の書き起こし 対 出力(試験の本物の音声と同じ)・copy_loss = 元の写しの書き起こし 対 乱した写しの書き起こし。発話 = 検証の Irodori の声の e1(男声 n・女声 n)。
mix=1 = 乱した条件ごとに 2 つの混ぜを足す: {cond}_y = 包絡 y だけ乱した音から(高さ・有声・f0 は元の音)/ {cond}_pv = 高さ・有声・f0 だけ乱した音から(y は元の音)・CER の基準 = 元の写し。
    cd training && uv run --no-sync python snv_diag/i9_rec_probe.py [snv_p3_x snv_i8_m13] [n=6] [conds=clean,pink20,pink10,music10,rev] [mix=0] [out=results/snv_i9/rec_probe.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p0_front as P
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
SR = P.SR


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    X /= np.sqrt(np.maximum(f, 20.0))
    x = np.fft.irfft(X, n)
    return x / (np.sqrt((x ** 2).mean()) + 1e-12)


def music(n: int, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(n)
    t0 = 0
    while t0 < n:
        L = int(rng.uniform(0.4, 1.2) * SR)
        root = 110.0 * 2 ** (rng.integers(0, 12) / 12)
        notes = [root * 2 ** (k / 12) for k in (0, int(rng.choice([3, 4])), 7)] + [root / 2]
        tt = np.arange(L) / SR
        env = np.minimum(1.0, tt / 0.02) * np.exp(-tt / rng.uniform(0.3, 1.5))
        seg = sum(sum(np.sin(2 * np.pi * f * h * tt + rng.uniform(0, 2 * np.pi)) / h for h in range(1, 11) if f * h < 8000) * (2 ** rng.integers(0, 2)) for f in notes)
        seg = seg * env
        e = min(n, t0 + L)
        out[t0:e] += seg[: e - t0]
        t0 += int(L * rng.uniform(0.6, 1.0))
    return out / (np.sqrt((out ** 2).mean()) + 1e-12)


def reverb(x: np.ndarray, rng: np.random.Generator, rt60: float = 0.5, drr_db: float = 3.0) -> np.ndarray:
    L = int(rt60 * 1.5 * SR)
    tt = np.arange(L) / SR
    tail = rng.standard_normal(L) * np.exp(-6.9 * tt / rt60)
    tail[: int(0.003 * SR)] = 0.0
    tail *= 10 ** (-drr_db / 20) / (np.sqrt((tail ** 2).sum()) + 1e-12)
    h = np.zeros(L)
    h[0] = 1.0
    y = np.convolve(x, h + tail)[: len(x)]
    return y * (np.sqrt((x ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-12))


def babble(n: int, rng: np.random.Generator, others: list) -> np.ndarray:
    out = np.zeros(n)
    for i in rng.choice(len(others), size=3, replace=False):
        o = others[i]
        o = np.tile(o, int(np.ceil(n / len(o))) + 1)
        st = int(rng.integers(0, len(o) - n))
        seg = o[st:st + n]
        out += seg / (np.sqrt((seg ** 2).mean()) + 1e-12)
    return out / (np.sqrt((out ** 2).mean()) + 1e-12)


def hum(n: int, rng: np.random.Generator) -> np.ndarray:
    tt = np.arange(n) / SR
    h = sum(np.sin(2 * np.pi * 50.0 * k * tt + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 41))
    h = h / (np.sqrt((h ** 2).mean()) + 1e-12) + pink(n, rng)
    return h / (np.sqrt((h ** 2).mean()) + 1e-12)


def perturb(x: np.ndarray, cond: str, rng: np.random.Generator, others: list | None = None) -> np.ndarray:
    if cond == "clean":
        return x
    rms = np.sqrt((x ** 2).mean())
    if cond.startswith("babble"):
        return x + babble(len(x), rng, others) * rms * 10 ** (-float(cond[6:]) / 20)
    if cond.startswith("hum"):
        return x + hum(len(x), rng) * rms * 10 ** (-float(cond[3:]) / 20)
    if cond == "rev12":
        y = reverb(x, rng, rt60=1.2, drr_db=0.0)
        return y + pink(len(y), rng) * rms * 10 ** (-30 / 20)
    if cond.startswith("pink"):
        return x + pink(len(x), rng) * rms * 10 ** (-float(cond[4:]) / 20)
    if cond.startswith("music"):
        return x + music(len(x), rng) * rms * 10 ** (-float(cond[5:]) / 20)
    if cond == "rev":
        y = reverb(x, rng)
        return y + pink(len(y), rng) * rms * 10 ** (-30 / 20)
    if cond.startswith("gain"):
        return x * 10 ** (float(cond[4:]) / 20)
    raise SystemExit(cond)


def utt_of(x: np.ndarray) -> dict:
    Tn = P.n_frames(len(x))
    f = P.analyze(x[: Tn * P.H].astype(np.float32))
    f0, env, per = f["f0"], f["env"], f["per"]
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": x[: Tn * P.H].astype(np.float32)}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_p3_x", "snv_i8_m13"]
    n = int(opt.get("n", 6))
    conds = opt.get("conds", "clean,pink20,pink10,music10,rev").split(",")
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    val = split["val"]
    voices = [v for v in val if v.startswith("em")][:n] + [v for v in val if v.startswith("ef")][:: max(1, len([v for v in val if v.startswith("ef")]) // n)][:n]
    load = lambda p: librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float64)
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr_(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y = torch.from_numpy(u["y"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        v = torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    systems = [EV.System(t, gen) for t in tags]
    keep: list = []
    rows = []
    bab = [v for v in val if v not in voices][:9]
    others = [load(sorted((ROOT / "data/snv_e1" / v).glob("t*.wav"))[5]) for v in bab] if any(c.startswith("babble") for c in conds) else []
    for vi, vv in enumerate(voices):
        wavs = sorted((ROOT / "data/snv_e1" / vv).glob("t*.wav"))
        src, refs = wavs[0], wavs[1:4]
        xs, xr = load(src), [load(p) for p in refs]
        ref_clean = None
        clean_u, clean_st = utt_of(xs), [utt_of(x) for x in xr]
        keep.append((clean_u, clean_st))
        cases = []
        for ci, cond in enumerate(conds):
            rng = np.random.default_rng(1000 * vi + ci)
            u = clean_u if cond == "clean" else utt_of(perturb(xs, cond, rng, others))
            st = clean_st if cond == "clean" else [utt_of(perturb(x, cond, rng, others)) for x in xr]
            cases.append((cond, u, st))
            if opt.get("mix") == "1" and cond != "clean":
                pv = lambda a, b: a | {k: b[k] for k in ("lf", "v", "f0")}
                cases.append((f"{cond}_y", pv(u, clean_u), [pv(a, b) for a, b in zip(st, clean_st)]))
                cases.append((f"{cond}_pv", pv(clean_u, u), [pv(b, a) for a, b in zip(st, clean_st)]))
        for cond, u, st in cases:
            keep.append((u, st))
            ref = tr_(copy(u)) if "_" not in cond else ref_clean
            if cond == "clean":
                ref_clean = ref
            for s in systems:
                for tgt in [None] + tgts:
                    S = s.structure(f"self|{cond}|{vv}", st) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                    out = tr_(s.convert(u, st, S, 0))
                    r = {"voice": vv, "sex": vv[:2], "cond": cond, "sys": s.tag, "tgt": tgt, "cer": cer(ref, out), "copy_loss": cer(ref_clean, ref) if ref_clean is not None else None,
                         "voiced_frac": float(u["v"].mean()), "len_ratio": round(len(out) / max(1, len(ref)), 3)}
                    rows.append(r)
                    print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "conds": conds, "voices": voices, "summary": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["summary"][cond] = {}
        for s in tags:
            for kind, sel in (("self", lambda r: r["tgt"] is None), ("F", lambda r: r["tgt"] is not None)):
                R = [r for r in rows if r["cond"] == cond and r["sys"] == s and sel(r)]
                res["summary"][cond][f"{s}|{kind}"] = {"cer": round(float(np.mean([r["cer"] for r in R])), 4), "cer_median": round(float(np.median([r["cer"] for r in R])), 4)}
        R = [r for r in rows if r["cond"] == cond and r["copy_loss"] is not None]
        res["summary"][cond]["copy_loss"] = round(float(np.mean([r["copy_loss"] for r in R])), 4)
        res["summary"][cond]["voiced_frac"] = round(float(np.mean([r["voiced_frac"] for r in rows if r["cond"] == cond])), 3)
    res["rows"] = rows
    print(json.dumps(res["summary"], indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i9/rec_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
