"""段 I10 の後の切り分け(学習なし): 本物の男声の自己の reamp の崩れは、源の録音の条件そのもの(床の雑音・伴奏)か、登録と源の録音の条件の食い違いか。
本物の登録(先頭 30 s)は無声のフレームが無音(c0 = 床)・試験の区間は床の雑音(外郎売)や伴奏(namikawa_0〜2)を持つ。学習(M16)は源と参照がいつも同じ場。
条件: real_self / real_F_k0 = 登録は元のまま(i8_text_probe と同じ)/ real_self_loo / real_F_k0_loo = 登録 = 同じ話者の別の試験の区間 5 本(同じ録音の条件)/
jvn_src_self = JVNV の源にだけ外郎売の床の雑音(外郎売の区間の休止から切り出し・外郎売と同じ SN 比)/ jvn_both_self = 源と登録の両方に / jv_self = 元の JVNV。
CER = large-v3 の仮名で、写しの描画の書き起こし 対 出力(i8_text_probe と同じ)。
    cd training && uv run --no-sync python snv_diag/i10_enroll_probe.py [snv_p3_x snv_i8_m13 snv_i10_e_rec snv_i10_combo] [out=results/snv_i10/enroll_probe.json]
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

ROOT = Path(__file__).resolve().parents[2]
JV = ROOT / "data/jvnv/jvnv_v1"
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
DEV = EV.DEV


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-20))


def floor_noise() -> tuple[np.ndarray, float]:
    segs, sp = [], []
    xf = int(0.005 * P.SR)
    for i in range(6):
        x = librosa.load(str(T.FEAT / f"real/uirou_{i}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
        u = T.load_utt(T.FEAT / f"real/uirou_{i}.npz")
        c0, v = u["y"][0], u["v"]
        q = np.quantile(c0[v < 0.5], 0.3) if (v < 0.5).any() else -np.inf
        quiet = (v < 0.5) & (c0 <= q)
        sp.append(np.concatenate([x[t * P.H:(t + 1) * P.H] for t in np.flatnonzero(v > 0.5)]))
        t = 0
        while t < len(quiet):
            if quiet[t]:
                s = t
                while t < len(quiet) and quiet[t]:
                    t += 1
                if t - s >= 10:
                    segs.append(x[(s + 1) * P.H:(t - 1) * P.H])
            t += 1
    out = segs[0]
    ramp = np.linspace(0, 1, xf, dtype=np.float32)
    for s in segs[1:]:
        if len(s) <= 2 * xf:
            continue
        out = np.concatenate([out[:-xf], out[-xf:] * (1 - ramp) + s[:xf] * ramp, s[xf:]])
    snr = 20 * np.log10(rms(np.concatenate(sp)) / rms(out))
    return out, float(snr)


def jv_wave(name: str) -> np.ndarray:
    spk, emo = name.split("_")[:2]
    return librosa.load(str(JV / spk / emo / "regular" / f"{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)


def feats_of(x: np.ndarray, dst: Path) -> dict:
    if not dst.exists():
        n = P.n_frames(len(x))
        f = P.analyze(x[: n * P.H])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    return T.load_utt(dst)


def with_floor(x: np.ndarray, v: np.ndarray, noise: np.ndarray, snr: float, off: int) -> np.ndarray:
    n = np.resize(np.roll(noise, -off), len(x))
    sp = rms(np.concatenate([x[t * P.H:(t + 1) * P.H] for t in np.flatnonzero(v > 0.5)]))
    return (x + n * (sp / rms(noise) / 10 ** (snr / 20))).astype(np.float32)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_p3_x", "snv_i8_m13", "snv_i10_e_rec", "snv_i10_combo"]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y, lf, v = (torch.from_numpy(u[k])[None].to(DEV) for k in ("y", "lf", "v"))
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"][:2], meta["sources"]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    real = {s: EV.utt(f"real/{s}") for s in srcs}
    enroll = {sp: [EV.utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    loo = {s: [real[o] for o in srcs if o != s and o.rsplit("_", 1)[0] == s.rsplit("_", 1)[0]] for s in srcs}
    noise, snr = floor_noise()
    print("uirou floor", len(noise) / P.SR, "s", "snr", round(snr, 1), "dB", flush=True)
    sub = T.FEAT / "jvnv_uiroufloor"
    trials: list = []
    for s in srcs:
        trials.append(("real_self", s, real[s], enroll[s.rsplit("_", 1)[0]], [None], None))
        trials.append(("real_self_loo", s, real[s], loo[s], [None], None))
        trials.append(("real_F_k0", s, real[s], enroll[s.rsplit("_", 1)[0]], tgts, None))
        trials.append(("real_F_k0_loo", s, real[s], loo[s], tgts, None))
    for spk in ("M1", "M2"):
        en_c = [T.load_utt(T.FEAT / f"jvnv/{spk}_{e}_regular_02.npz") for e in EMOS]
        en_n = [feats_of(with_floor(jv_wave(f"{spk}_{e}_regular_02"), c["v"], noise, snr, 7919 * j), sub / f"{spk}_{e}_regular_02.npz") for j, (e, c) in enumerate(zip(EMOS, en_c))]
        for j, e in enumerate(EMOS):
            nm = f"{spk}_{e}_regular_01"
            u_c = T.load_utt(T.FEAT / f"jvnv/{nm}.npz")
            u_n = feats_of(with_floor(jv_wave(nm), u_c["v"], noise, snr, 104729 * (j + 1)), sub / f"{nm}.npz")
            trials.append(("jv_self", nm, u_c, en_c, [None], u_c))
            trials.append(("jvn_src_self", nm, u_n, en_c, [None], u_c))
            trials.append(("jvn_both_self", nm, u_n, en_n, [None], u_c))
    systems = [EV.System(t, gen) for t in tags]
    rows: list = []
    ref_cache: dict = {}
    for cond, who, u, stat, tg, clean in trials:
        key = (who, id(clean) if clean is not None else id(u))
        if key not in ref_cache:
            ref_cache[key] = tr(copy(clean if clean is not None else u))
        ref = ref_cache[key]
        for s in systems:
            for tgt in tg:
                S = s.structure(f"self|{cond}|{who}", stat) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                out = tr(s.convert(u, stat, S, 0))
                r = {"cond": cond, "sys": s.tag, "src": who, "tgt": tgt, "cer": cer(ref, out), "voiced": round(float(u["v"].mean()), 3), "ref": ref[:60], "out": out[:60]}
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "floor_snr_db": round(snr, 2), "floor_sec": round(len(noise) / P.SR, 2), "conditions": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["conditions"][cond] = {}
        for t in tags:
            R = [r for r in rows if r["cond"] == cond and r["sys"] == t]
            res["conditions"][cond][t] = {"cer": round(float(np.mean([r["cer"] for r in R])), 4), "cer_median": round(float(np.median([r["cer"] for r in R])), 4), "trials": len(R)}
    res["rows"] = rows
    print(json.dumps(res["conditions"], ensure_ascii=False, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i10/enroll_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
