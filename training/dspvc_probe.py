"""dspvc_p0: 学習なしの男→女変換(決定的な変換だけ)の数値ゲートと耳プローブ。事前登録 results/dspvc_p0/prereg.yaml。

    CUDA_VISIBLE_DEVICES= uv run python dspvc_probe.py   # results/dspvc_p0/metrics.json・results/earbattery/dspvc_p0/
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
from a2_dsp_vc import VC, cer, ecapa, norm_text, spk_lists
from render_d1_ab import norm_trial
from s04_artic_probe import band_noise
from train_dec2 import load48

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/dspvc_p0"
EAR = ROOT / "results/earbattery/dspvc_p0"
ST = 11.0
ALPHA = 1.139
D_MS = 10.0


def convert(x: np.ndarray, cond: str, seed: int) -> np.ndarray:
    if cond == "source":
        return x.copy()
    Dn = int(D_MS * D.SR / 1000)
    lar, a_sub, e = D.analyze(x, 24)
    f0p, _ = D.causal_yin(x, voi_max=0.45)
    e, _ = D.rrps(e, f0p, 2 ** (ST / 12), Dn, voiced=D.voiced_known(len(x), f0p, Dn, D.F0_HOP))
    if cond == "pitch":
        return D.synthesize(e, a_sub)
    lar2 = D.warp_lar(lar, ALPHA)
    if cond == "pos_ctrl":
        rng = np.random.default_rng(seed)
        lar2 = lar2 + 0.5 * lar2.std(0, keepdims=True) * band_noise(lar2.shape[0], lar2.shape[1], 4, 15, rng)
    return D.synthesize(e, D.coef_schedule(lar2, len(x)))


def main() -> int:
    import whisper
    from sklearn.linear_model import LogisticRegression
    from s0_artic import harvest_f0
    OUT.mkdir(parents=True, exist_ok=True)
    eval_m, clf_m, clf_f = spk_lists()
    emb = ecapa()
    X, y = [], []
    for lab, spks in ((0, clf_m), (1, clf_f)):
        for s in spks:
            for u in (30, 31, 32):
                p = VC / f"wav48/{s}/{s}_{u:03d}.wav"
                if p.exists():
                    X.append(emb(load48(str(p)).astype(np.float64)))
                    y.append(lab)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    asr = whisper.load_model("base", device="cpu")
    conds = ("source", "pitch", "pitch_vtl", "pos_ctrl")
    rows = {c: [] for c in conds}
    ear_src = []
    for si, s in enumerate(eval_m):
        for u in (40, 41):
            p = VC / f"wav48/{s}/{s}_{u:03d}.wav"
            t = VC / f"txt/{s}/{s}_{u:03d}.txt"
            if not (p.exists() and t.exists()):
                continue
            x = load48(str(p)).astype(np.float64)[: 8 * D.SR]
            ref = norm_text(t.read_text())
            f0x, _ = harvest_f0(x)
            e0 = emb(x)
            outs = {}
            for c in conds:
                yv = convert(x, c, seed=1000 * si + u)
                outs[c] = yv
                tmp = OUT / "tmp.wav"
                pk = float(np.abs(yv).max())
                soundfile.write(tmp, (yv * (0.95 / pk if pk > 0.95 else 1.0)).astype(np.float32), D.SR)
                hyp = norm_text(asr.transcribe(str(tmp), language="en", fp16=False)["text"])
                f0y, _ = harvest_f0(np.clip(yv, -1, 1))
                ev = emb(yv)
                rows[c].append({"utt": f"{s}_{u:03d}", "p_female": float(clf.predict_proba(ev[None])[0, 1]),
                                "secs_src": float(ev @ e0), "cer": cer(ref, hyp),
                                "df0_st": float(12 * np.log2(np.median(f0y[f0y > 0]) / np.median(f0x[f0x > 0])))
                                if (f0y > 0).sum() > 10 and (f0x > 0).sum() > 10 else None})
            if u == 40 and len(ear_src) < 2:
                ear_src.append((f"{s}_{u:03d}", x, outs))
            print(s, u, {c: {k: round(v, 3) for k, v in rows[c][-1].items() if isinstance(v, float)} for c in conds}, flush=True)
    (OUT / "tmp.wav").unlink(missing_ok=True)
    summ = {c: {k: round(float(np.median([r[k] for r in rs if r[k] is not None])), 3)
                for k in ("p_female", "secs_src", "cer", "df0_st")} for c, rs in rows.items()}
    dcer = float(np.median([b["cer"] - a["cer"] for a, b in zip(rows["source"], rows["pitch_vtl"])]))
    gate = summ["pitch_vtl"]["p_female"] >= 0.5 and dcer <= 0.10
    rep = {"prereg": "results/dspvc_p0/prereg.yaml", "st": ST, "alpha": ALPHA, "D_ms": D_MS, "eval_speakers": eval_m,
           "summary_median": summ, "cer_increase_pitch_vtl_median": round(dcer, 3),
           "numeric_gate": {"rule": "pitch_vtl の P(female) 中央 ≥ 0.5 かつ CER 増分中央 ≤ 0.10", "pass": gate}, "rows": rows}
    (OUT / "metrics.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("summary_median", "cer_increase_pitch_vtl_median", "numeric_gate")}, ensure_ascii=False, indent=1))
    if not gate:
        print("数値ゲート FAIL: 耳の試料は作らない", flush=True)
        return 0
    EAR.mkdir(parents=True, exist_ok=True)
    clips, owner = {}, {}
    for uid, x, outs in ear_src:
        normed = norm_trial({c: outs[c] for c in conds}, outs["source"])
        for c in conds:
            clips[f"{uid}|{c}"] = normed[c]
    pk = max(float(np.abs(v).max()) for v in clips.values())
    a = 0.95 / pk if pk > 0.95 else 1.0
    names = list(clips)
    random.Random("dspvc_p0_20260927").shuffle(names)
    key = {"map": {}, "prereg": "results/dspvc_p0/prereg.yaml"}
    for i, nm in enumerate(names):
        soundfile.write(EAR / f"X{i + 1}.wav", (clips[nm] * a).astype(np.float32), D.SR)
        key["map"][f"X{i + 1}"] = nm
    (EAR / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    print("ear clips ->", EAR, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
