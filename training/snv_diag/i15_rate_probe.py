"""O12 の切り分け(paper_snv rev30 §9 O12・学習なし・開発の組): 試験の伴奏の無い区間は学習の声より速い(有声 1 秒あたりの c0 の山 外郎売 7.8〜8.9・namikawa_3〜5 10.6〜12.1 対 Irodori 6.9・JVNV 6.6・results/snv_i10/decomp_rate.json)。
声を跨ぐ変換の落ちが話す速さで増えるかを、JVNV の男声(M1・M2 の regular_01 = 12 本)を WSOLA で速めた音(r = 1.0 / 1.3 / 1.6・高さは保つ)で測る。登録(E の参照・自己の構造)も同じ r で速めた regular_02。
CER の参照 = 同じ r の音の写しの書き起こし(速さそのものの物差しの脆さは自己で測り、女声へ − 自己 の増え方を読む)。系 = snv_p1_eval.System・自己と A0F の目標 2 声(k 0)。
tref=1 = 目標の参照(A0F の 3 本)も同じ r で速める(源と参照の速さの食い違いを除いたとき、声を跨ぐ落ちが消えるか)。
    cd training && uv run --no-sync python snv_diag/i15_rate_probe.py snv_i10_c_pr [rates=1.0,1.3,1.6] [out=results/snv_i15/rate_probe.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as BP
import nvoc as N
import snv_p0_front as P
import snv_p1_eval as EV
import train_rvoc as TR

ROOT = EV.ROOT
DEV = EV.DEV
SR = P.SR
JV = ROOT / "data/jvnv/jvnv_v1"
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")


def wsola(x: np.ndarray, r: float, win: float = 0.04, tol: float = 0.01) -> np.ndarray:
    if abs(r - 1.0) < 1e-9:
        return x.copy()
    n, hs, d = int(win * SR), int(win * SR) // 2, int(tol * SR)
    w = np.hanning(n)
    ha = int(round(hs * r))
    m = (len(x) - n - d) // ha
    y = np.zeros(m * hs + n)
    norm = np.zeros_like(y)
    pos, prev = 0, None
    for k in range(m):
        c = k * ha
        if prev is not None:
            lo, hi = max(0, c - d), min(len(x) - n, c + d)
            seg = x[lo:hi + n]
            corr = np.correlate(seg, prev, "valid")
            c = lo + int(np.argmax(corr))
        fr = x[c:c + n]
        y[pos:pos + n] += fr * w
        norm[pos:pos + n] += w
        prev = x[c + hs:c + hs + n] if c + hs + n <= len(x) else fr
        pos += hs
    return (y / np.maximum(norm, 1e-6)).astype(np.float64)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_i10_c_pr"]
    rates = [float(r) for r in opt.get("rates", "1.0,1.3,1.6").split(",")]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y, lf, v = (torch.from_numpy(u[k])[None].to(DEV) for k in ("y", "lf", "v"))
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    systems = [EV.System(t, gen) for t in tags]
    load = lambda p: librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float64)
    a0w = {t: [load(EV.A0 / "wav" / f"{t}_{j}.wav") for j in (0, 1, 2)] for t in tgts}
    tref_of = {r: {t: ([BP.utt_of(wsola(x, r)) for x in a0w[t]] if opt.get("tref") == "1" else [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)]) for t in tgts} for r in rates}
    rows = []
    for spk in ("M1", "M2"):
        enr = [load(JV / spk / e / "regular" / f"{spk}_{e}_regular_02.wav") for e in EMOS]
        srcs = [load(JV / spk / e / "regular" / f"{spk}_{e}_regular_01.wav") for e in EMOS]
        for r in rates:
            st = [BP.utt_of(wsola(x, r)) for x in enr]
            for e, x in zip(EMOS, srcs):
                u = BP.utt_of(wsola(x, r))
                ref = tr(copy(u))
                for s in systems:
                    for tgt in [None] + tgts:
                        S = s.structure(f"self|{spk}|{r}", st) if tgt is None else s.structure(f"t|{tgt}|{r}|{opt.get('tref', '0')}", tref_of[r][tgt])
                        out = tr(s.convert(u, st, S, 0))
                        row = {"spk": spk, "emo": e, "rate": r, "sys": s.tag, "tgt": tgt, "cer": cer(ref, out), "ref": ref[:40], "out": out[:50]}
                        rows.append(row)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "rates": rates, "summary": {}}
    for t in tags:
        for r in rates:
            sf_ = [x["cer"] for x in rows if x["sys"] == t and x["rate"] == r and x["tgt"] is None]
            ff_ = [x["cer"] for x in rows if x["sys"] == t and x["rate"] == r and x["tgt"] is not None]
            res["summary"][f"{t}|{r}"] = {"self_median": round(float(np.median(sf_)), 4), "F_median": round(float(np.median(ff_)), 4), "self_mean": round(float(np.mean(sf_)), 4), "F_mean": round(float(np.mean(ff_)), 4)}
    res["rows"] = rows
    print(json.dumps(res["summary"], indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i15/rate_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
