"""段 I10 の後の切り分け(学習なし): 本物の録音の帯域の色(namikawa は JVNV より 4〜8 kHz +8 dB・8〜13 kHz +16〜19 dB・15 kHz で切れる)は、推論の符号器の出力に声の残りとして乗り、声を跨ぐ変換の内容を落とすか。
色 = 2 kHz から上の滑らかな持ち上げ(6 kHz +8 dB・9.5 kHz +16 dB・12 kHz +18 dB)+ 15 kHz から上を落とす(零位相・波形の上・診断だけ)。源と登録(参照)の両方に同じ色。
条件: jv_self / jv_F_k0 = JVNV の M1・M2(i10_jvnv_probe と同じ発話と登録)/ jveq_self / jveq_F_k0 = 色を付けたもの / ir_mF_k0 / ireq_mF_k0 = 検証の Irodori の男声(学習に無い文・登録 = 同じ声の E1 の発話 3 本)→ A0F の目標 2 声。
CER = large-v3 の仮名で、写しの描画の書き起こし 対 出力(色を付けた写しを基準に)。
    cd training && uv run --no-sync python snv_diag/i10_eq_probe.py [snv_i8_m13 snv_i9_ctl snv_i9_aug snv_i10_combo] [n_ir=8] [out=results/snv_i10/eq_probe.json]
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
EQ_F = np.array([0.0, 2000.0, 6000.0, 9500.0, 12000.0, 14500.0, 15500.0, 24000.0])
EQ_DB = np.array([0.0, 0.0, 8.0, 16.0, 18.0, 18.0, -60.0, -60.0])


def color(x: np.ndarray) -> np.ndarray:
    n = 1 << int(np.ceil(np.log2(len(x) + 1)))
    X = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1.0 / P.SR)
    g = 10 ** (np.interp(f, EQ_F, EQ_DB) / 20)
    y = np.fft.irfft(X * g, n)[: len(x)]
    return (y * (np.sqrt(np.mean(x ** 2)) / (np.sqrt(np.mean(y ** 2)) + 1e-12))).astype(np.float32)


def feats(x: np.ndarray, dst: Path) -> dict:
    if not dst.exists():
        n = P.n_frames(len(x))
        f = P.analyze(x[: n * P.H])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    return T.load_utt(dst)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_i8_m13", "snv_i9_ctl", "snv_i9_aug", "snv_i10_combo"]
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
    tgts = meta["targets"][:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    jv_wav = lambda nm: librosa.load(str(JV / nm.split("_")[0] / nm.split("_")[1] / "regular" / f"{nm}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
    ir_wav = lambda rel: librosa.load(str(ROOT / "data/snv_e1" / f"{rel}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
    trials: list = []
    for spk in ("M1", "M2"):
        en = {c: [feats(color(jv_wav(f"{spk}_{e}_regular_02")) if c == "eq" else jv_wav(f"{spk}_{e}_regular_02"), T.FEAT / ("jvnv_eq" if c == "eq" else "jvnv") / f"{spk}_{e}_regular_02.npz") for e in EMOS] for c in ("", "eq")}
        for e in EMOS:
            nm = f"{spk}_{e}_regular_01"
            for c in ("", "eq"):
                u = feats(color(jv_wav(nm)) if c == "eq" else jv_wav(nm), T.FEAT / ("jvnv_eq" if c == "eq" else "jvnv") / f"{nm}.npz")
                trials.append((f"jv{c}", nm, u, en[c]))
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    males = [v for v in split["val"] if v.startswith("em")]
    for v in males[: int(opt.get("n_ir", 8))]:
        fs = sorted((ROOT / "data/snv_e1" / v).glob("t*.wav"))
        for c in ("", "eq"):
            sub = T.FEAT / ("e1" if c == "" else "e1_eq") / v
            en = [feats(color(ir_wav(f"{v}/{p.stem}")) if c == "eq" else ir_wav(f"{v}/{p.stem}"), sub / f"{p.stem}.npz") for p in fs[:3]]
            p = fs[-1]
            u = feats(color(ir_wav(f"{v}/{p.stem}")) if c == "eq" else ir_wav(f"{v}/{p.stem}"), sub / f"{p.stem}.npz")
            trials.append((f"ir{c}", f"{v}/{p.stem}", u, en))
    systems = [EV.System(t, gen) for t in tags]
    rows: list = []
    for grp, who, u, stat in trials:
        ref = tr(copy(u))
        plan = [(f"{grp}_self", [None]), (f"{grp}_F_k0", tgts)] if grp.startswith("jv") else [(f"{grp}_mF_k0", tgts)]
        for cond, tg in plan:
            for s in systems:
                for tgt in tg:
                    S = s.structure(f"self|{grp}|{who}", stat) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                    out = tr(s.convert(u, stat, S, 0))
                    r = {"cond": cond, "sys": s.tag, "src": who, "tgt": tgt, "cer": cer(ref, out), "ref": ref[:60], "out": out[:60]}
                    rows.append(r)
                    print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "eq": {"f": EQ_F.tolist(), "db": EQ_DB.tolist()}, "conditions": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["conditions"][cond] = {}
        for t in tags:
            R = [r for r in rows if r["cond"] == cond and r["sys"] == t]
            res["conditions"][cond][t] = {"cer": round(float(np.mean([min(r["cer"], 1.0) for r in R])), 4), "cer_median": round(float(np.median([r["cer"] for r in R])), 4), "trials": len(R)}
    res["rows"] = rows
    print(json.dumps(res["conditions"], ensure_ascii=False, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i10/eq_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
