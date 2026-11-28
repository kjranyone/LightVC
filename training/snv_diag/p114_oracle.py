"""段 P114(学習なし・耳の材料): 高さの読みだけを替えて、製品の回路の出力を描く(半音 +12)。prod = 今の追跡・v4 = 段 P108 の試作・ora = オフラインの 2 つの推定器(pYIN と harvest)が合う所だけを有声にし、対数の高さを 5 点の中央値でならした値(先読みあり・診断だけ)。
源 = 本物の男声 4 人・合成の男声 2 声(通常の発話)・合成の男声の悲鳴 2 本・笑い・喘ぎ・泣き。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p114_oracle.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for k, v in {"SNV_V4_MODE": "multi", "SNV_V4_SPLIT": "400", "SNV_V4_UCOST": "0.5", "SNV_V4_UEXT": "0.35", "SNV_V4_UON": "0.2", "SNV_V4_EXCL": "1", "SNV_V4_BETA": "0.05"}.items():
    os.environ.setdefault(k, v)

import librosa
import numpy as np
import pyworld
import soundfile as sf
import torch
from scipy.signal import medfilt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import p107_guard as GD
import p114_hnr as H
import snv_f0v4 as V
import snv_out_guard as OG
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent


def oracle(x: np.ndarray, n: int, fmax: float) -> np.ndarray:
    r, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=EV.SR, target_sr=16000), fmin=60, fmax=fmax, sr=16000, frame_length=1024, hop_length=80, center=True)
    r = np.where(vf, np.nan_to_num(r), 0.0)
    h = pyworld.harvest(x.astype(np.float64), EV.SR, frame_period=5.0, f0_floor=60.0, f0_ceil=fmax)[0]
    m = min(n, len(r), len(h))
    r, h = r[:m], h[:m]
    ok = (r > 0) & (h > 0) & (np.abs(np.log2(np.maximum(r, 1) / np.maximum(h, 1))) < 0.2)
    lf = np.where(ok, np.log2(np.maximum(r, 1.0)), 0.0)
    sm = medfilt(np.where(ok, lf, np.nan_to_num(np.interp(np.arange(m), np.where(ok)[0], lf[ok]) if ok.any() else lf)), 5)
    out = np.zeros(n, np.float32)
    out[:m] = np.where(ok, 2.0 ** sm, 0.0)
    return out


def main() -> None:
    k = 12
    out = ROOT / "results/snv_p114/ear"
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    S = EV.Ens(E.SYS, EV.load_gen())
    tg = [PB.utt(rd(p)) for p in sorted((PB.IN / "tg").glob("*.f32"))]
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    sp3 = lambda v: [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)]
    src = {f"real{i + 1}": (rd(PB.IN / sp / "in.f32"), [rd(p) for p in sorted((PB.IN / sp / "op").glob("*.f32"))], 700.0) for i, sp in enumerate(PB.SPS)}
    for v in ("em594", "em595"):
        src[f"fam_{v}"] = (lw(ROOT / f"data/snv_e1/{v}/t05.wav"), sp3(v), 700.0)
    for v, c in (("em595", "scream"), ("em596", "scream"), ("em596", "laugh"), ("em595", "moan"), ("em597", "cry")):
        src[f"{c}_{v}"] = (lw(ROOT / f"data/snv_e1nv/{v}/{c}_0.wav"), sp3(v), 1600.0)
    res: dict = {}
    p0 = S.parts[0]
    for name, (x, ops, fmax) in src.items():
        u, enr = PB.utt(x), [PB.utt(o) for o in ops]
        lv = OG.speech_level([e["wav"] for e in enr])
        pu = p0.prep(u)
        _, yo = S.convert(u, enr, st, k, with_y=True)
        f0 = pu["f0"].astype(np.float32)
        n = len(f0)
        fv = V.f0_track(torch.from_numpy(u["wav"][: P.n_frames(len(u["wav"])) * P.H]))[0].numpy()
        fv = np.concatenate([fv, np.zeros(max(0, n - len(fv)), np.float32)])[:n]
        fo = oracle(u["wav"], n, fmax)
        r = {}
        for nm, f in (("prod", f0), ("v4", fv), ("ora", fo)):
            w = OG.apply(u["wav"], np.asarray(p0.render(GD.cond(pu, f), yo[:, :29], k, None), np.float32), lv, p_out)
            p = out / f"{name}_{nm}.wav"
            sf.write(p, w, EV.SR, subtype="PCM_16")
            h = H.measure(str(p))
            both = (f > 0) & (fo > 0)
            r[nm] = {"voiced": round(float((f > 0).mean()), 2), "hnr": h["hnr_med"], "gross_vs_ora": round(float((np.abs(np.log2(f[both] / fo[both])) > 0.2).mean()), 3) if both.any() else None, "voiced_where_ora_unvoiced": round(float(((f > 0) & (fo <= 0)).mean()), 3)}
        sf.write(out / f"{name}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    sf.write(out / "target_ref.wav", tg[0]["wav"][: EV.SR * 12], EV.SR, subtype="PCM_16")
    (ROOT / "results/snv_p114/oracle.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
