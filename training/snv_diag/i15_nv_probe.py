"""段 I15(paper_snv rev30・オーナー「例えば手をたたきながら笑うなど」)の音素の無い発声のノイズ耐性(学習なし・記録だけ)。
検証の Irodori 12 声(男 6・女 6)の音素の無い発声(data/snv_e1nv の laugh_0・scream_0・mixed_0)に、本物の手拍子と手の音(FSD50K の評価の組・i13_bgm_dev_probe の with_events と同じ)を声と時間を重ねて足し、
同じ系のきれいな入力の出力からのずれを測る: d_mel = 対数 mel(80 帯・60 Hz〜16 kHz)の差の平均(dB・きれいな出力のフレームの大きさが最大の −40 dB 以上のフレーム)・dv = 出力を前段 v3 で解析した有声の率の差。
登録(E の参照・自己の構造)= その声の e1 の t01〜t03 に同じ条件を足したもの。変換 = 自己と A0F の目標 2 声(k 0)。
    cd training && uv run --no-sync python snv_diag/i15_nv_probe.py snv_i10_c_pr snv_i10_c_pr+sep:snv_i15_sep [n=6] [conds=clap0,clap10,op5] [out=results/snv_i15/nv_probe.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as BP
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = EV.ROOT
SR = P.SR
ITEMS = ("laugh_0", "scream_0", "mixed_0")


def logmel(w: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=w.astype(np.float32), sr=SR, n_fft=2048, hop_length=480, n_mels=80, fmin=60, fmax=16000, center=False)
    return 10 * np.log10(S + 1e-10)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    n = int(opt.get("n", 6))
    conds = [BP.parse(c) for c in opt.get("conds", "clap0,clap10,op5").split(",")]
    evp = json.loads((ROOT / "data/snv_sep/fsd50k_eval_sources.json").read_text())["pools"]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    fem = [v for v in split["val"] if v.startswith("ef")]
    voices = [v for v in split["val"] if v.startswith("em")][:n] + fem[:: max(1, len(fem) // n)][:n]
    load = lambda p: librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float64)
    gen = EV.load_gen()
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    systems = [EV.System(t, gen) for t in tags]
    rows = []
    for vi, vv in enumerate(voices):
        xr = [load(ROOT / "data/snv_e1" / vv / f"t0{j}.wav") for j in (1, 2, 3)]
        clean_st = [BP.utt_of(x) for x in xr]
        for it in ITEMS:
            p = ROOT / "data/snv_e1nv" / vv / f"{it}.wav"
            if not p.exists():
                continue
            xs = load(p)
            cu = BP.utt_of(xs)
            for name, kind, lead, lvl in conds:
                srcs = evp["clap" if kind.startswith("clap") else "op"]
                _, z = BP.with_events(xs, srcs, 101 + vi, False, lvl)
                un = BP.utt_of(z)
                st_n = [BP.utt_of(BP.with_events(x, srcs, 7 + j, False, lvl)[1]) for j, x in enumerate(xr)]
                for s in systems:
                    for tgt in [None] + tgts:
                        Sc = s.structure(f"self|clean|{vv}", clean_st) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                        Sn = s.structure(f"self|{name}|{vv}", st_n) if tgt is None else Sc
                        wc = np.asarray(s.convert(cu, clean_st, Sc, 0), np.float64)
                        wn = np.asarray(s.convert(un, st_n, Sn, 0), np.float64)
                        L = min(len(wc), len(wn))
                        mc, mn = logmel(wc[:L]), logmel(wn[:L])
                        e = mc.max(0)
                        act = e > e.max() - 40
                        Tn = P.n_frames(L)
                        vc = P.analyze(wc[: Tn * P.H].astype(np.float32))["f0"] > 0
                        vn = P.analyze(wn[: Tn * P.H].astype(np.float32))["f0"] > 0
                        r = {"voice": vv, "item": it, "cond": name, "sys": s.tag, "tgt": tgt, "d_mel": round(float(np.abs(mc[:, act] - mn[:, act]).mean()), 3),
                             "dv": round(float(abs(vc.mean() - vn.mean())), 3), "v_clean": round(float(vc.mean()), 3)}
                        rows.append(r)
                        print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "voices": voices, "items": ITEMS, "conds": [c for c, _, _, _ in conds], "summary": {}}
    for name, _, _, _ in conds:
        for t in tags:
            for kind, sel in (("self", lambda r: r["tgt"] is None), ("F", lambda r: r["tgt"] is not None)):
                R = [r for r in rows if r["cond"] == name and r["sys"] == t and sel(r)]
                if R:
                    res["summary"].setdefault(name, {})[f"{t}|{kind}"] = {"d_mel_median": round(float(np.median([r["d_mel"] for r in R])), 3), "dv_median": round(float(np.median([r["dv"] for r in R])), 3), "n": len(R)}
    res["rows"] = rows
    print(json.dumps(res["summary"], indent=1, ensure_ascii=False), flush=True)
    (ROOT / opt.get("out", "results/snv_i15/nv_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
