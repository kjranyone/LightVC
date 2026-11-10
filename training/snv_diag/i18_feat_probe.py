"""段 I18 の後の切り分け(学習なし・変換も書き起こしもしない): 重みの変種を M23 の掛け方で通した特徴(推論の E が読む値)を、きれいな音の前段 v3 の特徴と比べる。
声 = 検証の Irodori 12 声(男 6・女 6)の e1 の 3 本(t00・t04・t05)。条件 = i13_bgm_dev_probe と同じ混ぜ(本物の手の音 = FSD50K の評価の組): clap0・clap10・op5・claplead15。
変種 = none(重みなし = 前段 v3)/ f<床>(理想の比の重みを床で止める)/ bin(理想の 2 値)/ lk<k>(学んだ重みの k 乗・既定 snv_i18_mask@step10k)/ i17(snv_i17_mask@step10k の学んだ重み)。
量(声ごとの中央値の中央値): 包絡の L1(c0〜c24)を手の音のあるフレーム(2048 点の格子で手の音のエネルギーが声の −10 dB を越える)と全てのフレームで・周期性の L1(手の音のフレーム)・
声の無い時間を足す条件では声の無い時間の c0 の平均(声の時間の有声のフレームの c0 の平均との差 = 手の音が包絡に残る量)。
    cd training && uv run --no-sync python snv_diag/i18_feat_probe.py [lmask=snv_i18_mask@step10k] [ovar=none,f0,f0.1,f0.01,f0.001,bin,lk1,lk2,lk4,i17] [conds=clap0,clap10,op5,claplead15] [out=results/snv_i18/feat_probe.json]
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
import i13_bgm_dev_probe as B
import snv_p0_front as P
import train_snv_mask as MK

ROOT = B.ROOT
SR = P.SR


def frame_power(x: np.ndarray, Tn: int) -> np.ndarray:
    return MK.power(torch.from_numpy(x[: Tn * P.H].astype(np.float32))[None])[0].double().numpy()


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    conds = [B.parse(c) for c in opt.get("conds", "clap0,clap10,op5,claplead15").split(",")]
    ovar = opt.get("ovar", "none,f0,f0.1,f0.01,f0.001,bin,lk1,lk2,lk4,i17").split(",")
    lname, _, lck = opt.get("lmask", "snv_i18_mask@step10k").partition("@")
    lmask = MK.load(ROOT / "results" / lname / f"{lck or 'last'}.pt", B.DEV)
    i17 = MK.load(ROOT / "results/snv_i17_mask/step10k.pt", B.DEV) if "i17" in ovar else None
    evp = json.loads((ROOT / "data/snv_sep/fsd50k_eval_sources.json").read_text())["pools"]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    val = split["val"]
    voices = [v for v in val if v.startswith("em")][:6] + [v for v in val if v.startswith("ef")][:: max(1, len([v for v in val if v.startswith("ef")]) // 6)][:6]
    load = lambda p: librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float64)
    rows = []
    for vi, vv in enumerate(voices):
        wavs = sorted((ROOT / "data/snv_e1" / vv).glob("t*.wav"))
        for ui, w in enumerate([wavs[0], wavs[4], wavs[5]]):
            xs = load(w)
            for name, kind, lead, lvl in conds:
                c, z = B.with_events(xs, evp["clap" if kind.startswith("clap") else "op"], 1 + vi + 100 * ui, lead, lvl, 3 if lead else 4)
                Tn = P.n_frames(len(z))
                uc = B.utt_of(c)
                ps, pn = frame_power(c, Tn), frame_power(z - c, Tn)
                ev = pn.sum(1) > 0.1 * ps.sum(1)
                vc = uc["v"][:Tn] > 0.5
                pad = np.zeros(Tn, bool)
                if lead:
                    pad[: int(B.LEAD * SR) // P.H] = True
                    pad[(int(B.LEAD * SR) + len(xs)) // P.H:] = True
                var = {}
                for v in ovar:
                    if v == "none":
                        var[v] = B.utt_of(z)
                    elif v == "i17":
                        var[v] = B.mask_variants(c, z, ["lk1"], i17)[0][1]
                    else:
                        var[v] = B.mask_variants(c, z, [v], lmask)[0][1]
                yc = uc["y"][:, :Tn]
                c0v = float(yc[0, vc].mean()) if vc.any() else 0.0
                for v, u in var.items():
                    y = u["y"][:, :Tn]
                    d = np.abs(y - yc)
                    r = {"voice": vv, "utt": ui, "cond": name, "var": v, "ev_frac": round(float(ev.mean()), 3),
                         "env_ev": float(d[:25, ev].mean()) if ev.any() else None, "env_all": float(d[:25].mean()),
                         "per_ev": float(d[25:29, ev].mean()) if ev.any() else None,
                         "env_ev_voiced": float(d[:25, ev & vc].mean()) if (ev & vc).any() else None}
                    if lead:
                        r["pad_c0"] = float(y[0, pad].mean() - c0v)
                        r["pad_c0_clean"] = float(yc[0, pad].mean() - c0v)
                    rows.append(r)
            print(vv, ui, flush=True)
    res = {"voices": voices, "conds": [c for c, _, _, _ in conds], "vars": ovar, "summary": {}}
    for name, _, _, _ in conds:
        res["summary"][name] = {}
        for v in ovar:
            R = [r for r in rows if r["cond"] == name and r["var"] == v]
            s = {}
            for k in ("env_ev", "env_all", "per_ev", "env_ev_voiced", "pad_c0", "pad_c0_clean"):
                per_voice = [np.median([r[k] for r in R if r["voice"] == vv and r.get(k) is not None]) for vv in voices if any(r["voice"] == vv and r.get(k) is not None for r in R)]
                if per_voice:
                    s[k] = round(float(np.median(per_voice)), 4)
            res["summary"][name][v] = s
    res["rows"] = rows
    print(json.dumps(res["summary"], indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i18/feat_probe.json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
