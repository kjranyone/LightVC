"""N2c(学習なし・物差しの較正): 正解との質感の統計の差 TEX。自己の reamp(正解の音が 1 対 1 で在る条件)で、出力が正解の質感からどれだけ離れたかを読む。
事前登録 = results/snv_n2c/prereg.yaml。素材は N2 のもの。tex_stats / tex_dist は A4 の評価から呼ぶ。出力 = results/snv_n2c/n2c.json・tex_scale.npy。
    cd training && uv run --no-sync python snv_n2c_tex.py <n2 の素材の dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

from snv_n2_judge import FAM, boot

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_n2c"
SR = 16000
HOP = 160
N_BAND = 32
MOD = ((4, 8), (8, 16), (16, 32), (32, 50))
FLAT = ((100, 1000), (1000, 3000), (3000, 8000))
GROUPS = {"mean": slice(0, 32), "std": slice(32, 64), "dstd": slice(64, 96), "mod": slice(96, 224), "flat": slice(224, 227)}
GATE = {"real_F": ["B2", "B3", "N20", "BZ", "LP4"], "iro_F": ["B2", "B3", "N20", "BZ"], "iro_M": ["Sg", "Sx"]}
LINE, LO = 0.90, 0.85
_FB = librosa.filters.mel(sr=SR, n_fft=512, n_mels=N_BAND, fmin=50, fmax=8000)
_F = librosa.fft_frequencies(sr=SR, n_fft=512)


def tex_stats(y16: np.ndarray) -> np.ndarray:
    P = np.abs(librosa.stft(y16.astype(np.float32), n_fft=512, hop_length=HOP)) ** 2
    M = _FB @ P
    e = M.sum(0)
    act = e > e.max() * 1e-4
    if act.sum() < 3:
        act[:] = True
    L = np.log(M + 1e-10)
    La = L[:, act]
    d = np.diff(La, axis=1) if La.shape[1] > 1 else np.zeros((N_BAND, 1))
    i0, i1 = np.where(act)[0][[0, -1]]
    env = M[:, i0:i1 + 1] ** 0.3
    env = env - env.mean(1, keepdims=True)
    n = max(env.shape[1], 8)
    ps = np.abs(np.fft.rfft(env, n=n, axis=1)) ** 2
    fm = np.fft.rfftfreq(n, d=HOP / SR)
    tot = ps[:, (fm >= 1) & (fm < 50)].sum(1) + 1e-12
    mod = np.stack([ps[:, (fm >= a) & (fm < b)].sum(1) / tot for a, b in MOD], 1).reshape(-1)
    Pa = P[:, act] + 1e-12
    flat = [float(np.mean(np.exp(np.mean(np.log(Pa[(_F >= a) & (_F < b)]), 0)) / np.mean(Pa[(_F >= a) & (_F < b)], 0))) for a, b in FLAT]
    return np.concatenate([La.mean(1), La.std(1), d.std(1), mod, np.array(flat)])


def tex_dist(a: np.ndarray, b: np.ndarray, scale: np.ndarray, part: slice = slice(None)) -> float:
    return float(np.mean(np.abs(a - b)[part] / scale[part]))


def mel_l1(y: np.ndarray, r: np.ndarray) -> float:
    n = min(len(y), len(r))
    A = np.log(_FB @ np.abs(librosa.stft(y[:n].astype(np.float32), n_fft=512, hop_length=HOP)) ** 2 + 1e-10)
    B = np.log(_FB @ np.abs(librosa.stft(r[:n].astype(np.float32), n_fft=512, hop_length=HOP)) ** 2 + 1e-10)
    return float(np.mean(np.abs(A - B)))


def main() -> int:
    src = Path(sys.argv[1])
    its = json.loads((ROOT / "results/snv_n2/items.json").read_text())
    st, wav = {}, {}
    for f in sorted(src.glob("*_*.npz")):
        if f.stem == "wavlm":
            continue
        d = np.load(f)
        for k in d.files:
            st[k] = tex_stats(d[k])
            wav[k] = d[k]
        print(f.stem, len(d.files), flush=True)
    scale = np.stack([st[f"clean|{it['name']}"] for it in its if it["split"] == "dev"]).std(0) + 1e-6
    np.save(OUT / "tex_scale.npy", scale)
    res: dict = {"prereg": "results/snv_n2c/prereg.yaml", "dims": int(len(scale)), "cells": {}, "l1_cells": {}, "group_cells": {}}
    for pop in ("real_F", "real_M", "iro_F", "iro_M"):
        for sp in ("dev", "test"):
            sel = [it for it in its if it["pop"] == pop and it["split"] == sp]
            grp = np.array([it["spk"] if pop.startswith("iro") else it["name"] for it in sel])
            for fam, (base, deg) in FAM.items():
                if fam in ("Sg", "Sx") and pop != "iro_M":
                    continue
                if fam == "V":
                    continue
                base = "Vf" if fam in ("Sg", "Sx") else "V"
                c = [st[f"clean|{it['name']}"] for it in sel]
                db = np.array([tex_dist(st[f"{deg}|{it['name']}"], ci, scale) for it, ci in zip(sel, c)])
                dv = np.array([tex_dist(st[f"{base}|{it['name']}"], ci, scale) for it, ci in zip(sel, c)])
                w = (db > dv) + 0.5 * (db == dv)
                res["cells"][f"{pop}|{sp}|{fam}"] = {"win": round(float(w.mean()), 3), "ci95": boot(w, grp), "n": len(sel),
                                                     "tex_deg": round(float(db.mean()), 3), "tex_base": round(float(dv.mean()), 3)}
                if sp == "test":
                    lb = np.array([mel_l1(wav[f"{deg}|{it['name']}"], wav[f"clean|{it['name']}"]) for it in sel])
                    lv = np.array([mel_l1(wav[f"{base}|{it['name']}"], wav[f"clean|{it['name']}"]) for it in sel])
                    res["l1_cells"][f"{pop}|{fam}"] = round(float(((lb > lv) + 0.5 * (lb == lv)).mean()), 3)
                    for g, part in GROUPS.items():
                        gb = np.array([tex_dist(st[f"{deg}|{it['name']}"], ci, scale, part) for it, ci in zip(sel, c)])
                        gv = np.array([tex_dist(st[f"{base}|{it['name']}"], ci, scale, part) for it, ci in zip(sel, c)])
                        res["group_cells"][f"{pop}|{fam}|{g}"] = round(float(((gb > gv) + 0.5 * (gb == gv)).mean()), 3)
    fails = []
    for pop, fams in GATE.items():
        for fam in fams:
            r = res["cells"][f"{pop}|test|{fam}"]
            if r["win"] < LINE or r["ci95"][0] < LO:
                fails.append(f"{pop}:{fam} {r['win']} {r['ci95']}")
    for pop in ("real_F", "iro_F"):
        w1, w2, w3 = (res["cells"][f"{pop}|test|{b}"]["win"] for b in ("B1", "B2", "B3"))
        if not (w1 <= w2 + 0.02 and w2 <= w3 + 0.02):
            fails.append(f"{pop}:mono {w1} {w2} {w3}")
    res["fails"] = fails
    res["pass"] = not fails
    (OUT / "n2c.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for pop in ("real_F", "iro_F", "iro_M", "real_M"):
        print(pop, {k.split("|")[2]: v["win"] for k, v in res["cells"].items() if k.startswith(f"{pop}|test|")}, flush=True)
        print("  L1", {k.split("|")[1]: v for k, v in res["l1_cells"].items() if k.startswith(f"{pop}|")}, flush=True)
    print("S groups", {k: v for k, v in res["group_cells"].items() if k.startswith("iro_M|S")}, flush=True)
    print("fails", fails, "PASS" if res["pass"] else "FAIL", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
