"""段 P105 step a(results/snv_p105/prereg.yaml・学習なし): 本物の声の写しで、平滑なしの界面の代価が「包絡の分析の窓を決める高さの読み違え」に依るかを見る。
包絡の窓だけを別の高さ(オフラインの harvest + stonemask = 先読みあり・診断だけ / 自分の高さの因果の中央値)で決め直し、描画器に渡す高さは変えない。
    cd training && uv run --no-sync python snv_diag/p105_f0win.py [n=64] [out=results/snv_p105/f0win.json]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_dev_eval as DE
import snv_p0_front as P
import pae as PA
import snv_p1_eval as EV
from snv_a0_measure import to16

ROOT = Path(__file__).resolve().parent.parent.parent


def med5(f0: np.ndarray) -> np.ndarray:
    q = np.concatenate([np.repeat(f0[:1], 4), f0])
    w = np.stack([q[i : i + len(f0)] for i in range(5)])
    m = np.sort(np.where(w > 0, w, np.nan), 0)
    k = (~np.isnan(m)).sum(0)
    out = np.array([m[(k[t] - 1) // 2, t] if k[t] else 0.0 for t in range(len(f0))])
    return np.where(f0 > 0, np.nan_to_num(out), 0.0).astype(np.float32)


def main() -> None:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, out = int(opt.get("n", 64)), ROOT / opt.get("out", "results/snv_p105/f0win.json")
    items = DE.cv_items("devA", n)
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(EV.DEV).eval()
    mos = lambda w: float(utmos(torch.from_numpy(to16(w.astype(np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV), 16000).item())
    rows: dict = {}
    gen = EV.load_gen()
    for it in items:
        u = it["u"]
        y, f0 = u["y"], u["f0"].astype(np.float32)
        T = len(f0)
        x = librosa.load(u["wav_path"], sr=P.SR, mono=True)[0].astype(np.float32)
        x = np.concatenate([x, np.zeros(max(0, T * P.H - len(x)), np.float32)])[: T * P.H]
        fo, tt = pyworld.harvest(x.astype(np.float64), P.SR, frame_period=5.0, f0_floor=60.0, f0_ceil=600.0)
        fo = pyworld.stonemask(x.astype(np.float64), fo, tt, P.SR)
        fo = np.concatenate([fo, np.zeros(max(0, T - len(fo)))])[:T].astype(np.float32)
        both = (f0 > 0) & (fo > 0)
        fw = {"own": f0, "orw": np.where(both, fo, f0).astype(np.float32), "med": med5(f0)}
        r = {"gross": float((np.abs(np.log2(np.maximum(f0, 1) / np.maximum(fo, 1)))[both] > 0.2).mean()) if both.any() else 0.0, "dev_cent": float(np.median(1200 * np.abs(np.log2(f0[both] / fo[both])))) if both.any() else 0.0}
        v = (f0 > 0).astype(np.float32)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        for k, f in fw.items():
            e = P.envelope(x, f)
            env = np.concatenate([((e[0] - PA.C0_SIL) / 10)[None], e[1:25] / 10], 0).astype(np.float32)
            if k == "own":
                r["own_recompute_err"] = float(np.abs(env - y[:25]).max())
            c0 = env[0]
            vv = (v[2:] > 0) & (v[1:-1] > 0) & (v[:-2] > 0)
            r[f"zz_{k}"] = float(np.sqrt(np.mean((c0[2:] - 2 * c0[1:-1] + c0[:-2])[vv] ** 2))) if vv.any() else 0.0
            for sm in ("1", "0"):
                os.environ["SNV_ESM"] = sm
                cond = torch.from_numpy(np.concatenate([env, lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(EV.DEV)
                cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
                w = EV.draw(gen, cond, torch.from_numpy(f0)[None].to(EV.DEV), None)
                r[f"{k}_{'sm' if sm == '1' else 'nosm'}"] = round(mos(w), 4)
        rows[it["src"]] = r
    ks = sorted(next(iter(rows.values())))
    s = {k: round(float(np.mean([r[k] for r in rows.values()])), 4) for k in ks}
    def d(a: str, b: str) -> list:
        q = np.array([r[a] - r[b] for r in rows.values()])
        return [round(float(q.mean()), 4), round(float(1.96 * q.std() / np.sqrt(len(q))), 4)]
    s["diff"] = {"own_nosm-own_sm": d("own_nosm", "own_sm"), "orw_nosm-orw_sm": d("orw_nosm", "orw_sm"), "med_nosm-med_sm": d("med_nosm", "med_sm"), "orw_nosm-own_nosm": d("orw_nosm", "own_nosm"), "orw_sm-own_sm": d("orw_sm", "own_sm"), "med_nosm-own_nosm": d("med_nosm", "own_nosm")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n": len(rows), "summary": s, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(s, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
