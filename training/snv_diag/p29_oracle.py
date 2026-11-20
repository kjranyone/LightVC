"""段 P29 step a の a3d(results/snv_p29/prereg.yaml・学習なし・評価の天井だけ): 声を跨ぐ変換が、操作者の時間・抑揚・有声の並びを保つ出力として届きうる余弦の天井。
組 = p29_same.py と同じ 242 組。目標の声 v が同じ文を読んだ発話の包絡と周期性を、源の発話の時間に写す(発話ごとに標準化した c0〜c12 と有声の旗の上の DTW・帯 0.2)。
warp = 写した包絡 + 同じ道で写した v の高さ(中央値だけ源に合わせる)。oracle = 写した包絡を源の高さと有声の並びで描く(源が有声で写した先が無声のフレームは、前後 8 フレームの中の最も近い有声のフレームに替える)。
時刻を揃えた対はこの天井の測りにだけ使う(学習・目標・損失に使わない)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p29_oracle.py [out=results/snv_p29/oracle.json]
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
import nvoc as N
import p29_same as S
import probe_rig as G
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV


def feat(u: dict) -> np.ndarray:
    y = u["y"][:13]
    return np.concatenate([(y - y.mean(1, keepdims=True)) / np.maximum(y.std(1, keepdims=True), 1e-6), 2.0 * u["v"][None]], 0)


def align(a: dict, b: dict) -> np.ndarray:
    _, wp = librosa.sequence.dtw(X=feat(a), Y=feat(b), metric="euclidean", global_constraints=True, band_rad=0.2)
    wp = wp[::-1]
    j = np.zeros(a["y"].shape[1], np.int64)
    for i in range(len(j)):
        q = wp[wp[:, 0] == i, 1]
        j[i] = int(np.median(q)) if len(q) else j[i - 1]
    return j


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = ROOT / opt.get("out", "results/snv_p29/oracle.json")
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(trim(y16)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    gen = EV.load_gen()

    @torch.no_grad()
    def render(y: np.ndarray, f0: np.ndarray) -> np.ndarray:
        v = (f0 > 0).astype(np.float32)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0).astype(np.float32))[None].to(DEV)
        ex = TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        return gen(cond, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    P = S.pairs()
    du = {(v, s): T.load_utt(T.FEAT / f"d/{v}/{s}.npz") for p in P for v, s in ((p["u"], p["su"]), (p["v"], p["sv"]))}
    Et = {v: emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")).astype(np.float32) for t in ("t00", "t01", "t02")])) for v in sorted({p["v"] for p in P})}
    med = lambda u: float(np.median(u["f0"][u["f0"] > 0]))
    rows = []
    for n, p in enumerate(P):
        s, t = du[(p["u"], p["su"])], du[(p["v"], p["sv"])]
        j = align(s, t)
        mis0 = float(np.mean(t["v"][j][s["v"] > 0] == 0))
        tv = np.flatnonzero(t["v"] > 0)
        jo = j.copy()
        for i in np.flatnonzero((s["v"] > 0) & (t["v"][j] == 0)):
            k = tv[np.argmin(np.abs(tv - j[i]))]
            if abs(int(k) - int(j[i])) <= 8:
                jo[i] = k
        mis1 = float(np.mean(t["v"][jo][s["v"] > 0] == 0))
        ew = emb(w16(render(t["y"][:, j], t["f0"][j] * (med(s) / med(t)))))
        eo = emb(w16(render(t["y"][:, jo], s["f0"])))
        rows.append({"u": p["u"], "v": p["v"], "ct_warp": round(float(ew @ Et[p["v"]]), 4), "ct_oracle": round(float(eo @ Et[p["v"]]), 4), "mis_before": round(mis0, 3), "mis_after": round(mis1, 3), "len_ratio": round(len(s["v"]) / len(t["v"]), 3)})
        if n % 40 == 0:
            print(n, rows[-1], flush=True)
    same = json.loads((ROOT / "results/snv_p29/same.json").read_text())
    sp = json.loads((ROOT / "results/snv_p29/same_sp.json").read_text())
    res = {"prereg": "results/snv_p29/prereg.yaml(a3d)", "summary": {}, "rows": rows}
    for sx in ("ef", "em"):
        I = [i for i, r in enumerate(rows) if r["u"].startswith(sx)]
        m = lambda xs: round(float(np.mean([xs[i] for i in I])), 4)
        o = {"n": len(I), "床": m([q["ct_floor"] for q in same["ref"]]), "天井(目標の自分の読み・源の高さ)": m([q["ct_ceil"] for q in same["ref"]]), "warp": m([r["ct_warp"] for r in rows]), "oracle": m([r["ct_oracle"] for r in rows]),
             "有声の食い違い(前 / 後)": [m([r["mis_before"] for r in rows]), m([r["mis_after"] for r in rows])]}
        for tag in ("snv_p24_c", "snv_p8_wst", "snv_i10_c_pr", "snv_p24_c+ms100"):
            if tag in same["rows"]:
                o[f"変換 {tag}"] = m([r["ct"] for r in same["rows"][tag]])
                us = sorted({rows[i]["u"] for i in I})
                d = np.array([np.mean([rows[i]["ct_oracle"] - same["rows"][tag][i]["ct"] for i in I if rows[i]["u"] == u]) for u in us])
                g = np.random.default_rng(0)
                bs = [float(np.mean(g.choice(d, len(d)))) for _ in range(4000)]
                o[f"oracle − 変換 {tag}"] = [round(float(d.mean()), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]]
        for tag in sp["rows"]:
            o[f"自己(源の高さ){tag}"] = m([r["ct_self_sp"] for r in sp["rows"][tag]])
        res["summary"][sx] = o
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
