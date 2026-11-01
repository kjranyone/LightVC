"""D394: 連続の値の本人らしさ(識別の余白)で候補を対にして比べる(順位の物差し D390 は 80 目標でも区間 ±1.5 で候補の差を判定できない)。
発話ごとに 余白 = cos(出力, 目標の中心) − 同じ組の他の目標の中心との cos の平均。中心 = <組>/cen_reg/cen_TTT.wav(声域をそろえた独立の中心・meta の ok のみ)。
埋め込み = WavLM-SV(microsoft/wavlm-base-plus-sv の x-vector・学習の同一性の損失 = ECAPA とは別の網)と ECAPA(学習と同じ網 = 参考・循環の恐れ)。
群ごとに dev / dev2 / dev3 の glob を与え、同じ発話の名前(pNNNN_tTTT)どうしで対にした差を目標でまとめたブートストラップ。
    uv run python d394_idmargin.py --out ../results/conv_p0/d394.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P

R = P.ROOT / "results/conv_p0"
H = P.H
GROUPS = {
    "PPG18": {"dev": f"{R}/d328/conv/p00*_CV.wav", "dev2": f"{R}/d325/dev2/PPG18/p00*.wav", "dev3": f"{R}/d325/dev3/PPG18/p00*.wav"},
    "A": {"dev": f"{R}/d352m/conv/p00*_CV.wav", "dev2": f"{R}/d353/dev2/PPG24/p00*.wav", "dev3": f"{R}/d353/dev3/PPG24/p00*.wav"},
    "B": {"dev": f"{R}/d374m/conv/p00*_CV.wav", "dev2": f"{R}/d374/dev2/PPG27/p00*.wav", "dev3": f"{R}/d374/dev3/PPG27/p00*.wav"},
    "C": {"dev": f"{R}/d386m_005000/conv/p00*_CV.wav", "dev2": f"{R}/d386/dev2/PPG29/p00*.wav", "dev3": f"{R}/d386/dev3/PPG29/p00*.wav"},
    "PROD": {s: f"{H}/{s}/d114/p00*_G03S10ZH.wav" for s in ("dev", "dev2", "dev3")},
    "KNN": {s: f"{H}/{s}/d135_real/p00*_KNN.wav" for s in ("dev", "dev2", "dev3")},
}


def load16(p) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return resample_poly(x.mean(1), 16000, sr).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="足す群: NAME=<dev の glob>|<dev2 の glob>|<dev3 の glob>(既定の群と対にして比べる)")
    ap.add_argument("--vs", nargs="*", default=[], help="足す群と比べる相手(既定 B)")
    ap.add_argument("--only", nargs="*", default=[], help="この群だけを測る(既定 = 全部)")
    a = ap.parse_args()
    for e in a.extra:
        nm, pats = e.split("=", 1)
        GROUPS[nm] = dict(zip(("dev", "dev2", "dev3"), pats.split("|")))
    if a.only:
        keep = set(a.only) | {e.split("=", 1)[0] for e in a.extra}
        for k in [k for k in GROUPS if k not in keep]:
            del GROUPS[k]
    dev = "cuda"
    from transformers import WavLMForXVector
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    import idloss as ID
    ec = ID.Ecapa(dev)

    @torch.no_grad()
    def emb(p) -> dict:
        x = torch.from_numpy(load16(p))[None].to(dev)
        w = torch.nn.functional.normalize(sv(input_values=x).embeddings, dim=-1)[0].cpu().numpy()
        x48, sr = sf.read(str(p), dtype="float32", always_2d=True)
        e = ec(torch.from_numpy(x48.mean(1))[None].to(dev))[0].cpu().numpy()
        return {"wsv": w, "ecapa": e / np.linalg.norm(e)}

    rep: dict = {"margin": {}, "paired": {}}
    per: dict = {}
    for s in ("dev", "dev2", "dev3"):
        meta = json.loads((H / s / "cen_reg" / "meta.json").read_text())
        ok = sorted(m["ti"] for m in meta if m["ok"])
        C = {ti: emb(H / s / "cen_reg" / f"cen_{ti:03d}.wav") for ti in ok}
        for g, pats in GROUPS.items():
            for f in sorted(glob.glob(pats[s])):
                k = Path(f).name[:10]
                ti = int(k[7:10])
                if ti not in C:
                    continue
                e = emb(f)
                row = {}
                for m in ("wsv", "ecapa"):
                    cs = {t: float(e[m] @ C[t][m]) for t in C}
                    row[m] = cs[ti] - np.mean([v for t, v in cs.items() if t != ti])
                per.setdefault(g, {})[f"{s}_{k}"] = row
        print("set", s, "targets ok", len(ok), {g: sum(1 for k in per.get(g, {}) if k.startswith(s + "_")) for g in GROUPS}, flush=True)
    for g, d in per.items():
        rep["margin"][g] = {m: round(float(np.mean([v[m] for v in d.values()])), 4) for m in ("wsv", "ecapa")} | {"n": len(d)}
    rng = np.random.default_rng(0)
    cmp = [(e.split("=", 1)[0], v) for e in a.extra for v in (a.vs or ["B"])] or [("C", "A"), ("C", "B"), ("A", "B"), ("C", "PROD"), ("A", "PROD"), ("C", "KNN"), ("PROD", "KNN"), ("C", "PPG18")]
    for a_, b_ in cmp:
        if a_ not in per or b_ not in per:
            continue
        keys = sorted(set(per[a_]) & set(per[b_]))
        tg = np.array([k.split("_t")[0].split("_")[0] + "_" + k.split("_t")[1][:3] for k in keys])
        out = {"n": len(keys)}
        for m in ("wsv", "ecapa"):
            d = np.array([per[a_][k][m] - per[b_][k][m] for k in keys])
            us = sorted(set(tg))
            idx = {u: np.flatnonzero(tg == u) for u in us}
            bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
            out[m] = {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        rep["paired"][f"{a_}-{b_}"] = out
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["margin"]), flush=True)
    for k, v in rep["paired"].items():
        print(k, v, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
