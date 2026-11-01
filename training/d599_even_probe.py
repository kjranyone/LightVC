"""D599(推論だけ): 同じ高さのまま偶数倍音を強めた音を、学習の同一性の損失と同じ ECAPA・学習外 LDA が報いるか(腕 60 の C1 の帰属)。
  --stage render: EVEN<k>B<bb> = 目標本人の別録音(cen_TTT の 8 s)の WORLD 分析で S1 = 合成(f0 × 狙い / 中央値)・S2 = 合成(2 × 同じ f0)を RMS をそろえて (1 − β) S1 + β S2。β = 0 は REFKC<k> と同じ合成。
  --stage id: ECAPA(idloss.Ecapa)と WavLM-SV の cos(刺激 8 s, 目標の参照 d37/_work/ref_TTT.wav の先頭 20 s)の対の差。
    uv run python d599_even_probe.py --stage render --sets dev2 dev3
    uv run python d599_even_probe.py --stage id --sets dev2 dev3 --out ../results/conv_p0/d599/id.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H, SR, boot, harvest

KS = (5, 12)
BS = (0, 30, 60)


def one(job: tuple) -> list:
    from d6_unit_cov import load48
    S, key, ti = job
    xs = load48(S / "d56" / f"{key}_SRC.wav").astype(np.float64)[: 8 * SR]
    xt = load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR]
    f0 = harvest(xt)
    t = np.arange(len(f0)) * 0.005
    sp = pw.cheaptrick(xt, f0, t, SR)
    ap = pw.d4c(xt, f0, t, SR)
    mt = float(np.median(f0[f0 > 0]))
    fs = harvest(xs)
    mo = float(np.median(fs[fs > 0]))
    chk = []
    for k in KS:
        want = mo * 2 ** (k / 12)
        s1 = pw.synthesize(f0 * want / mt, sp, ap, SR, 5.0)
        s2 = pw.synthesize(2 * f0 * want / mt, sp, ap, SR, 5.0)
        n = min(len(s1), len(s2))
        s1, s2 = s1[:n] / (np.sqrt((s1[:n] ** 2).mean()) + 1e-12), s2[:n] / (np.sqrt((s2[:n] ** 2).mean()) + 1e-12)
        for b in BS:
            nm = f"EVEN{k}B{b:02d}"
            o = S / "d570" / nm / f"{key}_{nm}.wav"
            y = (1 - b / 100) * s1 + (b / 100) * s2
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), SR, subtype="PCM_16")
            if b == 0:
                r = load48(S / "d570" / f"REFKC{k}" / f"{key}_REFKC{k}.wav").astype(np.float64)
                e = load48(o).astype(np.float64)
                m = min(len(r), len(e))
                chk.append((k, float(np.abs(r[:m] / (np.abs(r[:m]).max() + 1e-9) - e[:m] / (np.abs(e[:m]).max() + 1e-9)).max())))
    return chk


def jobs_for(sets: list) -> list:
    jobs = []
    for s in sets:
        S = H / s
        ok = {m["ti"] for m in json.loads((S / "cen_reg" / "meta.json").read_text()) if m["ok"]}
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            key = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if r["ti"] in ok and (S / "d570" / "REFKC5" / f"{key}_REFKC5.wav").exists():
                jobs.append((S, key, r["ti"]))
    return jobs


def render(sets: list) -> int:
    with Pool(NPROC) as pl:
        res = [c for r in pl.imap_unordered(one, jobs_for(sets)) for c in r]
    for k in KS:
        d = np.array([v for kk, v in res if kk == k])
        print(f"EVEN{k}B00 vs REFKC{k}: n {len(d)} max_abs_diff {d.max():.5f}", flush=True)
    return 0 if all(v < 2e-3 for _, v in res) else 2


def ident(sets: list, out: str) -> int:
    import torch
    import idloss as ID
    from d6_unit_cov import load48
    dev = "cuda"
    ec, wl = ID.Ecapa(dev), ID.WavlmSV(dev)

    def emb(x: np.ndarray) -> tuple:
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        with torch.no_grad():
            t_ = torch.from_numpy(x.astype(np.float32))[None].to(dev)
            return ec(t_)[0].cpu().numpy(), wl(t_)[0].cpu().numpy()

    vers = [f"EVEN{k}B{b:02d}" for k in KS for b in BS] + ["OURS5RY58G", "OURS5RY60G", "OURS12RY58G", "OURS12RY60G"]
    rows: dict = {v: {} for v in vers}
    for s in sets:
        S = H / s
        refs: dict = {}
        for v in vers:
            for f in sorted((S / "d570" / v).glob("p*_t*.wav")):
                m = re.match(r"(p\d+_t(\d+))_", f.name)
                key, ti = m.group(1), int(m.group(2))
                if ti not in refs:
                    refs[ti] = emb(load48(S / "d37" / "_work" / f"ref_{ti:03d}.wav")[: 20 * SR].astype(np.float64))
                e = emb(load48(f)[: 8 * SR].astype(np.float64))
                rows[v][(s, key)] = {"t": ti, "ecapa": float(e[0] @ refs[ti][0]), "wavlm": float(e[1] @ refs[ti][1])}
            print(s, v, len(rows[v]), flush=True)
    rng = np.random.default_rng(599)
    rep: dict = {"sets": {v: {m: round(float(np.mean([r[m] for r in rows[v].values()])), 4) for m in ("ecapa", "wavlm")} | {"n": len(rows[v])} for v in vers}, "pairs": {}}
    pairs = [(f"EVEN{k}B{b:02d}", f"EVEN{k}B00") for k in KS for b in BS[1:]] + [("OURS5RY60G", "OURS5RY58G"), ("OURS12RY60G", "OURS12RY58G")]
    for A_, B_ in pairs:
        ks = sorted(set(rows[A_]) & set(rows[B_]))
        tg = np.array([f"{key[0]}|{rows[A_][key]['t']}" for key in ks])
        for m in ("ecapa", "wavlm"):
            d = np.array([rows[A_][key][m] - rows[B_][key][m] for key in ks])
            rep["pairs"][f"{A_}-{B_}:{m}"] = {"n": len(ks), "diff": round(float(d.mean()), 4), "ci95": boot(d, tg, rng)}
            print(f"{A_}-{B_}:{m}", rep["pairs"][f"{A_}-{B_}:{m}"], flush=True)
    for v in vers:
        print(v, rep["sets"][v], flush=True)
    Path(out).write_text(json.dumps(rep, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("render", "id"), required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    return render(a.sets) if a.stage == "render" else ident(a.sets, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
