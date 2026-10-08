"""phys_e2: VCTK 実並行の男→女で、E のゼロショット推定と母集団定数を比べる。事前登録 results/phys_e2/prereg.yaml。

    CUDA_VISIBLE_DEVICES= uv run python eval_phys_e2.py --ckpt ../results/phys_e1/last.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
import physvc as PV
from a1_vtl_warp import VC, mfcc, voiced_10ms
from train_dec2 import load48
from train_phys_e1 import LEV_DB, R_MAX

ROOT = Path(__file__).resolve().parent.parent
LA = 480


def lar_lev(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lar = D.lar_frames(x, 24, la=LA)
    xp = D.preemph(x)
    K = lar.shape[0]
    pad = np.concatenate([np.zeros(D.W - LA), xp, np.zeros(D.H + LA)])
    fr = np.lib.stride_tricks.sliding_window_view(pad, D.W)[0:K * D.H:D.H]
    return lar, 10 * np.log10(np.maximum((fr ** 2).mean(1), 1e-12))


def reference(spk: str) -> np.ndarray:
    acc, n = [], 0
    for u in range(25, 80):
        p = VC / f"wav48/{spk}/{spk}_{u:03d}.wav"
        if not p.exists():
            continue
        lar, lev = lar_lev(load48(str(p)).astype(np.float64))
        s = lar[lev >= lev.max() - LEV_DB]
        acc.append(s)
        n += len(s)
        if n >= R_MAX:
            break
    return np.concatenate(acc)[:R_MAX].astype(np.float32)


def aligned(m: str, f: str) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    Lm, Lf = [], []
    for u in range(1, 25):
        pm, pf = VC / f"wav48/{m}/{m}_{u:03d}.wav", VC / f"wav48/{f}/{f}_{u:03d}.wav"
        if not (pm.exists() and pf.exists()):
            continue
        xm, xf = load48(str(pm)).astype(np.float64), load48(str(pf)).astype(np.float64)
        xm, _ = librosa.effects.trim(xm, top_db=35)
        xf, _ = librosa.effects.trim(xf, top_db=35)
        if len(xm) < D.SR // 2 or len(xf) < D.SR // 2:
            continue
        Mm, Mf = mfcc(xm), mfcc(xf)
        _, wp = librosa.sequence.dtw(X=Mm, Y=Mf, metric="cosine")
        vm, _ = voiced_10ms(xm, Mm.shape[1])
        vf, _ = voiced_10ms(xf, Mf.shape[1])
        keep = [(i, j) for i, j in wp[::-1] if vm[i] and vf[j]]
        if len(keep) < 20:
            continue
        lm, lf = D.lar_frames(xm, 24, la=LA), D.lar_frames(xf, 24, la=LA)
        Lm.append(lm[np.clip(np.array([i for i, _ in keep]) * 4, 0, len(lm) - 1)])
        Lf.append(lf[np.clip(np.array([j for _, j in keep]) * 4, 0, len(lf) - 1)])
    return np.concatenate(Lm).astype(np.float32), np.concatenate(Lf).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/phys_e1b/last.pt"))
    ap.add_argument("--arm", default="e1b", choices=["e1b", "e1c"])
    a = ap.parse_args()
    if a.arm == "e1c":
        import train_phys_e1c as TM
        grid, einp = PV.GRID_W, TM.einp
    else:
        import train_phys_e1 as TM
        grid, einp = PV.GRID, PV.shape
    enc = TM.Enc()
    enc.load_state_dict(torch.load(a.ckpt, map_location="cpu")["enc"])
    enc.eval()
    a1 = json.loads((ROOT / "results/artic_a1/vtl_warp.json").read_text())
    pair_alpha = {tuple(p["pair"]): p["alpha_median"] for p in a1["per_pair"]}
    rows = []
    from multiprocessing import Pool
    spks = sorted({s for pr in a1["pairs"] for s in pr})
    with Pool(8) as pool:
        refs = dict(zip(spks, (torch.from_numpy(r) for r in pool.map(reference, spks))))
        al = pool.starmap(aligned, [tuple(pr) for pr in a1["pairs"]])
    for (m, f), (Lm, Lf) in zip(a1["pairs"], al):
        Em, Ef = PV.env_grid(torch.from_numpy(Lm)[None], grid), PV.env_grid(torch.from_numpy(Lf)[None], grid)
        Rm, Rf = PV.env_grid(refs[m][None], grid), PV.env_grid(refs[f][None], grid)
        ones_r = torch.ones(1, Rm.shape[1])
        m_ref = Rm.mean(1)
        mk = torch.ones(1, Em.shape[1])
        with torch.no_grad():
            dzs = enc(einp(Rf), torch.ones(1, Rf.shape[1])) - enc(einp(Rm), ones_r)
        conds = {"none": torch.zeros(1, 6), "pop": torch.tensor([[np.log(1.139), 0, 0, 0, 0, 0]], dtype=torch.float32), "zs": dzs}
        if (m, f) in pair_alpha:
            conds["pair_uniform_ref"] = torch.tensor([[np.log(pair_alpha[(m, f)]), 0, 0, 0, 0, 0]], dtype=torch.float32)
        r = {"pair": [m, f], "n": int(Em.shape[1]), "dphi_zs": [round(float(v), 3) for v in dzs[0]]}
        for c, d in conds.items():
            per = PV.shape(PV.transform_env(Em, m_ref, d)) - PV.shape(Ef)
            band = torch.as_tensor(PV.band_of(per.shape[-1]))
            l = per[..., band].pow(2).mean(-1).sqrt()[0]
            r[c] = round(float(l.median()), 4)
        rows.append(r)
        print(json.dumps(r), flush=True)
    summ = {c: round(float(np.mean([r[c] for r in rows if c in r])), 4) for c in ("none", "pop", "zs", "pair_uniform_ref")}
    win = float(np.mean([r["zs"] < r["pop"] for r in rows]))
    rep = {"prereg": "results/phys_e2/prereg.yaml", "ckpt": a.ckpt, "arm": a.arm, "n_pairs": len(rows), "mean_shape_lsd": summ,
           "frac_pairs_zs_better_than_pop": round(win, 3),
           "dv_zs_median": round(float(np.median([r["dphi_zs"][0] for r in rows])), 4),
           "verdict": "PASS" if summ["zs"] < summ["pop"] and win >= 0.6 else "FAIL", "rows": rows}
    out = ROOT / ("results/phys_e2/result.json" if a.arm == "e1b" else f"results/phys_e2/result_{a.arm}.json")
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("mean_shape_lsd", "frac_pairs_zs_better_than_pop", "dv_zs_median", "verdict")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
