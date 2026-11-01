"""D108(学習なし): F1 のオクターブ誤り(2〜4%)を確信度で抑える因果な後処理。各フレームの確信度 c = 最大ビンの近傍 ±4 の事後の和。
  HOLD(τ, J): 有声区間の中で、前の採用値から |Δ log2| > J の跳びを確信度 c < τ のときは採らず前の値を保つ(因果)。
  UNV(τ): 確信度 c < τ のフレームを無声に。OCT(τ): c < τ かつ話者の基準の高さ(ここでは教師 f0 の発話の中央値で代用 = 製品では操作者の登録の中央値)から 0.6 oct 超外れたら基準に近いオクターブへ折り返す(+ HOLD)。
教師 f0 に対するオクターブ誤り(|Δ log2| > 0.5)・有声の取りこぼし / 誤検出・有声区間内の 0.6 oct 超の跳び / 秒を、実の男声(VCTK)・実の女声で比べる。出力 = results/conv_p0/d108.json。
    uv run python d108_f0_conf.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
VM = ["p226", "p227", "p232", "p237", "p241", "p243", "p245", "p246", "p247", "p251", "p252", "p254"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    import f0est as FE
    import f0hi as H
    import nvoc as N
    from d6_unit_cov import load48
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(s1["net"]); f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())
    fr = N.SR / N.HOP

    def decode(pr: np.ndarray) -> tuple:
        k = pr.argmax(0)
        T = pr.shape[1]
        f, c = np.zeros(T), np.zeros(T)
        for t in range(T):
            ii = np.clip(np.arange(k[t] - 4, k[t] + 5), 0, pr.shape[0] - 1)
            w = pr[ii, t]
            c[t] = w.sum()
            f[t] = np.exp((w * lhz[ii]).sum() / max(w.sum(), 1e-12))
        return f, c

    def hold(f: np.ndarray, v: np.ndarray, c: np.ndarray, tau: float, J: float) -> np.ndarray:
        out = np.where(v, f, 0.0)
        prev = 0.0
        for t in range(len(f)):
            if not v[t]:
                prev = 0.0
                continue
            if prev > 0 and abs(np.log2(f[t] / prev)) > J and c[t] < tau:
                out[t] = prev
            else:
                prev = f[t]
            if out[t] > 0:
                prev = out[t]
        return out

    def octfix(f: np.ndarray, v: np.ndarray, c: np.ndarray, tau: float, med: float) -> np.ndarray:
        g = f.copy()
        r = np.log2(np.maximum(g, 1e-9) / med)
        k = (c < tau) & v & (np.abs(r) > 0.6)
        g[k] = g[k] * 2.0 ** (-np.round(r[k]))
        return g

    def metrics(ft: np.ndarray, f: np.ndarray) -> tuple:
        b = (ft > 0) & (f > 0)
        oct_ = float(np.mean(np.abs(np.log2(f[b] / ft[b])) > 0.5)) if b.any() else 0.0
        miss = float(np.mean((ft > 0) & (f <= 0)) / max(np.mean(ft > 0), 1e-9))
        fa = float(np.mean((ft <= 0) & (f > 0)) / max(np.mean(ft <= 0), 1e-9))
        lf = np.log2(np.where(f > 0, f, 1.0))
        dv = np.abs(np.diff(lf))[(f[1:] > 0) & (f[:-1] > 0)]
        return oct_, miss, fa, float((dv > 0.6).sum() / (len(f) / fr))

    sets = {"male_real_vctk": [f for s in VM for f in sorted(glob.glob(str(ROOT / f"data/vctk/VCTK-Corpus/VCTK-Corpus/wav48/{s}/*.wav")))[10:14]],
            "female_real": sorted(glob.glob(str(Path(a.s) / "d37/cen_*.wav")))}
    variants = {"base": None, "hold_t7_j05": (0.7, 0.5), "unv_t3": 0.3, "oct_t5": 0.5, "oct_t7": 0.7, "oct_t7_hold": 0.7, "oct_t9_hold": 0.9}
    import random
    med_by_set = {}
    rep = {}
    for nm, files in sets.items():
        acc = {v: [] for v in variants}
        for p in files:
            x = load48(p).astype(np.float32)[: 8 * 48000]
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            ft, _ = H.teacher_f0(x, n)
            ft = ft[:n].astype(np.float64)
            with torch.no_grad():
                v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x)[None].to(dev)], 1), n))
                vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
                pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
            m = min(len(vo), pr.shape[1])
            f, c = decode(pr[:, :m])
            vo, ftm = vo[:m], ft[:m]
            med = float(np.exp(np.median(np.log(ftm[ftm > 0])))) if (ftm > 0).any() else 150.0
            for vn, prm in variants.items():
                if vn == "base":
                    out = np.where(vo, f, 0.0)
                elif vn.startswith("hold"):
                    out = hold(f, vo, c, *prm)
                elif vn.startswith("oct"):
                    g = octfix(f, vo, c, prm, med)
                    out = hold(g, vo, c, prm, 0.5) if vn.endswith("hold") else np.where(vo, g, 0.0)
                else:
                    out = np.where(vo & (c >= prm), f, 0.0)
                acc[vn].append(metrics(ftm, out))
        rep[nm] = {vn: dict(zip(["oct_err", "voicing_miss", "voicing_fa", "jumps_per_s"], [round(float(np.mean([r[i] for r in rs])), 4) for i in range(4)])) for vn, rs in acc.items()}
        print(nm, json.dumps(rep[nm]), flush=True)
    (ROOT / "results/conv_p0/d108.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
