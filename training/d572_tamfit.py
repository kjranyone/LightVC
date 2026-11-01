"""D572a(学習なし・探索・判定に使わない): 目標の動き方を少数の話者の値(2 次の追従系: 時定数 τ・減衰 ζ・振れ幅 ρ を包絡の帯 3 群ごと)で表せるか。
各目標の参照(20 s)で、単位の目標の列(表 T の硬い単位の列)を追従系に通した軌道が実の包絡(CheapTrick c1〜c24)にどれだけ近いかで値を当てはめ、
独立の録音(cen 25 s)で 静的(soft = T·事後 / hard = T[:, 単位])・追従(本人の値 / 他人の値 / 全員の中央値)の残差を比べる。
    uv run python d572_tamfit.py --out <s>(d571par)--res ../results/conv_p0/d572a.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ((0, 4), (4, 12), (12, 24))
TAUS = (5.0, 10.0, 20.0, 30.0, 45.0, 65.0, 90.0)
ZETAS = (0.5, 0.7, 1.0, 1.4)
RHOS = (0.8, 1.0, 1.2, 1.4)
DT = 0.005


def tam(x: np.ndarray, tau: float, zeta: float, rho: float) -> np.ndarray:
    w = 1000.0 / tau
    b, a = ss.bilinear([w * w], [1.0, 2 * zeta * w, w * w], fs=1.0 / DT)
    zi = ss.lfilter_zi(b, a)
    y = np.stack([ss.lfilter(b, a, r, zi=zi * r[0])[0] for r in x])
    m = x.mean(1, keepdims=True)
    return m + rho * (y - m)


def fit(E: np.ndarray, X: np.ndarray, v: np.ndarray) -> list:
    out = []
    for lo, hi in GROUPS:
        best, bp = 1e18, None
        for tau, zeta, rho in itertools.product(TAUS, ZETAS, RHOS):
            r = E[lo:hi, v] - tam(X[lo:hi], tau, zeta, rho)[:, v]
            s = float((r ** 2).mean())
            if s < best:
                best, bp = s, (tau, zeta, rho)
        out.append(bp)
    return out


def apply(X: np.ndarray, ps: list) -> np.ndarray:
    return np.concatenate([tam(X[lo:hi], *p) for (lo, hi), p in zip(GROUPS, ps)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as FH
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    from d6_unit_cov import load48
    dev = "cuda"
    sc = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev)
    c1.load_state_dict(sc["net"])
    c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt", map_location="cpu").cpu().numpy()
    K = Cb.shape[0]
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)

    @torch.no_grad()
    def feats(x: np.ndarray) -> tuple:
        n = len(x) // N.HOP
        x = x[: n * N.HOP].astype(np.float32)
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, n - p.shape[1])), mode="edge")
        ft, _ = FH.teacher_f0(x, n)
        ft = np.pad(ft[:n].astype(np.float64), (0, max(0, n - len(ft))))[:n]
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, ft), ((0, 0), (1, 1)), mode="edge")
        e = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n][1:25].astype(np.float64)
        return e, p, ft > 0

    od = Path(a.out)
    items = json.loads((od / "items.json").read_text())
    tg = {}
    for it in items:
        if it["ti"] in tg:
            continue
        rf = [feats(load48(r)) for r in it["ref"]]
        E = np.concatenate([f[0] for f in rf], 1)
        P_ = np.concatenate([f[1] for f in rf], 1)
        V = np.concatenate([f[2] for f in rf])
        T = C0.table(E, P_.argmax(0), K, Cb)
        ps = fit(E, T[:, P_.argmax(0)], V)
        ce, cp, cv = feats(load48(od / f"cen_{it['ti']:03d}.wav"))
        tg[it["ti"]] = {"T": T, "ps": ps, "cen": (ce, cp, cv)}
        print(it["ti"], ps, flush=True)
    keys = sorted(tg)
    med = [tuple(float(np.median([tg[t]["ps"][g][q] for t in keys])) for q in range(3)) for g in range(3)]
    rows = []
    for i, t in enumerate(keys):
        T, (ce, cp, cv) = tg[t]["T"], tg[t]["cen"]
        u = cp.argmax(0)
        other = tg[keys[(i + 1) % len(keys)]]["ps"]
        pred = {"soft": T @ cp, "hard": T[:, u], "tam_own": apply(T[:, u], tg[t]["ps"]), "tam_other": apply(T[:, u], other), "tam_med": apply(T[:, u], med)}
        rows.append({"ti": t} | {k: float(np.sqrt(((ce[:, cv] - y[:, cv]) ** 2).mean())) for k, y in pred.items()})
    rep = {"params_median": med, "params": {str(t): tg[t]["ps"] for t in keys},
           "rmse_mean": {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0] if k != "ti"}, "rows": rows}
    d = np.array([r["tam_own"] - r["tam_other"] for r in rows])
    rep["own_minus_other"] = {"mean": round(float(d.mean()), 4), "frac_own_better": round(float((d < 0).mean()), 3)}
    Path(a.res).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["rmse_mean"]), rep["own_minus_other"], "median params", med, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
