"""D272(学習なし・見当づけ・results/conv_p0/d271_verdict.yaml の next): 男声 → 女声の条件に、行ごとの因果な差分の利得(隣り合うフレームの差の大きさを目標の参照の値へ・漏れ λ 0.2・登録の定数)+ 行ごとの分布の写し(D271)をかけて Python の rvoc3hi で鳴らす。
    uv run python d272_dgain.py --s <dev の親> --n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d271_condmv import ROWS, stats

ROOT = Path(__file__).resolve().parent.parent
ROWSX = ROWS["ENV"] + ROWS["C0"]


def dmag(c: np.ndarray) -> np.ndarray:
    v = c[:, 26] > 0.5
    idx = np.flatnonzero(v)
    ok = np.diff(idx) == 1
    return np.abs(np.diff(c[idx][:, ROWSX], axis=0))[ok].mean(0) + 1e-6


def dgain(c: np.ndarray, k: np.ndarray, lam: float) -> np.ndarray:
    x = c[:, ROWSX]
    y = np.zeros_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = (1 - lam) * (y[t - 1] + k * (x[t] - x[t - 1])) + lam * x[t]
    o = c.copy()
    o[:, ROWSX] = y
    return o


def mv(c: np.ndarray, so: dict, st: dict) -> np.ndarray:
    v = c[:, 26] > 0.5
    y = c.copy()
    for m, sel in (("v", v), ("u", ~v)):
        if m in so and m in st:
            for r in ROWS["ENV"] + ROWS["PER"] + ROWS["C0"]:
                if r >= 27 and m == "u":
                    continue
                y[sel, r] = (c[sel, r] - so[m][0][r]) / so[m][1][r] * st[m][1][r] + st[m][0][r]
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--lam", type=float, default=0.2)
    a = ap.parse_args()
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    D = Path(a.s) / "dev"
    od = D / "d271"
    s_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to("cuda")
    gen.load_state_dict(s_["ema"]); gen.eval().requires_grad_(False)

    @torch.no_grad()
    def render(c: np.ndarray) -> np.ndarray:
        cond = torch.from_numpy(c.T.astype(np.float32))[None].to("cuda")
        f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0).astype(np.float32)
        y = gen(cond, TR.excitation(torch.from_numpy(f0)[None].to("cuda"), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for f in sorted((D / "d56").glob("p00*_SRC.wav"))[: a.n]:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        c = np.fromfile(od / f"cv_{f.stem}.f32", "<f4").reshape(-1, 31).astype(np.float64)
        co = np.fromfile(od / f"opcv_{pi:04d}_t{ti:03d}.f32", "<f4").reshape(-1, 31).astype(np.float64)
        ct = np.fromfile(od / f"ref_{ti:03d}.f32", "<f4").reshape(-1, 31).astype(np.float64)
        k = dmag(ct) / dmag(co)
        sf.write(od / f"{f.stem[:-4]}_DG.wav", render(mv(dgain(c, k, a.lam), stats(dgain(co, k, a.lam)), stats(ct))), N.SR)
    print("rendered", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
