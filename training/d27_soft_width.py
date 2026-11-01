"""D27: 柔らかい引き(β = 1)の出力の時間平滑の幅: TAB = 幅 3(現行)・W1 = 平滑なし・W2 = 幅 2。ブザー(線)も測る。
旧: D25: 柔らかい引きの鋭さ β の掃引(D24 から派生): ê = Σ p^β(u) T[u] / Σ p^β(u)(因果 3 平均)。β = 0.5 / 1 / 2 / 4 と硬い引き(TAB)。
旧: D24: 明瞭さ(CER)と同一性(学習外 LDA)のトレードオフの地図(学習なし)。女声 → 女声(評価プール 150 人・元の声 = 次の目標の話者の自然な台詞)。
包絡の作り方だけを替える(f0 = 元の声の有声 log f0 を目標の中央値へ移動・c0 = 元の声の実の c0・周期性 = 目標の平均):
  SRCENV = 元の声の実の包絡(音高だけ移す・明瞭さの上限の目安)・TAB = 表の引き(硬い単位・因果 3 平均 = 製品)・SOFT = Σ p(u) T[u](C1 の事後)・TOP3 = 上位 3 単位の事後で重みづけ・
  MIX = 0.5·TAB + 0.5·元の声の包絡・DYN = 表 + 元の声の単位の中の動き(= D16 P1d)。
出力 = d4b の形式の wav(cen・pNNNN_tNNN_<cond>・SRC)。CER は d23_cer.py、同一性は d4b_lda_verifier.py で測る。
    uv run python d24_clarity_map.py --out_dir <scratchpad>/d24 --n_tg 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import diag_pool as P

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=60)
    a = ap.parse_args()
    import c1_content as CC
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    pool = P.load_pool(150, 3)
    S = len(pool)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)

    @torch.no_grad()
    def post(x):
        n = len(x) // N.HOP
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy()
        return np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]

    @torch.no_grad()
    def render(c0, env24, f0, per):
        cc = np.concatenate([c0[None] - PA.C0_SIL, env24], 0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    import eval_nvoc as E
    AM: dict = {}
    ca = lambda z: C3.causal_avg(torch.from_numpy(z.astype(np.float32))[None], 3)[0].numpy()
    pi = 0
    for ti in range(min(a.n_tg, S)):
        t = pool[ti]
        T = t["T"].astype(np.float64)
        sf.write(od / f"cen_{ti:03d}.wav", t["cen"], 48000)
        src = pool[(ti + 1) % S]
        for su in src["utts"]:
            u = su["u"].astype(int)
            n = len(u)
            p = post(su["x"])[:, :n]
            vo = su["f0"] > 0
            lfs = np.log(np.maximum(su["f0"], 1.0))
            med = float(np.median(lfs[vo])) if vo.any() else t["mu"]
            f0o = np.where(vo, np.exp(lfs - med + t["mu"]), 0.0)
            perc = np.repeat(t["per_mean"][:, None], n, 1)
            er = su["er"].astype(np.float64)
            raw = T @ p
            c2 = C3.causal_avg(torch.from_numpy(raw.astype(np.float32))[None], 2)[0].numpy()
            envs = {"TAB": ca(raw), "W1": raw, "W2": c2}
            for c, ev in envs.items():
                y = render(su["c0"], ev, f0o, perc)
                sf.write(od / f"p{pi:04d}_t{ti:03d}_{c}.wav", y, 48000)
                AM.setdefault(c, []).append(E.amline(y))
            sf.write(od / f"p{pi:04d}_t{ti:03d}_SRC.wav", su["x"], 48000)
            pi += 1
        print("render", ti, {k: round(float(np.mean(v)), 3) for k, v in AM.items()}, flush=True)
    import json as _j
    (od / "am.json").write_text(_j.dumps({k: round(float(np.mean(v)), 3) for k, v in AM.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
