"""D32 = D30 の --shrink 版: 元の話者の表を単位ごとの観測数 n_u で母集団の平均へ縮める T_src' = (n_u·T_src + κ·T_pop) / (n_u + κ)(κ = --kappas フレーム)。条件 DSH<κ>・DELTA(κ 0)・DELTA_POP(κ ∞)・TAB。
D30: 単位ごとの話者差を元の声の包絡に足す(学習なし)。女声 → 女声(評価プール・元の声 = 次の目標の話者の自然な台詞・D24 と同じ組)。
包絡だけを替える(f0 = 元の声の有声 log f0 を目標の中央値へ移動・c0 = 元の声の実の c0・周期性 = 目標の平均):
  SRCENV = 元の声の包絡のまま・TAB = 柔らかい引き Σ p(u) T[u](β 1・平滑なし = 製品の混ぜ 0)・M35 = 0.65·TAB + 0.35·元の包絡(製品のつまみ)・
  DELTA = 元の包絡 + Σ p(u)(T_tgt[u] − T_src[u])(T_src = 元の話者の参照 20s の表 = 操作者の登録)・
  DELTA_H = 元の包絡 + T_tgt[u] − T_src[u](硬い単位)・DELTA_POP = 元の包絡 + Σ p(u)(T_tgt[u] − 母集団平均の表[u])(操作者の登録なし)。
出力 = d4b の形式の wav(cen・pNNNN_tNNN_<cond>・SRC)。CER は d23_cer.py、同一性は d4b_lda_verifier.py で測る。
    uv run python d30_delta.py --out_dir <scratchpad>/d30 --n_tg 60
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
    ap.add_argument("--kappas", type=float, nargs="*", default=[])
    a = ap.parse_args()
    import c1_content as CC
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
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
    allT = np.stack([t["T"].astype(np.float64) for t in pool])
    if a.kappas:
        refs = P.load_refs(150, 3)
        cnt = np.stack([np.bincount(r["u"].astype(int), minlength=allT.shape[2]) for r in refs]).astype(np.float64)

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

    pi = 0
    for ti in range(min(a.n_tg, S)):
        t = pool[ti]
        si = (ti + 1) % S
        T = t["T"].astype(np.float64)
        Ts = pool[si]["T"].astype(np.float64)
        keep = np.ones(S, bool); keep[[ti, si]] = False
        Tp = allT[keep].mean(0)
        sf.write(od / f"cen_{ti:03d}.wav", t["cen"], 48000)
        for su in pool[si]["utts"]:
            u = su["u"].astype(int)
            n = len(u)
            p = post(su["x"])[:, :n]
            vo = su["f0"] > 0
            lfs = np.log(np.maximum(su["f0"], 1.0))
            med = float(np.median(lfs[vo])) if vo.any() else t["mu"]
            f0o = np.where(vo, np.exp(lfs - med + t["mu"]), 0.0)
            perc = np.repeat(t["per_mean"][:, None], n, 1)
            er = su["er"].astype(np.float64)
            soft = T @ p
            if a.kappas:
                envs = {"TAB": soft, "DELTA": er + (T - Ts) @ p, "DELTA_POP": er + (T - Tp) @ p}
                for kp in a.kappas:
                    nu = cnt[si][None]
                    envs[f"DSH{kp:g}"] = er + (T - (nu * Ts + kp * Tp) / (nu + kp)) @ p
            else:
                envs = {"SRCENV": er, "TAB": soft, "M35": 0.65 * soft + 0.35 * er, "DELTA": er + (T - Ts) @ p,
                        "DELTA_H": er + (T - Ts)[:, u], "DELTA_POP": er + (T - Tp) @ p}
            for c, ev in envs.items():
                y = render(su["c0"], ev, f0o, perc)
                sf.write(od / f"p{pi:04d}_t{ti:03d}_{c}.wav", y, 48000)
            sf.write(od / f"p{pi:04d}_t{ti:03d}_SRC.wav", su["x"], 48000)
            pi += 1
        print("render", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
