"""D111(学習なし): D109 v2 で変換が落ちる『shimmer』(隣接 10ms フレームの音量の揺れ・ガビ軸)の出どころ。D98: 女声の写し合成で F3 の包絡に替えると 0.61 → 0.76・F1 / F2 / 周期性は無関係。
変換の条件は男声から読む(F2 の c0・F3 の包絡)。仮説 = 低い f0(周期 8〜10ms)の男声では因果の推定器の出力にフレーム周期の揺れが乗る(教師の CheapTrick はピッチ適応窓で揺れない)。
元の男声(D56 の 60 文)と目標の女声(D37 の cen 30)で、教師(CheapTrick の c0・c1..c24)と製品の推定器(F2 + 因果 3 平均・F3)の軌道の
  rough = 有声の隣接フレームの |Δ| の中央値(c0 は c0n の単位・包絡は c_k/10 の次元平均)・hf = 軌道の 25Hz 超の変調のパワーの割合(有声区間・平均を引く)
を比べる。出力 = results/conv_p0/d111.json。
    uv run python d111_cond_ripple.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import train_c3 as C3
    import train_rvoc as TR
    from d6_unit_cov import load48
    S = Path(a.s)
    dev = "cuda"
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    mfront = CC.MelFront().to(dev)
    fps = N.SR / N.HOP

    def rough(z: np.ndarray, v: np.ndarray) -> float:
        b = v[1:] & v[:-1]
        d = np.abs(np.diff(z, axis=-1))
        d = d.mean(0) if d.ndim == 2 else d
        return float(np.median(d[b])) if b.any() else float("nan")

    def hfrac(z: np.ndarray, v: np.ndarray) -> float:
        zz = z if z.ndim == 2 else z[None]
        num = den = 0.0
        t = 0
        while t < len(v):
            if not v[t]:
                t += 1
                continue
            u = t
            while u < len(v) and v[u]:
                u += 1
            if u - t >= 32:
                seg = zz[:, t:u] - zz[:, t:u].mean(1, keepdims=True)
                P = np.abs(np.fft.rfft(seg * np.hanning(u - t), axis=1)) ** 2
                fq = np.fft.rfftfreq(u - t, 1 / fps)
                num += P[:, fq > 25].sum()
                den += P[:, 1:].sum()
            t = u
        return float(num / den) if den > 0 else float("nan")

    sets = {"male_src": sorted((S / "d56").glob("p*_SRC.wav")), "female_cen": sorted((S / "d37").glob("cen_*.wav"))}
    rep = {}
    for nm, fs in sets.items():
        acc: dict = {}
        for p in fs:
            x = load48(p).astype(np.float32)[: 8 * N.SR]
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            f0, _ = H.teacher_f0(x, n)
            f0 = f0[:n].astype(np.float64)
            xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
            e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
            env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
            c0 = (env[0] - PA.C0_SIL) / 10
            with torch.no_grad():
                m = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
                r = f2(m)[:, None]
                raw = r[0, 0].cpu().numpy()[:n].astype(np.float64)
                avg3 = C3.causal_avg(r, 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
                e3 = f3(m)[0].cpu().numpy()[1:25, :n].astype(np.float64)
            k = min(n, len(avg3), e3.shape[1])
            v = f0[:k] > 0
            vals = {"c0_teacher": c0[:k], "c0_f2raw": raw[:k], "c0_f2avg3": avg3[:k], "env_teacher": env[1:25, :k] / 10, "env_f3": e3[:, :k]}
            for kk, z in vals.items():
                acc.setdefault(f"{kk}_rough", []).append(rough(z, v))
                acc.setdefault(f"{kk}_hf", []).append(hfrac(z, v))
            acc.setdefault("f0_med", []).append(float(np.median(f0[f0 > 0])) if (f0 > 0).any() else float("nan"))
        rep[nm] = {k: round(float(np.nanmedian(v)), 4) for k, v in acc.items()}
        print(nm, json.dumps(rep[nm]), flush=True)
    (ROOT / "results/conv_p0/d111.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
