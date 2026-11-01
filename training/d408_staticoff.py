"""D408: 自己再構成の区間(2.8 s)の実際の包絡と表の引き(Σ p T / 10)の、区間の平均のずれ(静的なずれ = 録音・収録日の違い)の大きさと、c0n・事後から読めるか。
D407: 表への追従は最初の微調整(ppgvoc1)で失われた・遅い揺らぎの潜在は区間の平均を除く = 静的なずれはどの入力にも与えられず、生成器は c0n・事後から推すしかない(仮説)。
学習と同じ DS・front・C1 で n 区間を作り(CPU)、静的なずれ m = mean_t(env − base)[24] の大きさ(行ごとの標準偏差・包絡の行の時間の標準偏差との比)と、
m を c0n の区間の平均と事後の区間の平均からリッジで予測した R²(区間でまとめた交差検証)を出す。
    uv run python d408_staticoff.py --n 400 --out ../results/conv_p0/d408.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import pae as PA
import train_ppgvoc as P
import train_rvoc as TR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    TR.SEG = 33600 * 4
    TR.T = TR.SEG // 240
    P.TAB = P.ROOT / "data/rvoc_tables_real"
    rows, spks, _ = P.load_rows({h[0] for h in P.e2e_held()}, True)
    spk_idx = {s: i for i, s in enumerate(spks)}
    import c1_content as CC
    import train_c1 as T1
    st1 = torch.load(P.ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"]))
    c1n.load_state_dict(st1["net"])
    c1n.eval()
    mfc = CC.MelFront()
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None))
    ds = iter(TR.DS(rows, rows, 408, spk_idx=spk_idx, ctx=T1.PRE))
    M, C0, PM, SD, SPK = [], [], [], [], []
    with torch.no_grad():
        for i in range(a.n):
            xa, f0, env, xl, si = next(ds)
            xa, f0, env, xl = xa[None], f0[None], env[None], xl[None]
            ct = front(xa, f0, env)
            p = c1n(mfc(torch.cat([xl, xa[:, PA.A:PA.A + TR.SEG]], 1)))[..., -TR.T:].softmax(1)
            s = spks[int(si)]
            T = torch.from_numpy(np.load(P.TAB / f"{s}.npz")["T"]).float()
            base = torch.einsum("dk,bkt->bdt", T, p) / 10
            r = ct[:, 1:25] - base
            v = ct[:, 26] > 0.5
            if v.sum() < 50:
                continue
            M.append(r[0][:, v[0]].mean(-1).numpy())
            SD.append(r[0][:, v[0]].std(-1).numpy())
            C0.append(float(ct[0, 0, v[0]].mean()))
            PM.append(p[0][:, v[0]].mean(-1).numpy())
            SPK.append(s)
    M, SD, C0, PM = np.array(M), np.array(SD), np.array(C0), np.array(PM)
    rep = {"n": len(M), "static_offset_sd_by_row": np.round(M.std(0), 3).tolist(), "static_offset_mean_by_row": np.round(M.mean(0), 3).tolist(),
           "within_clip_sd_by_row": np.round(SD.mean(0), 3).tolist(), "ratio_static_to_within_rows1_4": np.round(M.std(0)[:4] / SD.mean(0)[:4], 2).tolist()}
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    g = np.array([hash(s) % 10 ** 6 for s in SPK])
    for name, X in (("c0n_mean", C0[:, None]), ("posterior_mean", PM), ("c0n+posterior", np.concatenate([C0[:, None], PM], 1))):
        pr = np.zeros_like(M)
        for tr, te in GroupKFold(5).split(X, M, g):
            pr[te] = Ridge(alpha=1.0).fit(X[tr], M[tr]).predict(X[te])
        r2 = 1 - ((M - pr) ** 2).sum(0) / ((M - M.mean(0)) ** 2).sum(0)
        rep[f"R2_{name}_rows1_6"] = np.round(r2[:6], 3).tolist()
        print(name, "R² (rows 1..6)", rep[f"R2_{name}_rows1_6"], flush=True)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith("R2")}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
