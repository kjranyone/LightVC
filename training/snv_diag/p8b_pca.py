"""P8b の固定の PCA(results/snv_p8b/prereg.yaml): 学習の声の池(split の train[::4])の登録(t00〜t02 の有声のフレーム・話者ごとに平均を引く・6 フレームおき)の凍結した c の主成分。
p8a_draw_voice.py の w64 / wst と同じ手順で作り、演奏の正規化 wst(train_snv_p3.py --play wst)と探りが同じ行列を使うよう保存する。
vsel=f(段 P70 step b・診断)= 池から男声(em*)を除く。
    cd training && uv run --no-sync python snv_diag/p8b_pca.py [cpc=results/snv_p2b_cpc/last.pt] [out=results/snv_p8b/pca_c_P2b.npy] [vsel=f]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_p1 as T
import train_snv_p2_cpc as CP
from snv_diag import p8a_draw_voice as DV


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ck = opt.get("cpc", "results/snv_p2b_cpc/last.pt")
    out = DV.ROOT / opt.get("out", "results/snv_p8b/pca_c_P2b.npy")
    st = torch.load(DV.ROOT / ck, map_location="cpu", weights_only=False)
    cm = CP.CPC().to(DV.DEV).eval()
    cm.load_state_dict(st["cpc"])
    pool = [v for v in json.loads((DV.ROOT / "results/snv_p1_n/split.json").read_text())["train"] if opt.get("vsel") != "f" or not v.startswith("em")][::4]
    xs = []
    for v in pool:
        cs = []
        for j in (0, 1, 2):
            u = T.load_utt(T.FEAT / f"e1/{v}/t0{j}.npz")
            with torch.no_grad():
                c = cm.encode(torch.from_numpy(CP.x_of(u, st["sig"]))[None].to(DV.DEV))[1][0].cpu().numpy().T
            cs.append(c[u["v"] > 0])
        x = np.concatenate(cs)
        xs.append((x - x.mean(0))[::DV.SUB])
    x = np.concatenate(xs)
    w, q = np.linalg.eigh(x.T @ x / len(x))
    P = np.ascontiguousarray(q[:, ::-1]).astype(np.float32)
    w = w[::-1]
    print("var64", round(float(w[:64].sum() / w.sum()), 3), "var32", round(float(w[:32].sum() / w.sum()), 3), flush=True)
    np.save(out, P)
    print("wrote", out, P.shape, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
