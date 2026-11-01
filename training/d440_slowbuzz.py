"""D440: ガビの線は推論の遅い音色の揺らぎ(乱数・σ)から来るか。学習の自己再構成では本物の揺らぎ(確率 0.7)+ 静的なずれを与え fline ≈ −0.2 dB・評価(乱数 σ)では 2.4 dB。
devff の Rust の書き出しを D2 で σ = 0 / 0.5 / 1.0 / 1.5 / 2.0 で描画し fline / amline の中央値を比べる。
    uv run python d440_slowbuzz.py --ckpt ../results/diag_ppgvoc41/product_002500_m0.pt --out ../results/conv_p0/d440.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d436_buzzcarrier import items_of
from eval_nvoc import amline
from fsync import fline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    ed = P.H / "e2e" / "eval_devff"
    its = items_of("devff")[: a.n]
    rep = {}
    for sg in a.sigmas:
        st["slow_sigma"] = sg
        P.slow_from_ckpt(st)
        if sg == 0:
            P.SLOW["on"] = False
        fl, am = [], []
        for k, ti in its:
            c, lg = P.load_dump(ed, k)
            y = P.render(g, c, lg, P.load_enroll(P.H / "devff" / "d85_work" / f"en_{ti:03d}"), dev)[: 8 * N.SR]
            fl.append(float(fline(torch.from_numpy(y)[None])[0]))
            am.append(amline(y))
        rep[f"sigma_{sg}"] = {"fline": round(float(np.median(fl)), 3), "amline": round(float(np.median(am)), 3)}
        print(sg, rep[f"sigma_{sg}"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
