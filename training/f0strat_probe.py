"""f0 で層別した写し合成の評価(学習なし): held21 を f0 中央値 < 450 / ≥ 490 の 2 群に分け、logmel・PESQ_h・変調の線を群ごとに平均。
    uv run python f0strat_probe.py --ckpt <ema.pt> --env_smooth 0.25 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
import rvoc as R
import train_rvoc as TR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env_smooth", type=float, default=None, help="省略時は ckpt の env_smooth(無ければ 0)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    g = R.RVoc(ch=st["cfg"]["ch"], kernels=tuple(st["cfg"]["kernels"]), dils=tuple(st["cfg"]["dils"]), d_cond=st["cfg"]["d_cond"]).to(dev)
    g.load_state_dict(st["ema"])
    g.eval()
    fr = TR.Front(st.get("cond", "pae"), TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
    rows = []
    with torch.no_grad():
        for it in TR.held(dev):
            v = it["f0_h"][it["f0_h"] > 0]
            m = E.metrics(TR.render(g, fr, it, it["f0_h"], dev), it["x"], N.DELAY, dev)
            rows.append((float(np.median(v)), m))
    rep = {}
    for name, sel in (("low_lt450", lambda f: f < 450), ("high_ge490", lambda f: f >= 490)):
        r = [m for f, m in rows if sel(f)]
        rep[name] = {"n": len(r), **{k: round(float(np.nanmean([x[k] for x in r])), 4) for k in ("logmel", "pesq", "am_db", "hf_db")}}
    print(json.dumps(rep))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
