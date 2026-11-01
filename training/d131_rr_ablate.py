"""D131(学習なし・R の寄与の切り分け): train_rr.py の重みで、検証の固定 96 組(評価外の女声・検証用の男声)を
  RR = そのまま・SWAP = 注意で読む記憶だけを別の評価外の女声の参照に差し替え(表の引きは本物の目標のまま = 記憶の経路だけの寄与)・UNIF = 注意を参照全体の一様に・TAB = 表(学習前)
で測る。SWAP で誤差が RR から TAB 側へ戻るほど、R は目標固有の情報を参照から読んでいる(戻らなければ改善は操作者側の整え)。
出力 = results/<tag>/ablate.json。
    uv run python d131_rr_ablate.py --tag diag_rr1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = RRm.Store()
    held = RRm.held_females()
    fems = sorted(s for s in {s for (c, s, _) in st.f if c == "fem_neutral"} if all(("fem_neutral", s, f"t{t:02d}_neutral") in st.f for t in range(10)))
    va_f = [s for s in fems if s in held]
    males = sorted(s for s in {s for (c, s, _) in st.f if c == "male_par"} if all(("male_par", s, f"f{t:02d}") in st.f for t in range(10)))
    va_m = [m for m in males if m in RRm.VAL_MALES and m in st.stats]
    vds = RRm.Pairs(st, va_f, va_m, 96, 0.5 if va_m else 0.0, 1, fixed=True)
    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    net = RRm.RR(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk")}, **{k_: tuple(ck["cfg"][k_]) for k_ in ("qd", "kd") if k_ in ck["cfg"]}).to(dev).eval()
    net.load_state_dict(ck["net"])
    items = [vds[i] for i in range(len(vds))]
    rows = []
    with torch.no_grad():
        for i, it in enumerate(items):
            sw = items[(i + 7) % len(items)]
            if sw["A"] == it["A"]:
                sw = items[(i + 11) % len(items)]
            b = RRm.collate([it]); bs = RRm.collate([{**it, "rp": sw["rp"], "rv": sw["rv"]}])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            bs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in bs.items()}
            v = b["yv"][0] & b["tm"][0]
            y = b["y"][0][:, v]

            def l1(e: torch.Tensor) -> float:
                return float((e[0][:, v] - y).abs().mean())

            e_rr, _, _ = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
            e_sw, _, _ = net(bs["q"], bs["tab"], net.memory(bs["rp"], bs["rv"]), bs["rm"])
            k, vv = net.memory(b["rp"], b["rv"])
            ku = torch.zeros_like(k)
            e_un, _, _ = net(b["q"], b["tab"], (ku, vv), b["rm"])
            rows.append({"male": bool(it["male"]), "RR": l1(e_rr), "SWAP": l1(e_sw), "UNIF": l1(e_un), "TAB": l1(b["tab"])})
    out = {}
    for kind in (False, True):
        r = [x for x in rows if x["male"] == kind]
        if r:
            out["mf" if kind else "ff"] = {c: round(float(np.median([x[c] for x in r])), 4) for c in ("RR", "SWAP", "UNIF", "TAB")} | {"n": len(r)}
    print(json.dumps(out), flush=True)
    (ROOT / "results" / a.tag / "ablate.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
