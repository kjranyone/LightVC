"""段 P75 step a0(results/snv_p75/prereg.yaml・学習なし・ASR なし): 符号どうしの読みの食い違い。
源の有声のフレームごとに、登録(先頭 tn 本)の有声のフレームから、符号の文脈 c で最も近いフレームを選ぶ(p18_nnc.code の c)。
d(a, b) = 2 つの符号の選びが、同じ登録の発話の中で win フレーム以内に無い率(源ごとの率の平均・話者で束ねた 95 % の区間)。
    cd training && SNV_DEV=cuda uv run --no-sync python snv_diag/p75_div.py [sets=cv] [codes=snv_p2b_cpc,snv_p24_xcpc,snv_p30_dcpc,snv_p25_xrcpc] [win=8] [out=results/snv_p75/div.json]
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_align as AL
import p18_nnc as NN
import snv_p1_eval as EV
import train_snv_p3 as P3
from p74_concord import boot

ROOT = EV.ROOT
DEV = EV.DEV


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    names, codes = opt.get("sets", "cv").split(","), opt.get("codes", "snv_p2b_cpc,snv_p24_xcpc,snv_p30_dcpc,snv_p25_xrcpc").split(",")
    tn, win = int(opt.get("tn", 3)), int(opt.get("win", 8))
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"]][:64] for nm, its in AL.build_sets(names, 64, split).items()}
    pick: dict = {}
    for cd in codes:
        Pl = P3.Play(ROOT / "results" / cd / "last.pt").to(DEV).eval()
        for nm, its in sets.items():
            for it in its:
                us = [it["u"]] + it["enr"][:tn]
                M = [u["v"] > 0.5 for u in us]
                xs = [c[torch.from_numpy(m).to(DEV)] for c, m in zip(NN.code(Pl, us, list(range(1, len(us))), "c"), M)]
                pos = np.concatenate([np.flatnonzero(m) + 100000 * i for i, m in enumerate(M[1:])])
                pick[(cd, nm, it["src"])] = pos[AL.nn(xs[0], torch.cat(xs[1:])).cpu().numpy()]
        del Pl
    res: dict = {"prereg": "results/snv_p75/prereg.yaml(step_a0)", "win": win, "tn": tn, "d": {}}
    for nm, its in sets.items():
        g = np.array([it["spk"] for it in its])
        for a, b in itertools.combinations(codes, 2):
            d = np.array([float((np.abs(pick[(a, nm, it["src"])] - pick[(b, nm, it["src"])]) > win).mean()) for it in its])
            res["d"][f"{nm}|{a}|{b}"] = boot(d, g)
    (ROOT / opt.get("out", "results/snv_p75/div.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
