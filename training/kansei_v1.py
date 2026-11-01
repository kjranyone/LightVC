"""官能指数 v1(D678): 凍結の WavLM-large の 0.5 s の窓の本物 / 合成の線形の検出器。値 = 有音の窓の決定値の平均 − 閾 τ(> 0 = 合成側 = 不合格・本物の calib の 95% が ≤ 0)。
fold = 学習から外した族(world / gen / render)か none(全部)。新しい方式の評価は、その方式の族を外した fold で読む。
    from kansei_v1 import KanseiV1; k = KanseiV1("render"); k.margin(["a.wav", ...])
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from d678_kansei_det import OUT, embed, wavlm, windows


class KanseiV1:
    def __init__(self, fold: str = "none"):
        m = pickle.loads((OUT / f"model_{fold}.pkl").read_bytes())
        self.sc, self.clf, self.tau = m["scaler"], m["clf"], m["tau"]
        self.net = wavlm()

    def margin(self, paths: list) -> list:
        out = []
        for p in paths:
            W = windows(str(p))
            if len(W) == 0:
                out.append(float("nan"))
                continue
            F = embed(self.net, W).astype(np.float32)
            out.append(float(self.clf.decision_function(self.sc.transform(F)).mean()) - self.tau)
        return out
