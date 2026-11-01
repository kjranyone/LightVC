"""因果の走行の正規化(包絡から走行平均を引く・走行の標準偏差で割る)が、包絡の話者の手がかりをどれだけ消すか。同じ探り。"""
from __future__ import annotations
import sys
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import a3_leak_base as b
def running(x: np.ndarray, tau_s: float, std: bool) -> np.ndarray:
    a = 1.0 / (tau_s * b.SR / b.HOP)
    m = np.zeros(x.shape[1]); v = np.ones(x.shape[1]); out = np.empty_like(x)
    m[:] = x[:20].mean(0) * 0
    for t in range(len(x)):
        m += a * (x[t] - m) if t else 0
        if t == 0: m = x[0].copy()
        v += a * ((x[t] - m) ** 2 - v)
        out[t] = (x[t] - m) / (np.sqrt(v) + 1e-3) if std else x[t] - m
    return out
for tau in (1.0, 3.0):
    for std in (False, True):
        b.report(f"run_tau{tau}_std{std}", {k: running(v, tau, std) for k, v in b.env.items()})
