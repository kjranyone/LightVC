"""製品の経路(Rust artic・柔らかい引き β・伸縮 α 固定・色づけ・残差 γ)の包絡の行を Python で作る(D217・学習の土台)。
入力 = Rust の LIGHTVC_DUMP_FRONT(フレームごとに C1 の logit [k] と F3 の包絡 [24])と、目標・操作者の登録(artic-enroll の enroll.bin・enroll.json)。
出力 = 条件の行 1..24(Rust の cond[1..25] と同じ単位・較正 B なし)。crates/lightvc-core/src/artic_vc.rs の process_block・set_operator_warp・set_color と enroll.rs の warp_matrix と同じ式。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
COLOR_SHRINK = 0.1


def load_enroll(d: Path) -> dict:
    j = json.loads((Path(d) / "enroll.json").read_text())
    D, K = int(j["D"]), int(j["K"])
    t = np.fromfile(Path(d) / "enroll.bin", "<f4").reshape(D, K)
    rc = np.asarray(j["res_cov"], np.float32).reshape(D, D) if len(j.get("res_cov") or []) == D * D else None
    return {"table": t, "res_cov": rc, "mu": float(j["mu"]), "d": D, "k": K}


def warp_matrix(alpha: float, consts: Path = ROOT / "results/enroll_export") -> np.ndarray:
    man = json.loads((consts / "manifest.json").read_text())
    w = np.fromfile(consts / "weights.bin", "<f4")
    o, sh = man["tensors"]["dct"]; dct = w[o: o + sh[0] * sh[1]].reshape(sh)
    o, sh = man["tensors"]["mel.fb"]; fb = w[o: o + sh[0] * sh[1]].reshape(sh)
    nm, nb = fb.shape
    sr = man["const"]["SR"]
    fc = fb.argmax(1).astype(np.float64) * (sr / 2.0) / (nb - 1)
    im = np.zeros((nm, nm))
    for i in range(nm):
        src = min(max(fc[i] / np.float64(np.float32(alpha)), fc[0]), fc[-1])
        j = max(int(np.searchsorted(fc, src, side="left")) - 1, 0)
        j = min(j, nm - 2)
        ww = min(max((src - fc[j]) / max(fc[j + 1] - fc[j], 1e-9), 0.0), 1.0)
        im[i, j] += 1.0 - ww
        im[i, j + 1] += ww
    B = dct[1:25].astype(np.float64)
    return (B @ im @ B.T).astype(np.float32)


def sym_pow(a: np.ndarray, p: float) -> np.ndarray:
    ev, V = np.linalg.eigh(a)
    return (V * np.power(np.maximum(ev, 1e-300), p)) @ V.T


def color(en: dict, op: dict, W: np.ndarray) -> np.ndarray:
    Wd = W.astype(np.float64)
    co = Wd @ op["res_cov"].astype(np.float64) @ Wd.T
    ct = en["res_cov"].astype(np.float64).copy()
    d = ct.shape[0]
    for c in (ct, co):
        c[np.diag_indices(d)] += COLOR_SHRINK * np.trace(c) / d
    return (sym_pow(ct, 0.5) @ sym_pow(co, -0.5)).astype(np.float32)


def read_front(path: Path, k: int, d: int = 24) -> tuple:
    f = np.fromfile(path, "<f4").reshape(-1, k + d)
    return f[:, :k], f[:, k:]


def base_env(logit: np.ndarray, e: np.ndarray, en: dict, op: dict, W: np.ndarray, M: np.ndarray, gamma: float = 0.3, beta: float = 1.0) -> np.ndarray:
    q = np.exp((logit - logit.max(1, keepdims=True)) * np.float32(beta)).astype(np.float32)
    z = q.sum(1, keepdims=True)
    tab = (q @ en["table"].T) / z / 10.0
    wt = W @ op["table"]
    eo = (q @ wt.T) / z / 10.0
    v = e @ W.T - eo
    dv = v @ M.T
    return (tab + np.float32(gamma) * dv).astype(np.float32)


class PBase:
    def __init__(self, root: Path, alpha: float = 1.3, gamma: float = 0.3, beta: float = 1.0):
        self.root, self.W, self.g, self.b = Path(root), warp_matrix(alpha), gamma, beta
        self.en: dict = {}; self.op: dict = {}; self.M: dict = {}

    def env(self, stem: str, spk: str, item: int, A: str) -> np.ndarray:
        ok = f"{spk}_{1 - item % 2}"
        if A not in self.en:
            self.en[A] = load_enroll(self.root / "en" / A)
        if ok not in self.op:
            self.op[ok] = load_enroll(self.root / "op" / ok)
        en, op = self.en[A], self.op[ok]
        if (A, ok) not in self.M:
            self.M[(A, ok)] = color(en, op, self.W)
        lg, e = read_front(self.root / "front" / f"{stem}.f32", en["k"])
        return base_env(lg, e, en, op, self.W, self.M[(A, ok)], self.g, self.b)
