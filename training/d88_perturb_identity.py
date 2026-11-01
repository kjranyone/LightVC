"""D88(学習なし): 動きの変換器 G(入力側で話者をかき混ぜる自己再構成)の前提検査。
入力の動き res に摂動 P(周波数伸縮 W_α・α ∈ [1.0, 1.4] 一様 + 別の話者の共分散への色づけ M = C_他^½ (W C_本人 Wᵀ)^−½・縮約 λ 0.1 = Rust --color と同じ)をかけたとき:
  (1) 動きの共分散からの話者の当て(前半の共分散 → 後半の共分散の Förstner 最近傍・24 人・偶然 1/24)が偶然の水準まで下がるか(話者の情報が消えるか)。
  (2) 摂動後の動きと元の動きの次元ごとの相関(時刻の構造が残るか = G が実現を復元できるか)。
女声 TTS 24 人 × neutral 10 文(前半 5 / 後半 5)・res = D74 と同じ。出力 = results/conv_p0/d88.json。
    uv run python d88_perturb_identity.py --n 24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48
from d74_res_cov import forstner

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/female_tts_corpus"
G: dict = {}


def init() -> None:
    from d81_dyn_descriptor import init as i0
    i0()
    from d81_dyn_descriptor import G as G0
    G.update(G0)


def res_files(spk: str) -> list:
    import conv_c0 as C0
    from d81_dyn_descriptor import analyse
    out = []
    for i in range(10):
        a = analyse(load48(CORPUS / spk / f"t{i:02d}_neutral.wav"))
        r = a["env"] - C0.table(a["env"], a["u"], G["Cb"].shape[0], G["Cb"]) @ a["p"]
        out.append((r - r.mean(1, keepdims=True))[:, a["v"]])
    return out


def wmat(alpha: float) -> np.ndarray:
    import nvoc as N
    import zsvc as Z
    Dm = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)
    fb = N.mel_fb().numpy()
    fc = fb.argmax(1) * (N.SR / 2) / (fb.shape[1] - 1)
    B = Dm[1:25]
    I = np.zeros((N.N_MEL, N.N_MEL))
    src = np.clip(fc / alpha, fc[0], fc[-1])
    jj = np.clip(np.searchsorted(fc, src) - 1, 0, N.N_MEL - 2)
    ww = np.clip((src - fc[jj]) / np.maximum(fc[jj + 1] - fc[jj], 1e-9), 0, 1)
    I[np.arange(N.N_MEL), jj] = 1 - ww
    I[np.arange(N.N_MEL), jj + 1] += ww
    return B @ I @ B.T


def msqrt(C: np.ndarray, p: float) -> np.ndarray:
    C = C + 0.1 * np.trace(C) / 24 * np.eye(24)
    w, V = np.linalg.eigh(C)
    return (V * np.maximum(w, 1e-12) ** p) @ V.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    spks = sorted(d.name for d in CORPUS.iterdir() if d.is_dir() and all((d / f"t{i:02d}_neutral.wav").exists() for i in range(10)))
    random.Random(88).shuffle(spks)
    spks = spks[: a.n]
    with Pool(a.workers, initializer=init) as pool:
        R = pool.map(res_files, spks)
    cov = lambda X: X @ X.T / X.shape[1] + 1e-6 * np.eye(24)
    H1 = [cov(np.concatenate(r[:5], 1)) for r in R]
    rng = np.random.default_rng(88)
    res = {}
    def mpow(M: np.ndarray, t: float) -> np.ndarray:
        w, V = np.linalg.eig(M)
        return np.real((V * (w.astype(complex) ** t)) @ np.linalg.inv(V))

    for name in ("none", "warp_color", "color", "color_half", "diag", "warp_diag"):
        H2, corr, ecorr = [], [], []
        for i, r in enumerate(R):
            X = np.concatenate(r[5:], 1)
            j = int(rng.choice([q for q in range(len(R)) if q != i]))
            W = wmat(rng.uniform(1.0, 1.4)) if name.startswith("warp") else np.eye(24)
            Cs = W @ cov(X) @ W.T
            if name == "none":
                M = np.eye(24)
            elif name in ("warp_color", "color"):
                M = msqrt(H1[j], 0.5) @ msqrt(Cs, -0.5)
            elif name == "color_half":
                M = np.real(mpow(msqrt(H1[j], 0.5) @ msqrt(Cs, -0.5), 0.5))
            else:
                M = np.diag(np.sqrt(np.diag(H1[j]) / np.maximum(np.diag(Cs), 1e-9)))
            Y = M @ W @ X
            corr.append(float(np.mean([np.corrcoef(X[d], Y[d])[0, 1] for d in range(24)])))
            ex, ey = np.sqrt((X ** 2).sum(0)), np.sqrt((Y ** 2).sum(0))
            ecorr.append(float(np.corrcoef(ex, ey)[0, 1]))
            H2.append(cov(Y))
        D = np.array([[forstner(H1[p], H2[q]) for p in range(len(R))] for q in range(len(R))])
        hit = float(np.mean(D.argmin(1) == np.arange(len(R))))
        res[name] = {"speaker_id_top1": round(hit, 3), "chance": round(1 / len(R), 3), "corr_orig_vs_perturbed": round(float(np.mean(corr)), 3), "energy_timing_corr": round(float(np.mean(ecorr)), 3)}
        print(name, json.dumps(res[name]), flush=True)
    (ROOT / "results/conv_p0/d88.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
