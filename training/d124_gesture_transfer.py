"""D124 = 査読(10-11)の T1(学習なし・診断専用): 構音 VC の標準の前提『操作者の(話者を正規化した)身振り × 目標の形態』は成り立つか = 別の話者の単位内の動きを、
登録の統計だけから作る写像で目標の動きへ写せるか。女声 TTS コーパス(全話者が同じ 10 文)・目標 A と操作者 B(別の女声)が同じ文 tK_neutral を読む 24 組。
動き res_X = CheapTrick 包絡 c1..c24 − Σp T_X[u](T_X = X の登録 = 同じ文 tK を除く 9 文の単位平均・製品と同じ定義)。B の動きを DTW(診断専用・対数メルのコサイン距離・librosa)で A の時間へ並べ、写像:
  ID = 恒等・GLOB = 大域の色づけ C_A^½ C_B^−½(各自の登録の有声の res の共分散・製品の --color と同じ)・GRP<G> = 単位を G 群(コードブックの k-means)に束ね群ごとの色づけ(群の共分散は大域へ縮約 λ = 200 フレーム)・
  ORACLE = 同じ組の 9 文のパラレル(DTW)で当てた最小二乗の線形写像(推論では使えない上限)・ORACLE_GRP = 群ごとの最小二乗(上限)。
床 = A 自身の同じ文の別の話し方(soft)の動きを DTW で並べたもの(本人の別の実現)。指標 = A の有声フレームで次元ごとの相関の平均(と B の動きの漏れの探針: 写した動きと B 本人の動きの相関)。
判定(査読の線): 推論で使える写像(GLOB / GRP)で ≥ 0.35 → 操作者の身振りは使える(標準の前提が成り立つ)・≤ 0.22 → 身振りそのものが話者固有(製品意図の判断が要る)。
出力 = results/conv_p0/d124.json。
    uv run python d124_gesture_transfer.py --n 24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48
from d70_xdyn import CORPUS

ROOT = Path(__file__).resolve().parent.parent


def msqrt(C: np.ndarray, p: float, eps: float = 1e-3) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps * w.max())
    return (V * w ** p) @ V.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--groups", type=int, nargs="*", default=[8, 16])
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    dev = "cuda"
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    grp = {G: KMeans(G, n_init=4, random_state=0).fit_predict(Cb) for G in a.groups}

    cache: dict = {}

    def analyse(p_: Path) -> dict:
        if p_ in cache:
            return cache[p_]
        x = load48(p_).astype(np.float32)
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[1:25, :n].astype(np.float64)
        with torch.no_grad():
            mf = mfront(torch.from_numpy(T1.prime(x))[None].to(dev))
            p = c1(mf)[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
            lm = mf[0, :, -n:].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        lm = np.pad(lm, ((0, 0), (0, max(0, n - lm.shape[1]))), mode="edge")[:, :n]
        cache[p_] = {"n": n, "f0": f0, "env": env, "p": p, "u": p.argmax(0), "lm": lm}
        return cache[p_]

    def enroll(spk: str, skip: int) -> dict:
        rs = [analyse(CORPUS / spk / f"t{i:02d}_neutral.wav") for i in range(10) if i != skip]
        E = np.concatenate([r["env"] for r in rs], 1); U = np.concatenate([r["u"] for r in rs]); P = np.concatenate([r["p"] for r in rs], 1)
        V = np.concatenate([r["f0"] > 0 for r in rs])
        T = C0.table(E, U, K, Cb)
        R = (E - T @ P)[:, V]
        Rc = R - R.mean(1, keepdims=True)
        Cg = Rc @ Rc.T / Rc.shape[1]
        out = {"T": T, "C": Cg, "Cgrp": {}}
        for G, lab in grp.items():
            gU = lab[U[V]]
            cs = []
            for g in range(G):
                m = gU == g
                ng = int(m.sum())
                Cl = Rc[:, m] @ Rc[:, m].T if ng else np.zeros_like(Cg)
                cs.append((Cl + 200 * Cg) / (ng + 200))
            out["Cgrp"][G] = cs
        return out

    def res(r: dict, T: np.ndarray) -> np.ndarray:
        return r["env"] - T @ r["p"]

    def align(rA: dict, rB: dict) -> np.ndarray:
        import librosa
        an = rA["lm"] / (np.linalg.norm(rA["lm"], axis=0, keepdims=True) + 1e-9)
        bn = rB["lm"] / (np.linalg.norm(rB["lm"], axis=0, keepdims=True) + 1e-9)
        _, wp = librosa.sequence.dtw(C=1 - an.T @ bn)
        j = np.zeros(rA["n"], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        return j

    def corr(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> float:
        return float(np.mean([np.corrcoef(x[d, v], y[d, v])[0, 1] for d in range(x.shape[0])]))

    spks = sorted(d.name for d in CORPUS.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in ("neutral", "soft")))
    random.Random(124).shuffle(spks)
    rows = []
    for k in range(a.n):
        sa, sb = spks[2 * k], spks[2 * k + 1]
        tk = k % 10
        EA, EB = enroll(sa, tk), enroll(sb, tk)
        A = analyse(CORPUS / sa / f"t{tk:02d}_neutral.wav")
        B = analyse(CORPUS / sb / f"t{tk:02d}_neutral.wav")
        As = analyse(CORPUS / sa / f"t{tk:02d}_soft.wav")
        rA, rB = res(A, EA["T"]), res(B, EB["T"])
        j = align(A, B)
        rBA = rB[:, j]
        uB = B["u"][j]
        v = A["f0"] > 0
        out = {"ID": rBA, "GLOB": msqrt(EA["C"], 0.5) @ msqrt(EB["C"], -0.5) @ rBA}
        for G, lab in grp.items():
            Ms = [msqrt(EA["Cgrp"][G][g], 0.5) @ msqrt(EB["Cgrp"][G][g], -0.5) for g in range(G)]
            out[f"GRP{G}"] = np.stack([Ms[lab[uB[i]]] @ rBA[:, i] for i in range(A["n"])], 1)
        Xs, Ys, Gs = [], [], []
        for i in range(10):
            if i == tk:
                continue
            a_ = analyse(CORPUS / sa / f"t{i:02d}_neutral.wav"); b_ = analyse(CORPUS / sb / f"t{i:02d}_neutral.wav")
            jj = align(a_, b_)
            vv = a_["f0"] > 0
            Xs.append(res(b_, EB["T"])[:, jj][:, vv]); Ys.append(res(a_, EA["T"])[:, vv]); Gs.append(b_["u"][jj][vv])
        X, Y, Gu = np.concatenate(Xs, 1), np.concatenate(Ys, 1), np.concatenate(Gs)
        lam = 1e-1 * np.trace(X @ X.T) / X.shape[0]
        Wo = Y @ X.T @ np.linalg.inv(X @ X.T + lam * np.eye(24))
        out["ORACLE"] = Wo @ rBA
        lab = grp[max(a.groups)]
        Wg = []
        for g in range(max(a.groups)):
            m = lab[Gu] == g
            Wg.append((Y[:, m] @ X[:, m].T + 200 * Wo) @ np.linalg.inv(X[:, m] @ X[:, m].T + lam * np.eye(24) + 200 * np.eye(24)) if m.any() else Wo)
        out["ORACLE_GRP"] = np.stack([Wg[lab[uB[i]]] @ rBA[:, i] for i in range(A["n"])], 1)
        js = align(A, As)
        EAs_T = EA["T"]
        row = {"A": sa, "B": sb, "text": tk, "FLOOR_self_soft": corr(rA, res(As, EAs_T)[:, js], v)}
        for c, m in out.items():
            row[c] = corr(rA, m, v)
            row[f"{c}_leakB"] = corr(rBA, m, v)
        rows.append({k_: (round(v_, 3) if isinstance(v_, float) else v_) for k_, v_ in row.items()})
        print(rows[-1], flush=True)
    keys = [k_ for k_ in rows[0] if k_ not in ("A", "B", "text")]
    summ = {k_: round(float(np.median([r[k_] for r in rows])), 3) for k_ in keys}
    print("median", summ, flush=True)
    (ROOT / "results/conv_p0/d124.json").write_text(json.dumps({"median": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
