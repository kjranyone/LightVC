"""C1(因果な内容符号器)の単位で構音目標表を引いたときの話者の同一性(c1_1 prereg の成功条件 (2)・conv_c0 の 225 組と同じ材料・学習なし)。

単位は元の男声 S0(製品の入力・RRPS をかける前・同じ長さ)から取り、包絡は R0(音域移動済み)の STFT に掛ける(conv_c0 と同じ描画)。
  TAB_E2_RE        E2(E2 自身の k-means)・R0 の単位 = conv_c0 の TAB_E2 を新しいコード経路で作り直し(再現の確認: 0.564 / 0.062)
  TAB_E2_S0        同じ・単位は S0 から
  TAB_C1_S0        C1 の最大の単位(開始状態 = 無音を先に流す)・表 = 参照に C1 を通した最大の単位の平均
  TAB_C1_SOFT_S0   C1 の事後で柔らかく引く(ê = Σ p(u) T[u]・表 = 参照の事後で重みづけた平均)= 製品の引き方
  TAB_CVc1_L3_S0   ContentVec を C1 のコードブックへ割り当て 3 フレーム遅らせた単位(同じコードブックの天井)
  ORACLE・R0       conv_c0 と同じ(キャッシュの wav)

    uv run python conv_c1.py --ladder <scratchpad>/r4_spk --work <scratchpad>/c0_vctk --c1 ../results/c1_1 --out ../results/conv_c0/vctk_c1.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import conv_c0 as C0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--males_per", type=int, default=5)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    import torch
    import artic_g2_unit as U
    import c1_content as CC
    import nvoc as N
    import train_c1 as T1
    import zsvc as Z
    from train_dec2 import load48
    C0.LAD["path"] = a.ladder
    g = C0.G()
    wd = Path(a.work)
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    fems, males = J["fems"], J["males"]
    pairs = [(males[(i * a.males_per + j) % len(males)], f) for i, f in enumerate(fems) for j in range(a.males_per)]
    if a.n:
        pairs = pairs[:a.n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = C0.Feats(dev)
    evals = set(fems) | set(males)
    others = sorted(p.name for p in (C0.VC / "wav48").iterdir() if p.name not in evals)
    pool_f = [F.e2(g.trim(load48(str(C0.utt(s, u))).astype(np.float64))) for s in others for u in range(41, 52) if C0.utt(s, u).exists()]
    book_e2 = U.kmeans(np.concatenate(pool_f), a.k, dev)
    c1d = Path(a.c1)
    st = torch.load(c1d / "last.pt", map_location="cpu", weights_only=False)
    net = CC.C1(st["cfg"]["k"], st["cfg"]["ch"], tuple(st["cfg"]["dils"])).to(dev).eval()
    net.load_state_dict(st["net"])
    front = CC.MelFront().to(dev)
    Cb = torch.load(c1d / "codebook.pt").to(dev)
    K = Cb.shape[0]
    tau = st.get("tau", 0.05)
    Dm = Z.dct_mat(128).numpy()

    @torch.no_grad()
    def c1_post(x: np.ndarray, n: int) -> np.ndarray:
        xp = torch.from_numpy(T1.prime(x.astype(np.float32)))[None].to(dev)
        lg = net(front(xp))[..., -(len(x) // N.HOP):]
        p = lg.softmax(1)[0].T.cpu().numpy()
        return np.pad(p, ((0, max(0, n - len(p))), (0, 0)), mode="edge")[:n]

    @torch.no_grad()
    def cv_units_c1book(x: np.ndarray, n: int) -> np.ndarray:
        h = torch.from_numpy(F.cv(x.astype(np.float64))).to(dev)
        u = (h @ Cb.T).argmax(1).cpu().numpy()
        t = np.arange(n) * 240 / 48000
        return u[np.clip(np.round((t - 0.0125) / 0.02).astype(int), 0, len(u) - 1)]

    lag = lambda u_, d_: np.concatenate([np.full(d_, u_[0]), u_[:-d_]])
    sp: dict = {}
    for s in fems:
        RF = np.load(wd / "sig" / f"{s}__REF.npz")
        c, x = RF["c"][1:25], RF["x"].astype(np.float64)
        n = c.shape[1]
        p = c1_post(x, n)
        ue2 = F.units("e2", x, book_e2, n)
        ucv = cv_units_c1book(x, n)
        sp[s] = {"tab_e2": C0.table(c, ue2, a.k, book_e2), "tab_c1": C0.table(c, p.argmax(1), K, Cb.cpu().numpy()),
                 "tab_c1s": (c @ p) / np.maximum(p.sum(0), 1e-6)[None], "tab_cv": C0.table(c, ucv, K, Cb.cpu().numpy())}
        miss = p.sum(0) < 1e-3
        if miss.any():
            sp[s]["tab_c1s"][:, miss] = c.mean(1, keepdims=True)
    names = ["ORACLE", "R0", "TAB_E2", "TAB_E2_RE", "TAB_E2_S0", "TAB_C1_S0", "TAB_C1_SOFT_S0", "TAB_CVc1_L3_S0"]
    for m, f in pairs:
        R = np.load(wd / "sig" / f"{m}__{f}__R0.npz")
        xr = R["x"].astype(np.float64)
        us = [u for u in C0.SENTS if C0.utt(m, u).exists() and C0.utt(f, u).exists()]
        xs = np.concatenate([g.norm(g.trim(load48(str(C0.utt(m, u))).astype(np.float64))) for u in us])
        assert len(xs) == len(xr), (m, f, len(xs), len(xr))
        Xb = g.stft(xr)
        cb = R["c"][1:25]
        Tb = cb.shape[1]
        d = sp[f]
        p = c1_post(xs, Tb)
        conds = {"TAB_E2_RE": C0.smooth(d["tab_e2"][:, F.units("e2", xr, book_e2, Tb)]),
                 "TAB_E2_S0": C0.smooth(d["tab_e2"][:, F.units("e2", xs, book_e2, Tb)]),
                 "TAB_C1_S0": C0.smooth(d["tab_c1"][:, p.argmax(1)]),
                 "TAB_C1_SOFT_S0": C0.smooth(d["tab_c1s"] @ p.T),
                 "TAB_CVc1_L3_S0": C0.smooth(d["tab_cv"][:, lag(cv_units_c1book(xs, Tb), 3)])}
        for nm, seq in conds.items():
            fo = wd / "wav" / f"{m}__{f}__{nm}.wav"
            if fo.exists():
                continue
            gd = Dm[1:25].T @ (seq - cb)
            y = g.istft(Xb * np.exp(g.to_lin(np.clip(gd, -6, 6))), len(xr))
            sf.write(fo, g.norm(y).astype(np.float32), 48000)
    print("render done", flush=True)
    rep = {"n_pairs": len(pairs), "c1_ckpt_step": st["step"], "k": K, "tau": tau}
    rep.update(C0.score(wd, pairs, fems, names, ("TAB_E2", "TAB_E2_S0", "R0")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
