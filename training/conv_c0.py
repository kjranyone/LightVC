"""変換器の起動前の検査 C0-1・C0-6・C0-7(VCTK 部)(current/converter.md §4・学習なし)。

組: VCTK 女 45 × 男 5 = 225 組(差し替えラダーと同じ話者)。R0 = 男声の共通文 3〜6 を RRPS で目標の音域へ・T0 = 目標の同じ文。
包絡 = CheapTrick(pae.envelope・harvest f0)の DCT c1..c24(c0 = 元のレベルは変えない)。R0 の STFT に「目標の系列 − R0 の包絡」を掛ける。
単位 = 評価話者を含まない VCTK 話者(発話 41〜51)で学習した k-means(E2 = 因果な生徒・製品の経路 / CV = ContentVec)。表 = 目標の参照(発話 41〜・≥ 10s)。
条件:
  ORACLE     目標の実軌道(DTW 整列)
  TAB_E2/CV  表を R0 自身の単位で引く(25ms 平滑)
  DEV_NEAR   TAB_E2 + 表が目標に最も近い別の女声 C の単位内の偏差(C の同じ文・C 自身の表と単位・DTW 整列)  = C0-6 (a)
  DEV_RAND   同じ・C は無作為の女声                                                                    = C0-6 (b)
  DYN_E2     単位 × 位置(連続する同じ単位の始・中・終)の表で引く(無ければ単位の表)                        = C0-6 (c)
  TAB_CVL3   TAB_CV の単位を 3 フレーム(15ms)遅らせる = 質は完全で遅れだけの因果な生徒の模擬
  TAB_E2A3   TAB_E2 の単位を 3 フレーム進める = E2 の遅れを補った質だけの比較(非因果・測定用)
  R0         差し替えなし
採点: 女声 45 人中の目標の順位(ECAPA・WavLM・重心 = 発話 60〜の 3 つ)。対応検定(順位の Wilcoxon・top-1 の McNemar)を TAB_E2 と比べる。

    uv run python conv_c0.py --ladder <scratchpad>/r4_spk --work <scratchpad>/c0_vctk --out ../results/conv_c0/vctk.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
SENTS = (3, 4, 5, 6)
LAD: dict = {}


def utt(s: str, u: int) -> Path:
    return VC / f"wav48/{s}/{s}_{u:03d}.wav"


def G():
    if "G" not in LAD:
        sys.path.insert(0, LAD["path"])
        import gen
        LAD["G"] = gen
    return LAD["G"]


def harvest(x: np.ndarray) -> np.ndarray:
    import librosa
    import pyworld
    x16 = librosa.resample(x.astype(np.float64), orig_sr=48000, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    return pyworld.stonemask(x16, f0, t, 16000)


def analyze(x: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """→ f0 [T]・c [25, T](フレーム t = サンプル tH 中心 = G.stft の center=True と同じ時刻)。"""
    import pae as PA
    f0 = harvest(x)
    f0 = np.pad(f0[:n_frames], (0, max(0, n_frames - len(f0))))
    xa = np.concatenate([np.zeros(PA.A), x, np.zeros(PA.A + 2048)])
    return f0, PA.envelope(xa, f0)


def prep_pair(job: tuple) -> str:
    m, f, wd = job
    g = G()
    from train_dec2 import load48
    out = Path(wd) / "sig" / f"{m}__{f}__R0.npz"
    if out.exists():
        return str(out)
    enr_m = [u for u in range(41, 60) if utt(m, u).exists()][:3]
    enr_f = [u for u in range(41, 60) if utt(f, u).exists()][:3]
    ratio = float(np.exp(g.lf0_med(f, enr_f) - g.lf0_med(m, enr_m)))
    ys = [g.norm(g.reg_shift(g.trim(load48(str(utt(m, u))).astype(np.float64)), ratio)) for u in SENTS if utt(m, u).exists() and utt(f, u).exists()]
    x = np.concatenate(ys)
    T = g.stft(x).shape[1]
    f0, c = analyze(x, T)
    np.savez(out, x=x.astype(np.float32), f0=f0, c=c, ratio=ratio)
    return str(out)


def prep_spk(job: tuple) -> str:
    s, kind, wd = job
    g = G()
    from train_dec2 import load48
    out = Path(wd) / "sig" / f"{s}__{kind}.npz"
    if out.exists():
        return str(out)
    if kind == "T0":
        xs = [g.norm(g.trim(load48(str(utt(s, u))).astype(np.float64))) for u in SENTS if utt(s, u).exists()]
    else:
        xs, tot = [], 0.0
        for u in range(41, 60):
            if tot >= 10.0:
                break
            if utt(s, u).exists():
                y = g.trim(load48(str(utt(s, u))).astype(np.float64))
                xs.append(y)
                tot += len(y) / 48000
    seg = [len(x) for x in xs]
    x = np.concatenate(xs)
    T = g.stft(x).shape[1]
    f0, c = analyze(x, T)
    np.savez(out, x=x.astype(np.float32), f0=f0, c=c, seg=np.array(seg))
    return str(out)


class Feats:
    def __init__(self, dev: str):
        import artic_g2_unit as U
        self.cv, self.e2 = U.CV(dev), U.E2(dev)

    def units(self, kind: str, x: np.ndarray, C: np.ndarray, n_frames: int) -> np.ndarray:
        h = (self.cv if kind == "cv" else self.e2)(x.astype(np.float64))
        u = (h @ C.T).argmax(1)
        t = np.arange(n_frames) * 240 / 48000
        if kind == "cv":
            j = np.round((t - 0.0125) / 0.02).astype(int)
        else:
            j = np.floor(t * 44100 / 256).astype(int)
        return u[np.clip(j, 0, len(u) - 1)]


def table(c: np.ndarray, u: np.ndarray, k: int, C: np.ndarray) -> np.ndarray:
    M = np.zeros((c.shape[0], k))
    has = np.zeros(k, bool)
    for j in range(k):
        m = u == j
        if m.any():
            M[:, j] = c[:, m].mean(1)
            has[j] = True
    if (~has).any():
        sim = C @ C.T
        sim[:, ~has] = -9
        M[:, ~has] = M[:, sim[~has].argmax(1)]
    return M


def runpos(u: np.ndarray) -> np.ndarray:
    pos = np.zeros(len(u), int)
    i = 0
    while i < len(u):
        j = i
        while j + 1 < len(u) and u[j + 1] == u[i]:
            j += 1
        L = j - i + 1
        pos[i:j + 1] = np.minimum(2, (3 * np.arange(L)) // L)
        i = j + 1
    return pos


def dyn_table(c: np.ndarray, u: np.ndarray, k: int, base: np.ndarray) -> np.ndarray:
    pos = runpos(u)
    M = np.repeat(base[:, :, None], 3, 2)
    for j in range(k):
        for p in range(3):
            m = (u == j) & (pos == p)
            if m.sum() >= 2:
                M[:, j, p] = c[:, m].mean(1)
    return M


def smooth(E: np.ndarray, w: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(E, size=w, axis=1, mode="nearest")


def score(wd: Path, pairs: list, fems: list, names: list, bases: tuple) -> dict:
    """女声 45 人中の目標の順位(ECAPA・WavLM-SV・重心 = 発話 60〜の 3 つ)と、bases に対する対応検定(順位の Wilcoxon・top-1 の McNemar)。"""
    import torch
    from a2_dsp_vc import ecapa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    from scipy.stats import wilcoxon, binomtest
    from train_dec2 import load48
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().float()

    def emb_w(x48):
        y = librosa.resample(x48.astype(np.float64), orig_sr=48000, target_sr=16000).astype(np.float32)
        with torch.no_grad():
            v = wm(**fe(y, sampling_rate=16000, return_tensors="pt")).embeddings[0].double().numpy()
        return v / np.linalg.norm(v)
    emb_e = ecapa()
    rep: dict = {}
    for ename, f_ in (("ecapa", emb_e), ("wavlm", emb_w)):
        Cn = {}
        for s in fems:
            c = np.mean([f_(load48(str(utt(s, u))).astype(np.float64)[:8 * 48000]) for u in [u for u in range(60, 200) if utt(s, u).exists()][:3]], 0)
            Cn[s] = c / np.linalg.norm(c)
        ranks = {}
        for nm in names:
            rk = []
            for m, f in pairs:
                x, _ = sf.read(wd / "wav" / f"{m}__{f}__{nm}.wav")
                v = f_(x.astype(np.float64)[:8 * 48000])
                sims = {s: float(v @ Cn[s]) for s in fems}
                rk.append(1 + sum(1 for s in fems if s != f and sims[s] > sims[f]))
            ranks[nm] = np.array(rk)
        res = {}
        for nm, rk in ranks.items():
            r = {"top1": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "median_rank": float(np.median(rk)), "mean_rank": round(float(rk.mean()), 2)}
            for base in bases:
                if nm != base and base in ranks:
                    b = ranks[base]
                    try:
                        r[f"wilcoxon_vs_{base}"] = float(wilcoxon(rk, b, alternative="less").pvalue)
                    except ValueError:
                        r[f"wilcoxon_vs_{base}"] = 1.0
                    w_, l_ = int(((rk == 1) & (b != 1)).sum()), int(((rk != 1) & (b == 1)).sum())
                    r[f"mcnemar_vs_{base}"] = {"win": w_, "loss": l_, "p": float(binomtest(w_, w_ + l_, 0.5, alternative="greater").pvalue) if w_ + l_ else 1.0}
                    tg = np.array([f for _, f in pairs])
                    ut = sorted(set(tg.tolist()))
                    ma = np.array([rk[tg == t].mean() for t in ut])
                    mb = np.array([b[tg == t].mean() for t in ut])
                    t1 = np.array([(rk[tg == t] == 1).mean() - (b[tg == t] == 1).mean() for t in ut])
                    try:
                        r[f"cluster_wilcoxon_vs_{base}"] = float(wilcoxon(ma, mb, alternative="less").pvalue)
                    except ValueError:
                        r[f"cluster_wilcoxon_vs_{base}"] = 1.0
                    r[f"cluster_top1_gain_vs_{base}"] = {"mean": round(float(t1.mean()), 3), "targets_better": int((t1 > 0).sum()), "targets_worse": int((t1 < 0).sum()), "n_targets": len(ut)}
            res[nm] = r
            print(ename, nm, r, flush=True)
        rep[ename] = res
        rep[ename + "_pair_ranks"] = {nm: [int(v) for v in rk] for nm, rk in ranks.items()}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--males_per", type=int, default=5)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()
    LAD["path"] = a.ladder
    wd = Path(a.work)
    (wd / "sig").mkdir(parents=True, exist_ok=True)
    (wd / "wav").mkdir(parents=True, exist_ok=True)
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    fems, males = J["fems"], J["males"]
    pairs = [(males[(i * a.males_per + j) % len(males)], f) for i, f in enumerate(fems) for j in range(a.males_per)]
    if a.n:
        pairs = pairs[:a.n]
    tf = sorted({f for _, f in pairs})
    with Pool(a.procs) as pool:
        pool.map(prep_spk, [(s, kd, str(wd)) for s in fems for kd in ("T0", "REF")], chunksize=1)
        pool.map(prep_pair, [(m, f, str(wd)) for m, f in pairs], chunksize=1)
    print("prep done", len(pairs), flush=True)

    import torch
    import artic_g2_unit as U
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    F = Feats(dev)
    g = G()
    from train_dec2 import load48
    evals = set(fems) | set(males)
    others = sorted(p.name for p in (VC / "wav48").iterdir() if p.name not in evals)
    books = {}
    for kind in ("e2", "cv"):
        ext = F.cv if kind == "cv" else F.e2
        pool_f = [ext(g.trim(load48(str(utt(s, u))).astype(np.float64))) for s in others for u in range(41, 52) if utt(s, u).exists()]
        books[kind] = U.kmeans(np.concatenate(pool_f), a.k, dev)
    print("codebook speakers", len(others), flush=True)

    sp: dict = {}
    for s in fems:
        T0 = np.load(wd / "sig" / f"{s}__T0.npz")
        RF = np.load(wd / "sig" / f"{s}__REF.npz")
        d = {"T0x": T0["x"], "T0c": T0["c"], "REFc": RF["c"]}
        for kind in ("e2", "cv"):
            ur = F.units(kind, RF["x"], books[kind], RF["c"].shape[1])
            d[f"tab_{kind}"] = table(RF["c"][1:25], ur, a.k, books[kind])
            if kind == "e2":
                d["dyn_e2"] = dyn_table(RF["c"][1:25], ur, a.k, d["tab_e2"])
                ut = F.units(kind, T0["x"], books[kind], T0["c"].shape[1])
                d["dev_T0"] = T0["c"][1:25] - d["tab_e2"][:, ut]
        sp[s] = d
    tabs = np.stack([sp[s]["tab_e2"].ravel() for s in fems])
    dist = ((tabs[:, None] - tabs[None]) ** 2).sum(-1)
    np.fill_diagonal(dist, np.inf)
    near = {s: fems[int(dist[i].argmin())] for i, s in enumerate(fems)}
    rng = np.random.default_rng(0)
    rand = {s: fems[int(rng.choice([j for j in range(len(fems)) if fems[j] != s]))] for s in fems}

    names = []
    for m, f in pairs:
        R = np.load(wd / "sig" / f"{m}__{f}__R0.npz")
        xr = R["x"].astype(np.float64)
        Xb = g.stft(xr)
        cb = R["c"][1:25]
        Tb = cb.shape[1]
        ub = {kind: F.units(kind, xr, books[kind], Tb) for kind in ("e2", "cv")}
        d = sp[f]

        def aligned(s):
            pos = g.align_map(xr, sp[s]["T0x"].astype(np.float64), Tb, sp[s]["T0c"].shape[1])
            return pos

        pos_f = aligned(f)
        conds = {"ORACLE": g.interp_frames(d["T0c"][1:25], pos_f),
                 "TAB_E2": smooth(d["tab_e2"][:, ub["e2"]]),
                 "TAB_CV": smooth(d["tab_cv"][:, ub["cv"]])}
        lag = lambda u_, d_: np.concatenate([np.full(d_, u_[0]), u_[:-d_]]) if d_ > 0 else np.concatenate([u_[-d_:], np.full(-d_, u_[-1])])
        conds["TAB_CVL3"] = smooth(d["tab_cv"][:, lag(ub["cv"], 3)])
        conds["TAB_E2A3"] = smooth(d["tab_e2"][:, lag(ub["e2"], -3)])
        pr = runpos(ub["e2"])
        conds["DYN_E2"] = smooth(d["dyn_e2"][:, ub["e2"], pr])
        for nm, other in (("DEV_NEAR", near[f]), ("DEV_RAND", rand[f])):
            dv = g.interp_frames(sp[other]["dev_T0"], aligned(other))
            conds[nm] = smooth(d["tab_e2"][:, ub["e2"]] + dv)
        for nm, seq in conds.items():
            fo = wd / "wav" / f"{m}__{f}__{nm}.wav"
            if fo.exists():
                continue
            gd = np.zeros((128, Tb))
            import zsvc as Z
            Dm = Z.dct_mat(128).numpy()
            gd = Dm[1:25].T @ (seq - cb)
            y = g.istft(Xb * np.exp(g.to_lin(np.clip(gd, -6, 6))), len(xr))
            sf.write(fo, g.norm(y).astype(np.float32), 48000)
        sf.write(wd / "wav" / f"{m}__{f}__R0.wav", g.norm(xr).astype(np.float32), 48000)
        names = sorted(set(names) | set(conds) | {"R0"})
    print("render done", flush=True)

    rep: dict = {"n_pairs": len(pairs), "k": a.k, "codebook_speakers": len(others), "near": near}
    rep.update(score(wd, pairs, fems, names, ("TAB_E2", "R0")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
