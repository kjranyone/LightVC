"""D617(推論だけ): 因果な F1(関門の物差し)が有声の立ち上がりで倍を読む率を、同じ高さ(150〜300 Hz)の本物の女声(FF_SRC ∪ CEN・自分の harvest)と候補で比べる。D615: k5 に残る jump(d109 と同じ: F1 の高さの隣り合う有声フレームの差 > 7.2 半音)の一つ一つを分ける。
立ち上がりからの位置(F1 の有声の連続の何フレーム目か)・倍読みか(跳びの片側の F1 が基準の高さ r から 9 半音超ずれる)・受け継ぎか(r 自身も 7.2 半音超跳ぶ・r が片側 0)・
跳びのフレームの H1 − H2(D607 の最小二乗)。r = 入力の harvest × 2^(k/12)(遅れ合わせ)。規則は results/conv_p0/d615/prereg.yaml。
    uv run python d617_f1_onset_real.py --out ../results/conv_p0/d617/f1_onset.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H
from d607_h1h2 import job

ONSET = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vers", nargs="+", default=["OURS5RY58H4G", "OURS5RY58G"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        tset = set()
        for nm in a.vers:
            k = int(re.match(r"OURS(\d+)", nm).group(1))
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                its.append((nm, s, key, int(ti), k, f))
                c = H / s / "cen_reg" / f"cen_{int(ti):03d}.wav"
                if int(ti) not in tset and c.exists():
                    tset.add(int(ti))
                    its.append(("CEN", s, f"t{int(ti):03d}", int(ti), 0, c))
    for f in sorted((H / "devff2" / "d56").glob("p*_t*_SRC.wav")):
        its.append(("FF_SRC", "devff2", f.stem, 1000 + len(its), 0, f))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    import torch
    import f0est as FE
    ROOT = Path(__file__).resolve().parent.parent
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())

    def f1track(x: np.ndarray) -> np.ndarray:
        x = x[: 8 * 48000].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // 240
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * 240])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        kk = pr[:, :m].argmax(0)
        ii = np.clip(kk[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        return np.where(vo[:m] & (c >= 0.5), np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12)), 0.0)

    rng = np.random.default_rng(617)
    G: dict = {}
    JL: dict = {}
    for (nm, s, key, ti, k, _), x, r, d12 in res:
        g = "REAL" if nm in ("FF_SRC", "CEN") else nm
        fv = f1track(x)
        n = min(len(fv), len(r))
        fv, rr = fv[:n], r[:n]
        on_tot = on_dbl = st_tot = st_dbl = 0
        run = np.zeros(n, int)
        for i in range(n):
            run[i] = run[i - 1] + 1 if rr[i] > 0 and i > 0 else int(rr[i] > 0)
        dbl = (fv > 0) & (rr > 0) & (12 * np.log2(np.where(fv > 0, fv, 1.0) / np.where(rr > 0, rr, 1.0)) > 9)
        for i in np.nonzero(run == 1)[0]:
            if 150 <= rr[i] < 300 and i + 1 < n:
                on_tot += 1
                on_dbl += int(dbl[i] or dbl[i + 1])
        m = (run > 20) & (rr >= 150) & (rr < 300) & (fv > 0)
        st_tot, st_dbl = int(m.sum()), int((dbl & m).sum())
        lo = (rr >= 150) & (rr < 300)
        vsec_lo = float(((fv > 0) & lo).sum()) / 200.0
        jl = 0
        for i in range(1, n):
            if fv[i] > 0 and fv[i - 1] > 0 and abs(12 * np.log2(fv[i] / fv[i - 1])) > 7.2 and (lo[i] or lo[i - 1] or (rr[i] == 0 and rr[i - 1] == 0 and 150 <= min(fv[i], fv[i - 1]) < 300)):
                jl += 1
        JL.setdefault(g, []).append((f"{s}|{ti}", jl, vsec_lo))
        G.setdefault(g, []).append((f"{s}|{ti}", on_tot, on_dbl, st_tot, st_dbl))
    rep: dict = {"groups": {}}
    for g, rows in G.items():
        A = np.array([r_[1:] for r_ in rows], float)
        tg = np.array([r_[0] for r_ in rows])
        ut = np.unique(tg)
        idx = {t: np.flatnonzero(tg == t) for t in ut}
        def rate(ii, a_, b_):
            s_ = A[ii].sum(0)
            return s_[b_] / max(s_[a_], 1)
        bs = [rate(np.concatenate([idx[t] for t in rng.choice(ut, len(ut))]), 0, 1) for _ in range(2000)]
        s_ = A.sum(0)
        rep["groups"][g] = {"onsets": int(s_[0]), "onset_double_rate": round(float(s_[1] / max(s_[0], 1)), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                            "steady_frames": int(s_[2]), "steady_double_rate": round(float(s_[3] / max(s_[2], 1)), 4)}
        print(g, rep["groups"][g], flush=True)
    rep["record_lowband_jumps"] = {}
    for g, rows in JL.items():
        A = np.array([r_[1:] for r_ in rows], float)
        tg = np.array([r_[0] for r_ in rows])
        ut = np.unique(tg)
        idx = {t: np.flatnonzero(tg == t) for t in ut}
        bs = []
        for _ in range(2000):
            ii = np.concatenate([idx[t] for t in rng.choice(ut, len(ut))])
            bs.append(A[ii, 0].sum() / max(A[ii, 1].sum(), 1e-3))
        rep["record_lowband_jumps"][g] = {"jumps": int(A[:, 0].sum()), "voiced_sec_150_300": round(float(A[:, 1].sum()), 1), "per_sec": round(float(A[:, 0].sum() / max(A[:, 1].sum(), 1e-3)), 3),
                                          "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]}
        print("lowband", g, rep["record_lowband_jumps"][g], flush=True)
    c_, r_ = rep["groups"]["OURS5RY58H4G"]["onset_double_rate"], rep["groups"]["REAL"]["onset_double_rate"]
    rep["P1"] = {"real_over_cand": round(r_ / max(c_, 1e-9), 3), "holds": bool(r_ >= 0.5 * c_)}
    print("P1", rep["P1"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
