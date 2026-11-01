"""D34(診断・学習なし): 同じ文の組(data/same_text_pairs・男声 TTS 3 人 × 女声 TTS 8 人 × 評価 20 文)で、男声の発話から作る包絡が、同じ文を読んだ女声の実の包絡にどれだけ近いか。
GV = log2(予測の時間方向の分散 / 女声の実の分散)の係数平均(0 = 本物と同じ動きの大きさ・負 = 過平滑)。WARP_a110 / a120 = 伸縮率を固定(D35 と同じ)。
時間合わせ = --align c1(C1 の事後の Bhattacharyya 係数の DTW・TAB に有利な偏りあり)/ wavlm(凍結 WavLM-large の --wl_layer 層の余弦距離・C1 と独立)。距離 = c1..c24 から戻したメル 128 帯の対数スペクトル距離(dB)・LSDdyn = 発話の平均を引いた動きの距離。両方が活性のフレームだけ。
表 = 評価外の 60 文から: 20s(製品の登録の量)と全部(約 3 分)。男声の母集団の平均 = D33 の Rust の登録(男声 TTS 47 人から同じ声の話者を除く)。
  SRC = 男声の包絡・TAB = Σp T_f・M35 = 0.65·TAB + 0.35·SRC・D20 / DFULL = SRC + Σp(T_f − T_m)・DPOPM = SRC + Σp(T_f − T_popM)・
  WARP20 / WARPFULL = Σp T_f + W_α(SRC − Σp T_m)(W_α = メル軸の周波数伸縮 f → f/α・α は表どうしの当てはめ)・VTLN = W_α SRC + 平均のずれ・CEIL = 女声自身の単位で引いた T_f(女声の時間・表の上限)。
    uv run python d34_parallel_map.py --d33 <scratchpad>/d33 --out ../results/conv_p0/d34_parallel.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
DB = 20.0 / np.log(10.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d33", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--align", choices=["c1", "wavlm"], default="wavlm")
    ap.add_argument("--wl_layer", type=int, default=12)
    ap.add_argument("--cache", default="")
    a = ap.parse_args()
    import librosa
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    import zsvc as Z
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    D = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)
    fb = N.mel_fb().numpy()
    fc = fb.argmax(1) * (N.SR / 2) / (fb.shape[1] - 1)
    B = D[1:25]

    def wmat(al: float) -> np.ndarray:
        I = np.zeros((N.N_MEL, N.N_MEL))
        src = np.clip(fc / al, fc[0], fc[-1])
        j = np.clip(np.searchsorted(fc, src) - 1, 0, N.N_MEL - 2)
        w = (src - fc[j]) / np.maximum(fc[j + 1] - fc[j], 1e-9)
        w = np.clip(w, 0, 1)
        I[np.arange(N.N_MEL), j] = 1 - w
        I[np.arange(N.N_MEL), j + 1] += w
        return B @ I @ B.T

    @torch.no_grad()
    def analyse(x):
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        return {"env": env[:, :n].astype(np.float64), "p": p, "u": p.argmax(0)}

    from scipy.signal import resample_poly
    from transformers import WavLMModel
    wl = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()

    @torch.no_grad()
    def wlfeat(x, n):
        h = wl(torch.from_numpy(resample_poly(x, 1, 3).astype(np.float32))[None].to(dev), output_hidden_states=True).hidden_states[a.wl_layer][0].cpu().numpy()
        h = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
        return h[np.minimum(np.arange(n) // 4, len(h) - 1)].astype(np.float64)

    root = ROOT / "data/same_text_pairs"
    males, fems = ["m226", "m227", "m232"], [f"f{c}" for c in "ABCDEFGH"]
    names = sorted(f.name for f in (root / "fA").glob("*.wav"))
    cats = sorted({n.rsplit("_", 1)[0] for n in names})
    ev = [n for c in cats for n in sorted(x for x in names if x.startswith(c + "_"))[12:16]]
    en = [n for n in names if n not in ev]
    import pickle
    cache = Path(a.cache) if a.cache else None
    cached = pickle.loads(cache.read_bytes()) if cache and cache.exists() else {}
    A, TAB20, TABF = {}, {}, {}
    for s in males + fems:
        A[s] = cached.get(s) or {n: analyse(load48(root / s / n)) for n in names}
        for n in ev:
            if f"wl{a.wl_layer}" not in A[s][n]:
                A[s][n][f"wl{a.wl_layer}"] = wlfeat(load48(root / s / n), A[s][n]["env"].shape[1])
        order = [en[i] for i in np.random.RandomState(0).permutation(len(en))]
        sel, tot = [], 0
        for n in order:
            if tot >= 20 * 200:
                break
            sel.append(n); tot += A[s][n]["env"].shape[1]
        mk = lambda ns: C0.table(np.concatenate([A[s][n]["env"][1:25] for n in ns], 1), np.concatenate([A[s][n]["u"] for n in ns]), K, Cb)
        TAB20[s], TABF[s] = mk(sel), mk(en)
        print("analysed", s, flush=True)
    if cache:
        cache.write_bytes(pickle.dumps(A))
    popd = Path(a.d33) / "_work"
    rust = {d.name[3:]: np.fromfile(d / "enroll.bin", dtype="<f4").reshape(24, K).astype(np.float64) for d in sorted(popd.glob("en_male_*"))}
    alphas = np.round(np.arange(1.0, 1.41, 0.02), 2)
    Ws = {al: wmat(al) for al in alphas}
    act = lambda env: (env[0] - PA.C0_SIL) / 10 > 1.0
    rep = {"pairs": {}, "alphas": {}}
    allm: dict = {}
    for m in males:
        popm = np.mean([v for k, v in rust.items() if k != f"male_p{m[1:]}"], 0)
        for f in fems:
            Tf = TAB20[f]
            best = {}
            for tag, Tm in (("20", TAB20[m]), ("F", TABF[m])):
                sf_ = B.T @ Tf
                sf_ = sf_ - sf_.mean(0, keepdims=True)
                ncc = []
                for al in alphas:
                    sm_ = B.T @ (Ws[al] @ Tm)
                    sm_ = sm_ - sm_.mean(0, keepdims=True)
                    ncc.append(float((sf_ * sm_).sum() / np.sqrt((sf_ ** 2).sum() * (sm_ ** 2).sum())))
                best[tag] = float(alphas[int(np.argmax(ncc))])
            rep["alphas"][f"{m}-{f}"] = best
            W20, WF = Ws[best["20"]], Ws[best["F"]]
            vb = (Tf - W20 @ TAB20[m]).mean(1, keepdims=True)
            res: dict = {}
            for n in ev:
                am, af = A[m][n], A[f][n]
                er, p = am["env"][1:25], am["p"]
                if a.align == "c1":
                    C = 1.0 - np.sqrt(am["p"]).T @ np.sqrt(af["p"])
                else:
                    C = 1.0 - am[f"wl{a.wl_layer}"] @ af[f"wl{a.wl_layer}"].T
                _, wp = librosa.sequence.dtw(C=C)
                wp = wp[::-1]
                ok = act(am["env"])[wp[:, 0]] & act(af["env"])[wp[:, 1]]
                im, jf = wp[ok, 0], wp[ok, 1]
                if len(im) < 50:
                    continue
                tab = Tf @ p
                preds = {"SRC": er, "TAB": tab, "M35": 0.65 * tab + 0.35 * er, "D20": er + (Tf - TAB20[m]) @ p, "DFULL": er + (Tf - TABF[m]) @ p,
                         "DPOPM": er + (Tf - popm) @ p, "WARP20": tab + W20 @ (er - TAB20[m] @ p), "WARPFULL": tab + WF @ (er - TABF[m] @ p),
                         "VTLN": W20 @ er + vb, "WARP_a110": tab + Ws[1.1] @ (er - TAB20[m] @ p), "WARP_a120": tab + Ws[1.2] @ (er - TAB20[m] @ p)}
                real = af["env"][1:25][:, jf]
                for k, v in preds.items():
                    pv = v[:, im]
                    lsd = DB * np.sqrt(((pv - real) ** 2).sum(0) / N.N_MEL).mean()
                    dyn = DB * np.sqrt((((pv - pv.mean(1, keepdims=True)) - (real - real.mean(1, keepdims=True))) ** 2).sum(0) / N.N_MEL).mean()
                    gv = float(np.mean(np.log2(pv.var(1) / np.maximum(real.var(1), 1e-12))))
                    res.setdefault(k, []).append((lsd, dyn, gv))
                fa = act(af["env"])
                cv = Tf[:, af["u"]][:, fa]
                rv = af["env"][1:25][:, fa]
                res.setdefault("CEIL", []).append((DB * np.sqrt(((cv - rv) ** 2).sum(0) / N.N_MEL).mean(),
                                                   DB * np.sqrt((((cv - cv.mean(1, keepdims=True)) - (rv - rv.mean(1, keepdims=True))) ** 2).sum(0) / N.N_MEL).mean(),
                                                   float(np.mean(np.log2(cv.var(1) / np.maximum(rv.var(1), 1e-12))))))
            rep["pairs"][f"{m}-{f}"] = {k: [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))] for k, v in res.items()}
            for k, v in res.items():
                allm.setdefault(k, []).extend(v)
            print(m, f, best, {k: round(v[0], 2) for k, v in rep["pairs"][f"{m}-{f}"].items()}, flush=True)
    rep["summary"] = {k: {"lsd_db": round(float(np.mean([x[0] for x in v])), 3), "lsd_dyn_db": round(float(np.mean([x[1] for x in v])), 3),
                          "gv_log2": round(float(np.mean([x[2] for x in v])), 3), "n": len(v)} for k, v in allm.items()}
    keys = list(allm)
    pl = {k: np.array([rep["pairs"][pr][k][0] for pr in rep["pairs"]]) for k in keys}
    rep["pair_wins_vs_M35"] = {k: int((pl[k] < pl["M35"]).sum()) for k in keys}
    print(json.dumps(rep["summary"], indent=1), rep["pair_wins_vs_M35"], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
