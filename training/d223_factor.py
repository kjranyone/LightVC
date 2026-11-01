"""D223(診断・results/conv_p0/d223_prereg.yaml): 製品の条件の 1 つの要素だけをゼロ位相の低域通過(8 Hz・帰属専用)で平滑して鳴らし、F1・F2 の域(mel 8〜22 番 ≈ 750–2370 Hz)の速い揺れの上乗せの減りを測る。
PST = 表の分・PSR = 操作者の残差の分(pbase.py で RUSTB − B を分ける)・PSP = 周期性の行・PSF = f0・PSC = c0。対照 = <D>/d139/*_G03P.wav(D220・同じ出力部)。
    uv run python d223_factor.py --root <dev の親> --out ../results/conv_p0/d223.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

import nvoc as N
import pbase as PB
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
FORMS = ("PST", "PSR", "PSP", "PSF", "PSC")
MID = slice(8, 23)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    a = ap.parse_args()
    import librosa
    import rvoc as R
    import train_rvoc as TR
    RT = Path(a.root)
    sr = N.SR
    dev = "cuda"
    s_ = torch.load(ROOT / a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(s_["ema"]); gen.eval().requires_grad_(False)

    @torch.no_grad()
    def render(c: np.ndarray) -> np.ndarray:
        f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0)
        cond = torch.from_numpy(c.T.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float64)

    cf = 1000.0 * 2.0 ** (np.arange(-13, 15) / 3.0)
    cf = cf[(cf >= 60) & (cf <= 20000)]

    def ltas(x: np.ndarray) -> np.ndarray:
        n = len(x) // 240
        e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        fq, _, Zs = ss.stft(x, sr, nperseg=2048, noverlap=2048 - 240)
        P = np.abs(Zs[:, :n]) ** 2
        P = P[:, act[: P.shape[1]]].mean(1)
        L = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() + 1e-20 for c in cf]))
        return L - L.mean()

    def eqfir(g: np.ndarray) -> np.ndarray:
        g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        g = np.clip(0.5 * (g - np.median(g)), -12, 12)
        fr = np.r_[0.0, cf / (sr / 2), 1.0]
        gl = 10 ** (np.r_[g[0], g, g[-1]] / 20)
        return ss.minimum_phase(ss.firwin2(1023, np.clip(fr, 0, 1), gl ** 2, fs=2.0), method="homomorphic")

    fb = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=40, fmin=50, fmax=10000)
    hp = ss.butter(4, 4.0, "highpass", fs=sr / N.HOP, output="sos")
    lp = ss.butter(2, 8.0, "lowpass", fs=sr / N.HOP, output="sos")

    def meas(x: np.ndarray) -> dict:
        x = x[: int(8.0 * sr)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // N.HOP
        e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        _, _, Z = ss.stft(x, sr, nperseg=2048, noverlap=2048 - N.HOP)
        P = np.abs(Z[:, :n]) ** 2
        act = act[: P.shape[1]]
        lm = 10 * np.log10(fb @ P + 1e-10)
        cep = np.fft.irfft(np.log(P + 1e-12), axis=0)
        cep[60: cep.shape[0] - 59] = 0.0
        ll = 10 * np.log10(fb @ np.exp(np.fft.rfft(cep, axis=0).real) + 1e-10)
        fast = ss.sosfiltfilt(hp, lm - lm.mean(1, keepdims=True), axis=1)
        return {"lmstd": float(lm[:, act].std(1).mean()), "raw": lm[:, act].std(1), "lift": ll[:, act].std(1), "fast": fast[:, act].var(1)}

    def sm(x: np.ndarray) -> np.ndarray:
        return ss.sosfiltfilt(lp, x, axis=0)

    def sm_v(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        t = np.arange(len(x))
        y = x.copy()
        if v.sum() < 4:
            return y
        xi = np.stack([np.interp(t, t[v], x[v, j]) for j in range(x.shape[1])], 1)
        y[v] = sm(xi)[v]
        return y

    W = PB.warp_matrix(1.3)
    rows: dict = {f: [] for f in ("G03P",) + FORMS}
    tref: dict = {}
    for D in SETS:
        S = RT / D
        od = S / "d223"; od.mkdir(exist_ok=True)
        cal = S / "d114/_cal_G03S10Z"
        work = S / "d85_work"
        tl: dict = {}
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if ti not in tl:
                tl[ti] = ltas(load48(S / "d37/_work" / f"ref_{ti:03d}.wav").astype(np.float64))
                tref[(D, ti)] = meas(load48(S / "d37/_work" / f"ref_{ti:03d}.wav"))
            h = eqfir(tl[ti] - ltas(load48(cal / f"op{pi:04d}_t{ti:03d}.wav").astype(np.float64)))
            C = np.fromfile(S / "d214" / f"{k}_RUSTB.f32", "<f4").reshape(-1, 31).astype(np.float64)
            bv = np.fromfile(cal / f"bias_{pi:04d}_t{ti:03d}.f32", "<f4").astype(np.float64)
            v = C[:, 26] > 0.5
            base = C[:, 1:25] - bv[None] * v[:, None]
            en, op = PB.load_enroll(work / f"en_{ti:03d}"), PB.load_enroll(work / f"op_{pi:04d}")
            lg, e = PB.read_front(S / "d214" / f"{k}_FRONT.f32", en["k"])
            q = np.exp(lg - lg.max(1, keepdims=True)); tab = (q @ en["table"].T) / q.sum(1, keepdims=True) / 10.0
            full = PB.base_env(lg, e, en, op, W, PB.color(en, op, W))
            m = min(len(full), len(base))
            assert abs(len(full) - len(base)) <= 2 and np.abs(full[:m] - base[:m]).max() < 1e-3, k
            tab = np.pad(tab, ((0, max(0, len(base) - len(tab))), (0, 0)), mode="edge")[: len(base)]
            res = base - tab
            rec = {"set": D, "key": k, "t": ti}
            rows["G03P"].append({**rec, **meas(sf.read(S / "d139" / f"{k}_G03P.wav", dtype="float32")[0])})
            for fm in FORMS:
                c = C.copy()
                if fm == "PST":
                    c[:, 1:25] = sm(tab) + res + bv[None] * v[:, None]
                elif fm == "PSR":
                    c[:, 1:25] = tab + sm(res) + bv[None] * v[:, None]
                elif fm == "PSP":
                    pp = sm_v(C[:, 27:31], v); c[:, 27:31] = np.where(v[:, None], pp, C[:, 27:31])
                elif fm == "PSF":
                    c[:, 25:26] = np.where(v[:, None], sm_v(C[:, 25:26], v), C[:, 25:26])
                else:
                    c[:, 0] = sm(C[:, 0])
                y = ss.lfilter(h, 1.0, render(c))
                y = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
                sf.write(od / f"{k}_{fm}.wav", y, sr)
                rows[fm].append({**rec, **meas(y)})
        print(D, flush=True)
    rep: dict = {}
    mid = lambda r, kk: float(np.mean(r[kk][MID] - tref[(r["set"], r["t"])][kk][MID]))
    for fm, rr in rows.items():
        rep[fm] = {"lmstd": round(float(np.median([r["lmstd"] for r in rr])), 3),
                   "lmstd_minus_tref": round(float(np.median([r["lmstd"] - tref[(r["set"], r["t"])]["lmstd"] for r in rr])), 3),
                   "mid_raw_std_minus_tref": round(float(np.median([mid(r, "raw") for r in rr])), 3),
                   "mid_lift_std_minus_tref": round(float(np.median([mid(r, "lift") for r in rr])), 3),
                   "mid_fast_var_minus_tref": round(float(np.median([mid(r, "fast") for r in rr])), 3)}
    base_ex = rep["G03P"]["mid_fast_var_minus_tref"]
    for fm in FORMS:
        dd = [mid(x, "fast") - mid(y, "fast") for x, y in zip(rows["G03P"], rows[fm])]
        rep[fm]["R_x"] = round(float(np.median(dd)) / base_ex, 3) if base_ex > 0 else None
        rep[fm]["mid_fast_var_drop"] = round(float(np.median(dd)), 3)
    rk = sorted(FORMS, key=lambda f: -rep[f]["R_x"])
    rep["rank"] = rk
    rep["verdict"] = "ONE" if rep[rk[0]]["R_x"] >= 0.5 else "TWO" if rep[rk[0]]["R_x"] + rep[rk[1]]["R_x"] >= 0.5 else "NONE"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
