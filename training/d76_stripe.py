"""D76(診断・学習なし): 単位の中の動き res の正体 = 高い f0 で包絡(CheapTrick → mel → DCT24)に残る倍音の縞が f0 につれて動くもの、という仮説の数値検査。
滑らかな包絡(同じ発話の単位平均の事後混合 INS·p)を本物の f0 の調波で合成し(各倍音の振幅 = 包絡を倍音の周波数で読む・位相は連続)、同じ解析で包絡を取り直す → res_re = 取り直した包絡 − その単位平均の事後混合。
res(本物)と res_re の相関(有声フレーム・次元ごと → 平均)と、res を res_re で回帰したときの R²(話者ごとに前半 / 後半で当てて他方で測る)を印字。目標の実音声 30 本(D37 の cen)。
出力 = results/conv_p0/d76.json。仮説が正しければ相関・R² が大きい → 必要なのは「動き」でなく f0 と整合した包絡(滑らかな包絡を f0 で合成して取り直す決定的な計算で作れる)。
    uv run python d76_stripe.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--save_dir", default="")
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    fb, dct = PA._np_consts()
    fbn = fb / np.maximum(fb.sum(1, keepdims=True), 1e-9)
    fc = (fbn * np.arange(fb.shape[1])[None]).sum(1) * (N.SR / 2) / (fb.shape[1] - 1)
    D25 = dct[:25]
    Dinv = np.linalg.pinv(D25)

    def smooth_env(xa: np.ndarray, f0: np.ndarray, n: int) -> np.ndarray:
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        return (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)

    def synth(c: np.ndarray, f0: np.ndarray, n: int) -> np.ndarray:
        logmel = Dinv @ c
        t = np.arange(n * N.HOP)
        fpos = (np.arange(n) * N.HOP).astype(np.float64)
        f0v = np.where(f0 > 0, f0, 0.0)
        fs = np.interp(t, fpos, f0v)
        ph = 2 * np.pi * np.cumsum(fs) / N.SR
        y = np.zeros(len(t))
        vmask = np.interp(t, fpos, (f0 > 0).astype(np.float64))
        kmax = int((N.SR / 2) / max(f0v[f0v > 0].min() if (f0v > 0).any() else 100.0, 50.0))
        for k in range(1, kmax + 1):
            fk = k * f0v
            ok = (fk > 0) & (fk < N.SR / 2 - 200)
            if not ok.any():
                break
            la = np.array([np.interp(fk[i], fc, logmel[:, i]) if ok[i] else -20.0 for i in range(n)])
            amp = np.interp(t, fpos, np.exp(la) * ok)
            y += amp * np.sin(k * ph)
        y *= vmask
        return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)

    def res_of(c: np.ndarray, p: np.ndarray) -> np.ndarray:
        er = c[1:25]
        return er - C0.table(er, p.argmax(0), K, Cb) @ p

    corr_all, r2_all, rms_re, rms_real = [], [], [], []
    for ti in range(a.max):
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        env = smooth_env(xa, f0, n)
        with torch.no_grad():
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        res = res_of(env, p)
        sm = env.copy()
        sm[1:25] = env[1:25] - res
        ys = synth(sm, f0, n)
        if a.save_dir:
            Path(a.save_dir).mkdir(parents=True, exist_ok=True)
            sf.write(Path(a.save_dir) / f"synth_{ti:03d}.wav", ys, N.SR)
        ysa = np.concatenate([np.zeros(PA.A, np.float32), ys, np.zeros(PA.A, np.float32)])
        env_re = smooth_env(ysa, f0, n)
        rre = res_of(env_re, p)
        v = f0 > 0
        rv, qv = res[:, v], rre[:, v]
        corr_all.append(float(np.mean([np.corrcoef(rv[d], qv[d])[0, 1] for d in range(24)])))
        h = v.sum() // 2
        sc = []
        for tr, te in ((slice(0, h), slice(h, None)), (slice(h, None), slice(0, h))):
            X, Y = qv[:, tr], rv[:, tr]
            Xa = np.concatenate([X, np.ones((1, X.shape[1]))])
            B = Y @ Xa.T @ np.linalg.inv(Xa @ Xa.T + 1.0 * np.eye(25))
            Xt = np.concatenate([qv[:, te], np.ones((1, qv[:, te].shape[1]))])
            Yt = rv[:, te]
            sc.append(float(1 - ((Yt - B @ Xt) ** 2).sum() / ((Yt - Yt.mean(1, keepdims=True)) ** 2).sum()))
        r2_all.append(float(np.mean(sc)))
        rms_re.append(float(rre[:, v].std())); rms_real.append(float(res[:, v].std()))
        print("target", ti, "corr", round(corr_all[-1], 3), "heldout R2", round(r2_all[-1], 3), "rms re/real", round(rms_re[-1], 3), round(rms_real[-1], 3), "f0 med", round(float(np.median(f0[v])), 1), flush=True)
    rep = {"corr_mean": round(float(np.mean(corr_all)), 4), "corr_median": round(float(np.median(corr_all)), 4),
           "heldout_r2_mean": round(float(np.mean(r2_all)), 4), "heldout_r2_median": round(float(np.median(r2_all)), 4),
           "rms_re_mean": round(float(np.mean(rms_re)), 4), "rms_real_mean": round(float(np.mean(rms_real)), 4)}
    print(json.dumps(rep), flush=True)
    (ROOT / "results/conv_p0/d76.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
