"""D69(診断・学習なし): 単位の中の包絡の動き res(本物の包絡 − 同じ発話の単位平均の事後混合)のうち、因果に手に入る韻律(c0・log f0・有声・周期性 4 帯)から線形に予測できる割合。
話者ごとに発話を前半 / 後半に分けて片方でリッジ回帰を当て、もう片方で決定係数 R² を測る(両方向の平均)。res 全体・遅い成分(LP fc)・速い成分(HP)で別に測る。
特徴 = 現在のフレームと因果の過去(lags フレーム前)の韻律・その 2 次の項(c0², lf², c0·lf)。目標の実音声 30 本(D37 の cen)・解析は D68 と同じ。
    uv run python d69_res_prosody.py --s <scratchpad> --out ../results/conv_p0/d69.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def feats(c0n: np.ndarray, lf: np.ndarray, v: np.ndarray, per: np.ndarray, lags: list) -> np.ndarray:
    base = np.concatenate([c0n[None], lf[None], v[None], per * v[None], (c0n ** 2)[None], (lf ** 2)[None], (c0n * lf)[None]], 0)
    out = [base]
    for L in lags:
        out.append(np.concatenate([np.repeat(base[:, :1], L, 1), base[:, :-L]], 1))
    return np.concatenate(out + [np.ones((1, base.shape[1]))], 0)


def r2(y: np.ndarray, yh: np.ndarray) -> float:
    return float(1 - ((y - yh) ** 2).sum() / max(((y - y.mean(1, keepdims=True)) ** 2).sum(), 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fc", type=float, default=8.0)
    ap.add_argument("--lam", type=float, default=1.0)
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
    fr = N.SR / N.HOP
    sos = butter(4, a.fc, fs=fr, output="sos")
    variants = {"now": [], "lag5_20": [5, 20], "lag5_20_60": [5, 20, 60]}
    acc: dict = {v: {"all": [], "slow": [], "fast": []} for v in variants}
    for ti in range(30):
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        with torch.no_grad():
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        er = env[1:25]
        res = er - C0.table(er, p.argmax(0), K, Cb) @ p
        lo = sosfiltfilt(sos, res, axis=1)
        parts = {"all": res, "slow": lo, "fast": res - lo}
        v = (f0 > 0).astype(np.float64)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)
        c0n = (env[0] - PA.C0_SIL) / 10
        h = n // 2
        for vn, lags in variants.items():
            Z = feats(c0n, lf, v, np.clip(per, 0, 1.5), lags)
            for pn, Y in parts.items():
                sc = []
                for tr, te in ((slice(0, h), slice(h, n)), (slice(h, n), slice(0, h))):
                    Zt, Yt = Z[:, tr], Y[:, tr]
                    B = Yt @ Zt.T @ np.linalg.inv(Zt @ Zt.T + a.lam * np.eye(Z.shape[0]))
                    sc.append(r2(Y[:, te], B @ Z[:, te]))
                acc[vn][pn].append(float(np.mean(sc)))
        print("target", ti, {pn: round(acc["lag5_20"][pn][-1], 3) for pn in parts}, flush=True)
    rep = {"fc_hz": a.fc, "lam": a.lam, "heldout_half_r2_mean": {vn: {pn: round(float(np.mean(x)), 4) for pn, x in d.items()} for vn, d in acc.items()},
           "heldout_half_r2_median": {vn: {pn: round(float(np.median(x)), 4) for pn, x in d.items()} for vn, d in acc.items()}}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
