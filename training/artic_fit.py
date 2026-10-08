"""構音の逆推定 段 1(合成による分析・学習なし): 声道の物理形状(vtract)を倍音の振幅へ当てはめる。current/artic_inv.md。

観測: 精密 f0(harvest)の各フレーム(10ms)で、3 周期のハン窓をかけて倍音 k·f0(0〜F_MAX)の振幅を直接求める(ピッチ適応)。
モデル: log A(x) = log Ā + Σ_{k=1..K} q_k cos(π k x)・声道長 L(発話ごと)・声門の傾斜(傾き)と利得(フレームごと)。
     包絡(dB)= vtract.envelope_db(A, L, f) + glottal_tilt_db(f; FC, slope) + gain。照合は倍音の位置だけ。
当てはめ: 全フレームを同時に Adam。拘束 = 時間方向の差分(滑らかさ)+ q の大きさ(中立の形への引き)。

    uv run python artic_fit.py --g0 --g1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import vtract as V

SR = 48000
HOP = 480
F_MAX = 5000.0
N_SEC = 20
K = 6
LOSS_SCALE = False
FC = 120.0
A0 = 3e-4
ROOT = Path(__file__).resolve().parent.parent


def harvest(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    import pyworld
    x16 = librosa.resample(x.astype(np.float64), orig_sr=SR, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=900, frame_period=HOP / SR * 1000)
    return pyworld.stonemask(x16, f0, t, 16000), t


def harmonic_obs_exact(x: np.ndarray, f0: np.ndarray, kmax: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """参照実装(倍音の周波数で DFT を直接)。harmonic_obs の検査用。"""
    T = len(f0)
    fr = np.zeros((T, kmax))
    am = np.full((T, kmax), -120.0)
    mk = np.zeros((T, kmax), bool)
    k = np.arange(1, kmax + 1)
    for t in range(T):
        if f0[t] <= 0:
            continue
        P = SR / f0[t]
        W = int(3 * P) | 1
        c = t * HOP
        lo, hi = c - W // 2, c + W // 2 + 1
        if lo < 0 or hi > len(x):
            continue
        seg = x[lo:hi] * np.hanning(W)
        n = np.arange(W) - W // 2
        fk = k * f0[t]
        ok = fk <= F_MAX
        ph = np.exp(-2j * np.pi * np.outer(fk[ok], n) / SR)
        a = np.abs(ph @ seg) / (np.hanning(W).sum() / 2)
        fr[t, ok] = fk[ok]
        am[t, ok] = 20 * np.log10(a + 1e-9)
        mk[t, ok] = True
    return fr, am, mk


def harmonic_obs(x: np.ndarray, f0: np.ndarray, kmax: int = 64, nfft: int = 16384) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ freqs [T, kmax]・amp_db [T, kmax]・mask [T, kmax](有声かつ k·f0 ≤ F_MAX)。フレーム t の中心は t·HOP。
    3 周期のハン窓(中心揃え・観測の道具)を零詰め FFT し、倍音の周波数で |X| を線形補間(harmonic_obs_exact と 0.1dB 以内)。"""
    T = len(f0)
    fr = np.zeros((T, kmax))
    am = np.full((T, kmax), -120.0)
    mk = np.zeros((T, kmax), bool)
    k = np.arange(1, kmax + 1)
    for t in range(T):
        if f0[t] <= 0:
            continue
        P = SR / f0[t]
        W = int(3 * P) | 1
        c = t * HOP
        lo, hi = c - W // 2, c + W // 2 + 1
        if lo < 0 or hi > len(x):
            continue
        w = np.hanning(W)
        seg = x[lo:hi] * w
        buf = np.zeros(nfft)
        buf[:W // 2 + 1] = seg[W // 2:]
        buf[-(W // 2):] = seg[:W // 2]
        X = np.abs(np.fft.rfft(buf)) / (w.sum() / 2)
        fk = k * f0[t]
        ok = fk <= F_MAX
        b = fk[ok] * nfft / SR
        i0 = np.floor(b).astype(int)
        a = b - i0
        val = X[i0] * (1 - a) + X[i0 + 1] * a
        fr[t, ok] = fk[ok]
        am[t, ok] = 20 * np.log10(val + 1e-9)
        mk[t, ok] = True
    return fr, am, mk


def basis(n: int | None = None, k: int | None = None) -> torch.Tensor:
    n = N_SEC if n is None else n
    k = K if k is None else k
    x = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    return torch.stack([torch.cos(math.pi * (i + 1) * x) for i in range(k)], 0)


def model_db(q: torch.Tensor, logL: torch.Tensor, src: torch.Tensor, gain: torch.Tensor, fr: torch.Tensor, B: torch.Tensor,
             f0: torch.Tensor) -> torch.Tensor:
    """q [T,K]・logL []・src [T,2](log γ = log(fg/f0)・log fa)・gain [T]・fr [T,H]・f0 [T] → [T,H] dB。
    声門流: 声門フォルマント fg = γ·f0 の 2 次の低域(−12dB/oct)× 閉鎖相の傾斜 fa の 1 次の低域(−6dB/oct)。放射 +6dB/oct は jω。"""
    areas = (A0 * torch.exp(q @ B)).clamp(5e-6, 2e-3)
    L = torch.exp(logL).expand(q.shape[0])
    f = fr.clamp(min=20.0)
    h = transfer_pointwise(areas, L, f, torch.exp(src[:, 2]) if src.shape[1] > 2 else None)
    env = 20 * torch.log10((2 * math.pi * f) * h.abs() + 1e-12)
    fg = (torch.exp(src[:, 0]) * f0)[:, None]
    fa = torch.exp(src[:, 1])[:, None]
    g = -20 * torch.log10(1 + (f / fg) ** 2) - 10 * torch.log10(1 + (f / fa) ** 2)
    return env + g + gain[:, None]


def transfer_pointwise(areas: torch.Tensor, L: torch.Tensor, f: torch.Tensor, loss_scale: torch.Tensor | None = None) -> torch.Tensor:
    """フレームごとに周波数の並びが違う(倍音)ので、areas [T,N]・L [T]・f [T,H] で評価する版。loss_scale [T] は壁の損失(帯域幅)の倍率。"""
    N = areas.shape[-1]
    l = (L / N)[:, None, None]
    w = 2 * math.pi * f
    al = V.wall_loss(f) * (loss_scale[:, None] if loss_scale is not None else 1.0)
    gam = (al + 1j * w / V.C)[:, None, :]
    Z = (V.RHO * V.C / areas)[:, :, None]
    gl = gam * l
    zero = torch.zeros_like(Z)
    ch, sh = torch.cosh(gl) + zero, torch.sinh(gl) + zero
    m11, m12, m21, m22 = ch, Z * sh, sh / Z, ch
    p11, p12, p21, p22 = m11[:, 0], m12[:, 0], m21[:, 0], m22[:, 0]
    for i in range(1, N):
        a11, a12, a21, a22 = m11[:, i], m12[:, i], m21[:, i], m22[:, i]
        p11, p12, p21, p22 = p11 * a11 + p12 * a21, p11 * a12 + p12 * a22, p21 * a11 + p22 * a21, p21 * a12 + p22 * a22
    k = w / V.C
    a = torch.sqrt(areas[:, -1] / math.pi)[:, None]
    ka = k * a
    zr = (V.RHO * V.C / areas[:, -1])[:, None] * ((ka ** 2) / 2 + 1j * (8 * ka / (3 * math.pi)))
    return 1.0 / (p21 * zr + p22)



LGRID = (0.13, 0.145, 0.16, 0.175, 0.19)
FGRID = np.exp(np.linspace(np.log(80.0), np.log(F_MAX), 160))


def codebook(m: int = 20000, sigma: float = 0.8, dev: str = "cpu", seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """声道だけの包絡(放射込み・声門なし)を FGRID 上で: q [M·|L|, K]・logL [M·|L|]・env [M·|L|, |FGRID|](dB)。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(m, K, generator=g, dtype=torch.float64) * sigma
    q[0] = 0.0
    B = basis()
    f = torch.from_numpy(FGRID)
    qs, ls, es = [], [], []
    for L in LGRID:
        areas = (A0 * torch.exp(q @ B)).clamp(5e-6, 2e-3)
        env = []
        for i in range(0, m, 4000):
            env.append(V.envelope_db(areas[i:i + 4000].to(dev), torch.full((min(4000, m - i),), L, dtype=torch.float64, device=dev), f.to(dev)).cpu())
        qs.append(q)
        ls.append(torch.full((m,), math.log(L), dtype=torch.float64))
        es.append(torch.cat(env))
    return torch.cat(qs).to(dev), torch.cat(ls).to(dev), torch.cat(es).to(dev)


def dp_init(fr: np.ndarray, am: np.ndarray, mk: np.ndarray, f0: np.ndarray, cb: tuple, topk: int = 32, lam: float = 2.0,
            dev: str = "cpu", fix_L: float | None = None, chunk: int = 48) -> tuple[np.ndarray, float]:
    """各有声フレームで、コードブックの包絡を倍音の位置へ補間し、利得と傾き(log f の 1 次)を最小二乗で除いた残差で近い候補 topk を選ぶ
    (フレームをかたまりで GPU 上でまとめて計算)。声道長は発話で共通(候補の L ごとに DP し、最良の L を選ぶ)。
    経路の遷移コスト = lam·|q_t − q_{t−1}|²。→ q の初期値 [Tv, K]・L。"""
    cq, cl, ce = cb
    vt = np.nonzero(mk.any(1))[0]
    lf = torch.from_numpy(np.log(FGRID)).to(dev)
    H = mk.shape[1]
    frv = torch.from_numpy(np.where(mk[vt], fr[vt], FGRID[0])).to(dev)
    amv = torch.from_numpy(am[vt]).to(dev)
    mkv = torch.from_numpy(mk[vt]).to(dev).double()
    x = torch.log(frv)
    j = (torch.searchsorted(lf, x) - 1).clamp(0, len(FGRID) - 2)
    a = (x - lf[j]) / (lf[j + 1] - lf[j])
    top = torch.where(mkv > 0, amv, torch.full_like(amv, -1e9)).amax(1, keepdim=True)
    w = mkv * torch.sigmoid((amv - (top - 25.0)) / 3.0)
    xm = ((x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1e-9))
    xc = (x - xm) * (mkv > 0)
    Ls = LGRID if fix_L is None else (min(LGRID, key=lambda z: abs(z - math.exp(fix_L))),)
    best = None
    for L in Ls:
        sel = torch.nonzero((cl - math.log(L)).abs() < 1e-6).squeeze(1)
        E, Q = ce[sel].float(), cq[sel]
        costs, idxs = [], []
        for c0 in range(0, len(vt), chunk):
            sl = slice(c0, c0 + chunk)
            jj, aa, ww, yy, xx = j[sl], a[sl].float(), w[sl].float(), amv[sl].float(), xc[sl].float()
            Ei = E[:, jj] * (1 - aa) + E[:, jj + 1] * aa
            R = yy[None] - Ei
            sw = ww.sum(1).clamp(min=1e-9)
            m0 = (R * ww).sum(2) / sw
            Rc = R - m0[..., None]
            sxx = (ww * xx * xx).sum(1).clamp(min=1e-9)
            b1 = (Rc * ww * xx).sum(2) / sxx
            res = Rc - b1[..., None] * xx
            c = (res ** 2 * ww).sum(2) / sw
            v_, ix = torch.topk(-c, topk, dim=0)
            costs += list((-v_).T.double())
            idxs += list(ix.T)
        Tn = len(vt)
        acc = costs[0].clone()
        back = []
        for i in range(1, Tn):
            qa, qb = Q[idxs[i - 1]], Q[idxs[i]]
            tr = lam * ((qb[:, None, :] - qa[None, :, :]) ** 2).sum(2) * (1.0 if vt[i] - vt[i - 1] == 1 else 0.0)
            m_, a_ = (acc[None, :] + tr).min(1)
            acc = costs[i] + m_
            back.append(a_)
        tot = float(acc.min())
        if best is None or tot < best[0]:
            path = [int(acc.argmin())]
            for b in reversed(back):
                path.append(int(b[path[-1]]))
            path = path[::-1]
            best = (tot, torch.stack([Q[idxs[i][path[i]]] for i in range(Tn)]).cpu().numpy(), L)
    return best[1], best[2]


def fit(fr: np.ndarray, am: np.ndarray, mk: np.ndarray, f0: np.ndarray, iters: int = 2000, lam_t: float = 1.0, lam_q: float = 0.05,
        dev: str = "cpu", fix_L: float | None = None, L_prior: float = 0.155, lam_L: float = 20.0, cb: tuple | None = None) -> dict:
    """有声フレームだけで当てはめる。重み = フレームの最大から 25dB 以内の倍音を主に(雑音の支配する谷を弱める)・Huber。
    返り値: q [T,K]・L・src [T,2]・err_db(重み付き RMS・倍音の位置)・err_db_top(最大から 25dB 以内の RMS)。"""
    vt = np.nonzero(mk.any(1))[0]
    B = basis().to(dev)
    frt = torch.from_numpy(fr[vt]).to(dev)
    amt = torch.from_numpy(am[vt]).to(dev)
    mkt = torch.from_numpy(mk[vt]).to(dev).double()
    f0t = torch.from_numpy(f0[vt].astype(np.float64)).to(dev)
    top = torch.where(mkt > 0, amt, torch.full_like(amt, -1e9)).amax(1, keepdim=True)
    wt = mkt * torch.sigmoid((amt - (top - 25.0)) / 3.0)
    T = len(vt)
    if cb is not None:
        q0, L0 = dp_init(fr, am, mk, f0, cb, dev=dev, fix_L=fix_L)
        if q0.shape[1] < K:
            q0 = np.pad(q0, ((0, 0), (0, K - q0.shape[1])))
        if fix_L is None:
            L_prior = L0
    q = (torch.from_numpy(q0).to(dev) if cb is not None else torch.zeros(T, K, dtype=torch.float64, device=dev)).clone().requires_grad_(True)
    cols = [torch.full((T,), math.log(1.5)), torch.full((T,), math.log(2000.0))] + ([torch.zeros(T)] if LOSS_SCALE else [])
    src = torch.stack(cols, 1).to(dev).requires_grad_(True)
    zero_gain = torch.zeros(T, dtype=torch.float64, device=dev)
    logL = torch.tensor(fix_L if fix_L is not None else math.log(L_prior), dtype=torch.float64, device=dev, requires_grad=fix_L is None)
    lr0 = 0.01 if cb is not None else 0.05
    params = [q, src] + ([logL] if fix_L is None else [])

    def pred_fn() -> torch.Tensor:
        m0 = model_db(q, logL, src, zero_gain, frt, B, f0t)
        gain = ((amt - m0) * wt).sum(1) / wt.sum(1).clamp(min=1e-9)
        return m0 + gain[:, None]
    opt = torch.optim.Adam(params, lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr0 / 10)
    ct = torch.from_numpy((np.diff(vt) == 1).astype(np.float64)).to(dev)
    for it in range(iters):
        opt.zero_grad()
        pred = pred_fn()
        data = (torch.nn.functional.huber_loss(pred, amt, reduction="none", delta=6.0) * wt).sum() / wt.sum()
        smooth = ((q[1:] - q[:-1]) ** 2).sum(1) * ct + 0.3 * ((src[1:] - src[:-1]) ** 2).sum(1) * ct
        reg = lam_t * smooth.sum() / max(1, T) + lam_q * (q ** 2).sum() / max(1, T)
        if fix_L is None:
            reg = reg + lam_L * (logL - math.log(L_prior)) ** 2
        (data + reg).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            if fix_L is None:
                logL.clamp_(math.log(0.11), math.log(0.21))
            src[:, 0].clamp_(math.log(0.4), math.log(5.0))
            src[:, 1].clamp_(math.log(300.0), math.log(12000.0))
            if LOSS_SCALE:
                src[:, 2].clamp_(math.log(0.3), math.log(3.0))
    with torch.no_grad():
        pred = pred_fn()
        e2 = (pred - amt) ** 2
        err = float(torch.sqrt((e2 * wt).sum() / wt.sum()))
        topm = mkt * (amt > top - 25.0).double()
        err_top = float(torch.sqrt((e2 * topm).sum() / topm.sum()))
    qf = np.full((len(mk), K), np.nan)
    qf[vt] = q.detach().cpu().numpy()
    return {"q": qf, "L": float(torch.exp(logL).detach()), "src": src.detach().cpu().numpy(), "err_db": err, "err_db_top": err_top, "voiced": vt}


def items(n_f: int = 4, n_m: int = 4) -> list[tuple[str, np.ndarray]]:
    from train_ddsp_vc import load48
    import eval_nvoc as E
    out = [("F:" + it["stem"][:8], it["x"][: 6 * SR].astype(np.float64)) for it in E.held_items()[:n_f]]
    vc = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
    for s in ("p245", "p251", "p298", "p226")[:n_m]:
        import librosa
        x, _ = librosa.load(str(vc / s / f"{s}_010.wav"), sr=SR)
        x, _ = librosa.effects.trim(x, top_db=35)
        out.append(("M:" + s, x[: 6 * SR].astype(np.float64)))
    return out


def dct24_traj(x: np.ndarray) -> np.ndarray:
    import nvoc as N
    import zsvc as Z
    front = Z.ZSVC().float().eval()
    seg = torch.cat([torch.zeros(N.WIN - N.HOP, dtype=torch.float32), torch.from_numpy(x.astype(np.float32))])[None]
    with torch.no_grad():
        mel = N.NVoc.mel_ctx(front, seg)
        c = torch.einsum("km,bmt->bkt", front.dct.float(), mel.float())[0, 1:25].T.numpy()
    return c[::2]


def g0g1(dev: str) -> dict:
    import pyworld
    import librosa
    rep: dict = {"g0": [], "g1": []}
    cb = codebook(dev=dev)
    for name, x in items():
        f0, _ = harvest(x)
        fr, am, mk = harmonic_obs(x, f0)
        r = fit(fr, am, mk, f0, dev=dev, cb=cb)
        rep["g0"].append({"utt": name, "err_db": round(r["err_db"], 2), "err_db_top": round(r["err_db_top"], 2), "L_cm": round(100 * r["L"], 2), "voiced_frames": int(len(r["voiced"]))})
        print("G0", rep["g0"][-1], flush=True)
        x16 = librosa.resample(x, orig_sr=SR, target_sr=16000)
        fw, tw = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=900, frame_period=HOP / SR * 1000)
        sp = pyworld.cheaptrick(x16, fw, tw, 16000)
        ap = pyworld.d4c(x16, fw, tw, 16000)
        res = {}
        for st in (0, -4, 4):
            y16 = pyworld.synthesize(fw * 2 ** (st / 12), sp, ap, 16000, HOP / SR * 1000)
            y = librosa.resample(y16, orig_sr=16000, target_sr=SR)[: len(x)]
            fy, _ = harvest(y)
            fry, amy, mky = harmonic_obs(y, fy)
            res[st] = (fit(fry, amy, mky, fy, dev=dev, fix_L=math.log(r["L"]), cb=cb), dct24_traj(y), fy)
        base_q, base_c, base_f = res[0]
        out = {"utt": name}
        for st in (-4, 4):
            q1, c1, f1 = res[st]
            n = min(len(base_q["q"]), len(q1["q"]))
            v = ~np.isnan(base_q["q"][:n, 0]) & ~np.isnan(q1["q"][:n, 0])
            nat_q = np.nanstd(base_q["q"][:n][v], 0)
            dq = np.sqrt(((base_q["q"][:n][v] - q1["q"][:n][v]) ** 2).mean(0)) / np.maximum(nat_q, 1e-6)
            m = min(len(base_c), len(c1), n)
            vv = v[:m]
            nat_c = base_c[:m][vv].std(0)
            dc = np.sqrt(((base_c[:m][vv] - c1[:m][vv]) ** 2).mean(0)) / np.maximum(nat_c, 1e-6)
            out[f"q_move_over_nat_{st:+d}"] = round(float(np.median(dq)), 3)
            out[f"dct24_move_over_nat_{st:+d}"] = round(float(np.median(dc)), 3)
        rep["g1"].append(out)
        print("G1", out, flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/artic_inv/g0g1.json"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    rep = g0g1(dev)
    e = [r["err_db"] for r in rep["g0"]]
    rep["summary"] = {"g0_err_db_F": round(float(np.mean([r["err_db"] for r in rep["g0"] if r["utt"].startswith("F")])), 2),
                      "g0_err_db_M": round(float(np.mean([r["err_db"] for r in rep["g0"] if r["utt"].startswith("M")])), 2),
                      "g1_q_move_median": round(float(np.median([v for r in rep["g1"] for k, v in r.items() if k.startswith("q_move")])), 3),
                      "g1_dct24_move_median": round(float(np.median([v for r in rep["g1"] for k, v in r.items() if k.startswith("dct24_move")])), 3)}
    print(json.dumps(rep["summary"], ensure_ascii=False), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
