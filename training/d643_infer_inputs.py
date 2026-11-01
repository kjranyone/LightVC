"""D643: 推論(製品の Rust の条件・dev2 / dev3 の 360 本)で内容の網の出力が表の引きより暗い原因を、入力を 1 つずつ学習の分布の値に替えて切り分ける(描画なし・行の形の高域 − 低域の dB)。
C2N = C2 を学習の C2 の平均のベクトル(定数)に・LFN = log f0 を学習の有声の中央値との差だけ一律にずらす(定数)・BOTH = 両方。学習の値は学習と同じ取り出しの 30 バッチ。規則は results/conv_p0/d643/prereg.yaml。
    uv run python d643_infer_inputs.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from d631_world_swap import C, ROOT, SETS, keys
from d634_artic_render import DCT, PRE
from d637_envnet_eval import DC, net, rdir

OUT = ROOT / "results/conv_p0/d643"


def train_stats() -> dict:
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_ppgvoc as P
    import train_rvoc as TR
    import c1_content as CC
    dev = "cuda"
    TR.SEG = 33600 * 4
    TR.T = TR.SEG // N.HOP
    SEG, T = TR.SEG, TR.T
    held = P.e2e_held()
    rows, spks, _ = P.load_rows({h[0] for h in held}, True)
    spk_idx = {s: i for i, s in enumerate(spks)}
    P.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    sm = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to(dev)
    dl = iter(torch.utils.data.DataLoader(TR.DS(rows, rows, 643, spk_idx=spk_idx, ctx=T1.PRE, sexpert=0.5), batch_size=8, num_workers=8))
    c2s, lfs = [], []
    for _ in range(30):
        bt = [t.to(dev) for t in next(dl)]
        xa, f0, env, xl = bt[:4]
        xp = bt[5] if len(bt) > 5 else xa
        with torch.no_grad():
            ct = front(xp, f0, env)
            xc = torch.cat([xl, xp[:, PA.A:PA.A + SEG]], 1)
            c2 = P.cvin_frames(xc, T)
        sp = ct[:, 0] > 0.05
        c2s.append(c2.permute(0, 2, 1)[sp].double().mean(0).cpu().numpy())
        v = ct[:, 26] > 0.5
        lfs.append(ct[:, 25][v].cpu().numpy())
    return {"c2_mean": np.mean(c2s, 0).tolist(), "lf_med": float(np.median(np.concatenate(lfs)))}


def hilo(rows: np.ndarray, m: np.ndarray) -> float:
    import librosa
    fc = librosa.mel_frequencies(130, fmin=0.0, fmax=24000.0)[1:-1]
    hi = fc >= 4000
    lm = (rows[m] * 10) @ DCT[1:25]
    return float((20 / np.log(10)) * (lm[:, hi].mean(1) - lm[:, ~hi].mean(1)).mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ts = train_stats()
    torch.set_num_threads(4)
    lf_inf = []
    data = []
    for s in SETS:
        for k in C:
            for key in keys(s):
                rc = np.fromfile(rdir(s, k) / f"{key}.rcond16", "<f2").reshape(-1, DC + 24).astype(np.float32)
                data.append((s, k, key, rc))
                v = rc[PRE:PRE + 1600, 26] > 0.5
                lf_inf.append(rc[PRE:PRE + 1600, 25][v])
    dlf = ts["lf_med"] - float(np.median(np.concatenate(lf_inf)))
    c2m = np.array(ts["c2_mean"], np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a.T))[None]
    res = {a: {k: [] for k in C} for a in ("TAB", "EN", "C2N", "LFN", "BOTH")}
    for s, k, key, rc in data:
        m = rc[PRE:PRE + 1600, 26] > 0.5
        lf2 = np.where(rc[:, 26:27] > 0.5, rc[:, 25:26] + dlf, rc[:, 25:26]).astype(np.float32)
        c2n = np.broadcast_to(c2m, (len(rc), len(c2m))).astype(np.float32)
        res["TAB"][k].append(hilo(rc[PRE:PRE + 1600, 1:25], m))
        for a, lf_, c2_ in (("EN", rc[:, 25:26], rc[:, 231:DC]), ("C2N", rc[:, 25:26], c2n), ("LFN", lf2, rc[:, 231:DC]), ("BOTH", lf2, c2n)):
            with torch.no_grad():
                y = net()(t(rc[:, 31:231]), t(rc[:, 0:1]), t(lf_), t(rc[:, 26:27]), t(rc[:, 1:25]), t(c2_))[0].numpy().T
            res[a][k].append(hilo(y[PRE:PRE + 1600], m))
    med = {a: {k: round(float(np.median(v)), 3) for k, v in r.items()} for a, r in res.items()}
    gap = {k: med["TAB"][k] - med["EN"][k] for k in C}
    rec = {a: {k: round((med[a][k] - med["EN"][k]) / gap[k], 3) if abs(gap[k]) > 1e-6 else None for k in C} for a in ("C2N", "LFN", "BOTH")}
    rep = {"train": {"lf_med": ts["lf_med"]}, "lf_shift": round(dlf, 4), "hilo_median_db": med, "gap_TAB_minus_EN_db": {k: round(v, 3) for k, v in gap.items()}, "recovered_frac": rec}
    rep["cause"] = {a: bool(all(rec[a][k] is not None and rec[a][k] >= 0.5 for k in C)) for a in ("C2N", "LFN")}
    (OUT / "result.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
