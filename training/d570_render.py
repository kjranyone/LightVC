"""D570(推論だけ・構音モデルの段 0): 男声 → 女声を固定の半音 k で、今の変換 OURS_k と Graillon 相当の錨 GU_k,α を開発の組に描画する。規則は OVERNIGHT の D570。
OURS_k = P57S0(Rust --no-f0-shift の書き出し・25 列以外は製品の書き出しと一致を確かめる)+ 有声の log f0 に k 半音・水準ずらし・操作者の遅い揺らぎ 0。
GU_k,α = WORLD(harvest 16 kHz・CheapTrick / D4C 48 kHz・5 ms)で f0 × 2^(k/12)・包絡と非周期性を周波数方向に一律の α で伸縮(sp'(f) = sp(f/α))。
出力 = <H>/<組>/d570/<版>/<pNNNN_tTTT>_<版>.wav(ピーク 0.9)。
    uv run python d570_render.py --sets dev2 dev3
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import soundfile as sf
import torch

import nvoc as N
import train_ppgvoc as P
import exc_h1 as X
from d273_data import BIN, job
from d274_evaldump import COM
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results/diag_ppgvoc57/product_002500_m0.pt"
PRE = 12.0
KS = (5, 7, 12)
ALPHAS = (1.00, 1.12, 1.20, 1.28, 1.36, 1.44)


def gu_one(args: tuple) -> int:
    src, od, k, ks = args
    x = load48(src).astype(np.float64)[: 8 * N.SR]
    fp = 1000.0 * N.HOP / N.SR
    f0, t = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=50.0, f0_ceil=800.0, frame_period=fp)
    sp = pw.cheaptrick(x, f0, t, N.SR)
    ap = pw.d4c(x, f0, t, N.SR)
    fq = np.linspace(0, N.SR / 2, sp.shape[1])
    n = 0
    for kk in ks:
        for a in ALPHAS:
            nm = f"GU{kk}_a{int(round(a * 100))}"
            o = od / nm / f"{k}_{nm}.wav"
            if o.exists():
                continue
            sp2 = np.stack([np.interp(fq / a, fq, r) for r in sp]) if a != 1.0 else sp
            ap2 = np.stack([np.interp(fq / a, fq, r) for r in ap]) if a != 1.0 else ap
            y = pw.synthesize(f0 * 2 ** (kk / 12), np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), N.SR, fp)
            o.parent.mkdir(parents=True, exist_ok=True)
            sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR, subtype="PCM_16")
            n += 1
    return n


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--sets", nargs="+", required=True)
    ap_.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap_.add_argument("--ckpt", default=str(CKPT))
    ap_.add_argument("--sfx", default="", help="OURS<k><sfx>(別の重みの版)・空でなければ GU は描かない")
    ap_.add_argument("--en_sub", default="d85_work", help="生成器に渡す表の登録の置き場(Rust の条件の書き出しは常に d85_work)")
    ap_.add_argument("--h1fix", type=float, default=0.0, help="D611: 有声の枠の表の包絡に目標の声域から f0 までの傾き σ dB/oct(0 = なし)")
    ap_.add_argument("--h1fix_taper", action="store_true")
    ap_.add_argument("--exc_h1", type=float, default=0.0, help="D618: 励起の第 1 倍音の重み dB(有声で f0 < 目標の声域・0 = なし)")
    a = ap_.parse_args()
    ks = tuple(a.ks)
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 0.0
    P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    import train_c1 as TC
    X.install()
    n0 = int(PRE * N.SR)
    for s in a.sets:
        S = P.H / s
        ED0, ED = P.H / "e2e" / f"eval_{s}_pre", P.H / "e2e" / f"eval_{s}_pre_nof0"
        ED.mkdir(parents=True, exist_ok=True)
        od = S / "d570"
        pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"]
                 if (S / "d467" / f"p{r['pi']:04d}_t{r['ti']:03d}.wav").exists() and ((ED0 / f"p{r['pi']:04d}_t{r['ti']:03d}.cond").exists() or not ED0.exists())]
        jobs = []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["--no-f0-shift", "-i", S / "d467" / f"{k}.wav"], ED / f"{k}.cond", ED / f"{k}.front"))
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        with Pool(6) as pl:
            ngu = 0 if a.sfx else sum(pl.imap_unordered(gu_one, [(S / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav", od, f"p{r['pi']:04d}_t{r['ti']:03d}", ks) for r in pairs]))
        worst = 0.0
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, lg = P.load_dump(ED, k)
            if ED0.exists():
                c1, lg1 = P.load_dump(ED0, k)
                n = min(len(c), len(c1))
                oth = [j for j in range(31) if j != 25]
                d_ = max(float(np.abs(c[:n, oth] - c1[:n, oth]).max()), float(np.abs(lg[:n] - lg1[:n]).max()))
                worst = max(worst, d_)
                assert d_ < 1e-4, (k, d_)
            pre = c[:2400, 0]
            mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
            c = c.copy()
            c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
            vv = c[:, 26] > 0.5
            cvf = None
            en = P.load_enroll(S / a.en_sub / f"en_{r['ti']:03d}")
            P.H1FIX.update(on=a.h1fix > 0, sigma=a.h1fix, taper=a.h1fix_taper, ft=float(np.exp(json.loads((S / a.en_sub / f"en_{r['ti']:03d}" / "enroll.json").read_text())["mu"])))
            X.EXH1.update(on=a.exc_h1 > 0, gain_db=a.exc_h1, ft=P.H1FIX["ft"])
            for kk in ks:
                o = od / f"OURS{kk}{a.sfx}" / f"{k}_OURS{kk}{a.sfx}.wav"
                if o.exists():
                    continue
                if cvf is None:
                    cvf = P.cvin_frames(torch.from_numpy(TC.prime(load48(S / "d467" / f"{k}.wav").astype(np.float32)))[None].to(dev), len(lg))
                ck = c.copy()
                ck[:, 25] = np.where(vv, c[:, 25] + kk * np.log(2) / 12, c[:, 25])
                y = P.render(g, ck, lg, en, dev, T_op=P.op_tab(S, k), cvf=cvf)[n0:n0 + 8 * N.SR]
                o.parent.mkdir(parents=True, exist_ok=True)
                sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print(s, "pairs", len(pairs), "gu_written", ngu, "max_abs_diff_other_cols", worst, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
