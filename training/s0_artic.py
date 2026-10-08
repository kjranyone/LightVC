"""Artic-A2 Step 0(学習なし)の実行器。判定基準は current/artic_a2.md §8 で事前固定。結果は results/artic_s0/<item>.json。

  s01: 因果 LPC 分析合成の可逆性(float64/float32)・未来不変性(残差・LAR)
  s02: LAR の遅さ(変調スペクトル・codec 潜在との比較)と、残差=元係数・合成=低域通過 LAR での再合成
  s03: ピッチ盲性 (a) 話者内中心化 log f0 の予測 R²(線形・MLP・話者群 5 分割) (b) 包絡ピークの倍音張り付き率(CheapTrick 比)
  s05: 残差 TD-PSOLA の往復劣化(+4/+8/+12 半音)・片道の f0 精度・先読み実測(女声 held24・男声 VCTK)
  s06: 耳バッテリー(会話・喘ぎ・囁き)での安定性・可逆性
  s07: 遅延台帳
  s07b: 因果パイプライン(因果 f0・因果マーク・出力遅延 D の PSOLA)の D 掃引(開発集合)→ D*(§8.1)
  s05b: 検査集合(VCTK 男声 15 人 × 2 発話)で D* の因果パイプライン(§8.1・1 回だけ)
  s07c: 方式変更後(残差の再標本化シフト RRPS)の D 掃引(開発集合・基準は §8.1 と同一)→ D*

S0-3(a) の対象は「話者内で中心化した log f0」(実行前に固定: 話者間の生理的相関=声道長と声の高さは漏れではないため)。
生の log f0 と CheapTrick ケプストラムの R² は参考として併記する。

    CUDA_VISIBLE_DEVICES= uv run python s0_artic.py s01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
import artic_dsp as D_
from train_d1 import build_index
from train_cfmys import F0FIX, F0_FPS
from train_dec2 import load48

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/artic_s0"
ORDERS = (16, 24, 32, 48)
ANCHOR = {"logmel": 0.195, "mrstft": 1.02}
YIN_VOI = 0.25
YIN_VOI_CONT = 0.45
LA = 480


def held24(max_sec: float = 8.0):
    pairs, lats, held = build_index(0)
    out = []
    for s in held:
        c = sorted(p for p in pairs if p.parent.name == s)
        if not c:
            continue
        f = c[0]
        d = torch.load(f, map_location="cpu", weights_only=False)
        x = load48(d["path"]).astype(np.float64)[: int(max_sec * D.SR)]
        f0 = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu", weights_only=False)["f0"].numpy()
        out.append({"stem": f.stem, "x": x, "f0": f0.astype(np.float64), "spk": s, "path": d["path"], "lat": lats.get(f.stem)})
    return out


_M = None


def spec_metrics(y: np.ndarray, x: np.ndarray) -> dict:
    global _M
    from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
    if _M is None:
        _M = [build_mel(nf, hm, nm) for nf, hm, nm in MEL_SPECS]
    n = min(len(y), len(x))
    Y = torch.from_numpy(y[:n]).float().view(1, 1, -1)
    X = torch.from_numpy(x[:n]).float().view(1, 1, -1)
    with torch.no_grad():
        return {"logmel": float(logmel_l1(Y, X, _M)), "mrstft": float(mrstft(Y, X))}


def snr_db(x: np.ndarray, y: np.ndarray) -> float:
    return float(10 * np.log10((x ** 2).sum() / max(((x - y) ** 2).sum(), 1e-300)))


def f0_times(f0: np.ndarray) -> np.ndarray:
    return np.arange(len(f0)) / F0_FPS


def s01() -> dict:
    from ship_check import future_invariance
    items = held24()
    rep = {"snr_db": {}, "future_invariance": {}}
    for p in ORDERS:
        s64, s32 = [], []
        for it in items:
            x = it["x"]
            _, a64, e64 = D.analyze(x, p, np.float64)
            s64.append(snr_db(x, D.synthesize(e64, a64)))
            _, a32, e32 = D.analyze(x, p, np.float32)
            s32.append(snr_db(x, D.synthesize(e32, a32).astype(np.float64)))
        rep["snr_db"][p] = {"float64_min": round(min(s64), 1), "float64_median": round(float(np.median(s64)), 1),
                            "float32_min": round(min(s32), 1), "float32_median": round(float(np.median(s32)), 1)}
        print("S0-1 order", p, rep["snr_db"][p], flush=True)
    for p in (24, 48):
        la_e = future_invariance(lambda x: torch.from_numpy(D.analyze(x.numpy().astype(np.float64), p)[2]),
                                 hop=1, n=96000, sr=D.SR)
        la_l = future_invariance(lambda x: torch.from_numpy(D.lar_frames(x.numpy().astype(np.float64), p).T.copy()),
                                 hop=D.H, n=96000, sr=D.SR)
        rep["future_invariance"][p] = {"residual": str(la_e), "lar_frames": str(la_l)}
        print("S0-1 invariance", p, rep["future_invariance"][p], flush=True)
    worst32 = min(v["float32_min"] for v in rep["snr_db"].values())
    rep["criterion"] = "float32 で SNR ≥ 60dB・先読み 0"
    rep["verdict"] = {p: ("PASS" if rep["snr_db"][p]["float32_min"] >= 60 else "FAIL") for p in ORDERS}
    rep["worst_float32_min_snr"] = worst32
    return rep


def mod_fraction(tr: np.ndarray, fs: float, bands: dict) -> dict:
    t = tr - tr.mean(0, keepdims=True)
    P = np.abs(np.fft.rfft(t, axis=0)) ** 2
    f = np.fft.rfftfreq(t.shape[0], 1 / fs)
    tot = P[1:].sum(0)
    w = t.var(0)
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f > lo) & (f <= hi)
        frac = P[m].sum(0) / np.maximum(tot, 1e-30)
        out[name] = float((frac * w).sum() / max(w.sum(), 1e-30))
    return out


def s02() -> dict:
    items = held24()
    bands = {">10Hz": (10, 1e9), ">15Hz": (15, 1e9), "15-50Hz": (15, 50)}
    rep = {"modulation": {}, "recon": {}, "anchor": ANCHOR}
    zb = []
    for it in items:
        if it["lat"] is not None:
            z = torch.load(it["lat"], map_location="cpu", weights_only=False)["z"].float().numpy()
            zb.append(mod_fraction(z[: int(len(it["x"]) / D.SR * 100)].astype(np.float64), 100.0,
                                   {"15-50Hz": (15, 50), ">15Hz": (15, 1e9)}))
    rep["modulation"]["ys1_latent_z"] = {k: round(float(np.mean([m[k] for m in zb])), 4) for k in zb[0]}
    for p in ORDERS:
        mods, rec = [], {}
        for it in items:
            x = it["x"]
            lar, a_sub, e = D.analyze(x, p)
            mods.append(mod_fraction(lar, D.SR / D.H, bands))
            for tag, (fc, causal) in {"identity": (None, True), "causal15": (15, True), "causal10": (10, True),
                                      "zerophase15": (15, False), "zerophase10": (10, False)}.items():
                lar2 = lar if fc is None else D.lar_lowpass(lar, fc, causal)
                y = D.synthesize(e, D.coef_schedule(lar2, len(x)))
                rec.setdefault(tag, []).append(spec_metrics(y, x))
        rep["modulation"][f"lar_p{p}"] = {k: round(float(np.mean([m[k] for m in mods])), 4) for k in bands}
        rep["recon"][f"p{p}"] = {tag: {m: round(float(np.mean([r[m] for r in v])), 4) for m in ("logmel", "mrstft")}
                                  for tag, v in rec.items()}
        print("S0-2 order", p, rep["modulation"][f"lar_p{p}"], rep["recon"][f"p{p}"], flush=True)
    rep["criterion"] = "因果 15Hz 低域通過で logmel ≤ 0.195(錨以下)"
    rep["verdict"] = {p: ("PASS" if rep["recon"][f"p{p}"]["causal15"]["logmel"] <= ANCHOR["logmel"] else "FAIL")
                      for p in ORDERS}
    print("S0-2 ys1 latent z modulation", rep["modulation"]["ys1_latent_z"], flush=True)
    return rep


def lpc_env_db(a: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    w = 2 * np.pi * freqs / D.SR
    k = np.arange(1, a.shape[-1] + 1)
    A = 1 + (a[..., None, :] * np.exp(-1j * w[:, None] * k[None, :])).sum(-1)
    return -20 * np.log10(np.abs(A) + 1e-12)


def lock_rate(env_db: np.ndarray, freqs: np.ndarray, f0: float) -> tuple[int, int]:
    from scipy.signal import find_peaks
    pk, _ = find_peaks(env_db, prominence=3.0)
    if len(pk) == 0:
        return 0, 0
    h = freqs[pk] / f0
    d = np.abs(h - np.round(h))
    return int((d <= 0.1).sum()), len(pk)


def s03() -> dict:
    import pyworld
    import librosa
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    pairs, lats, held = build_index(0)
    hs = set(held)
    spks = [s for s in sorted({p.parent.name for p in pairs}) if s not in hs][:40]
    rep = {"target": "話者内中心化 log f0(有声)・参考: 生 log f0・CheapTrick ケプストラム", "probe": {}, "lock": {}}
    lags = (0, 4, 8, 16, 32)
    feats = {f"lar_p{p}": [] for p in (24, 48)}
    feats["cheaptrick_cep24"] = []
    ys, groups = [], []
    for si, s in enumerate(spks):
        for f in sorted(p for p in pairs if p.parent.name == s)[:4]:
            d = torch.load(f, map_location="cpu", weights_only=False)
            x = load48(d["path"]).astype(np.float64)[: 8 * D.SR]
            f0r = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu", weights_only=False)["f0"].numpy().astype(np.float64)
            lar = {p: D.lar_frames(x, p) for p in (24, 48)}
            K = lar[24].shape[0]
            x16 = librosa.resample(x, orig_sr=D.SR, target_sr=16000)
            tf = f0_times(f0r)
            tf = tf[tf < len(x16) / 16000]
            f0c = f0r[: len(tf)]
            sp = pyworld.cheaptrick(x16, np.where(f0c > 0, f0c, 0.0), tf, 16000)
            cep = np.fft.irfft(np.log(sp + 1e-12), axis=1)[:, :24]
            idx = np.arange(32, K, 4)
            tc = (idx * D.H - D.W / 2) / D.SR
            fi = np.clip(np.round(tc * F0_FPS).astype(int), 0, len(f0c) - 1)
            fv = f0c[fi]
            m = fv > 0
            if m.sum() < 10:
                continue
            for p in (24, 48):
                feats[f"lar_p{p}"].append(np.concatenate([lar[p][idx - L] for L in lags], 1)[m])
            ci = np.clip(np.round(tc * F0_FPS).astype(int), 0, len(cep) - 1)
            lagi = [np.clip(ci - int(round(L * D.H / D.SR * F0_FPS)), 0, len(cep) - 1) for L in lags]
            feats["cheaptrick_cep24"].append(np.concatenate([cep[li] for li in lagi], 1)[m])
            ys.append(np.log(fv[m]))
            groups.append(np.full(m.sum(), si))
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    yc = y.copy()
    for s in np.unique(g):
        yc[g == s] -= y[g == s].mean()
    rep["n_frames"] = int(len(y))
    rep["n_speakers"] = int(len(np.unique(g)))
    for name, fl in feats.items():
        X = np.concatenate(fl)
        out = {}
        for tname, tgt in (("centered", yc), ("raw", y)):
            r_lin, r_mlp = [], []
            for tr, te in GroupKFold(5).split(X, tgt, g):
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                r_lin.append(r2_score(tgt[te], Ridge(alpha=10.0).fit(Xtr, tgt[tr]).predict(Xte)))
                mlp = MLPRegressor(hidden_layer_sizes=(256, 256), early_stopping=True, max_iter=100, random_state=0)
                r_mlp.append(r2_score(tgt[te], mlp.fit(Xtr, tgt[tr]).predict(Xte)))
            out[tname] = {"ridge_r2": round(float(np.mean(r_lin)), 4), "mlp_r2": round(float(np.mean(r_mlp)), 4)}
        rep["probe"][name] = out
        print("S0-3a", name, out, flush=True)
    freqs = np.arange(50, 5000, 5.0)
    bins = {"<250Hz": (0, 250), "250-400Hz": (250, 400), ">=400Hz": (400, 2000)}
    acc = {k: {"lpc24": [0, 0], "lpc48": [0, 0], "cheaptrick": [0, 0]} for k in bins}
    for it in held24():
        x, f0r = it["x"], it["f0"]
        x16 = librosa.resample(x, orig_sr=D.SR, target_sr=16000)
        tf = f0_times(f0r)
        tf = tf[tf < len(x16) / 16000]
        f0c = f0r[: len(tf)]
        sp = pyworld.cheaptrick(x16, np.where(f0c > 0, f0c, 0.0), tf, 16000)
        spf = np.fft.rfftfreq((sp.shape[1] - 1) * 2, 1 / 16000)
        for p in (24, 48):
            lar = D.lar_frames(x, p)
            a = D.k_to_a(D.lar_to_k(lar))
            for k in range(8, lar.shape[0], 8):
                tc = (k * D.H - D.W / 2) / D.SR
                fi = int(round(tc * F0_FPS))
                if fi >= len(f0c) or f0c[fi] <= 0:
                    continue
                f0 = f0c[fi]
                bn = next(b for b, (lo, hi) in bins.items() if lo <= f0 < hi)
                l, n = lock_rate(lpc_env_db(a[k], freqs), freqs, f0)
                acc[bn][f"lpc{p}"][0] += l
                acc[bn][f"lpc{p}"][1] += n
                if p == 24:
                    envc = 10 * np.log10(np.interp(freqs, spf, sp[fi]) + 1e-12)
                    l2, n2 = lock_rate(envc, freqs, f0)
                    acc[bn]["cheaptrick"][0] += l2
                    acc[bn]["cheaptrick"][1] += n2
    for bn, d in acc.items():
        rep["lock"][bn] = {k: {"rate": round(v[0] / max(v[1], 1), 3), "n_peaks": v[1]} for k, v in d.items()}
    rep["lock_chance"] = 0.2
    print("S0-3b", json.dumps(rep["lock"], ensure_ascii=False), flush=True)
    rep["criterion"] = "(a) 中心化 log f0 の MLP R² ≤ 0.1 かつ (b) 倍音張り付き率が CheapTrick と同程度(差 ≤ 0.10)"
    verdict = {}
    for p in (24, 48):
        a_ok = rep["probe"][f"lar_p{p}"]["centered"]["mlp_r2"] <= 0.1
        b_ok = all(rep["lock"][bn][f"lpc{p}"]["rate"] - rep["lock"][bn]["cheaptrick"]["rate"] <= 0.10
                   for bn in bins if rep["lock"][bn]["cheaptrick"]["n_peaks"] > 0)
        verdict[p] = {"a": a_ok, "b": b_ok, "PASS": a_ok and b_ok}
    rep["verdict"] = verdict
    return rep


def vctk_male(n: int = 8):
    import glob
    fs = []
    for spk in ("p226", "p227", "p232"):
        fs += sorted(glob.glob(str(ROOT / f"data/vctk/VCTK-Corpus/VCTK-Corpus/wav48/{spk}/*.wav")))[:3]
    return fs[:n]


def harvest_f0(x: np.ndarray):
    import pyworld
    import librosa
    x16 = librosa.resample(x, orig_sr=D.SR, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=900, frame_period=5.0)
    return pyworld.stonemask(x16, f0, t, 16000), t


def s05(order: int = 24) -> dict:
    rep = {"order": order, "female_held24": {}, "male_vctk": {}, "lookahead_ms": {}}
    sets = {"female_held24": [(it["x"], f0_times(it["f0"]), it["f0"]) for it in held24()]}
    males = []
    for p in vctk_male():
        x = load48(p).astype(np.float64)[: 8 * D.SR]
        f0, t = harvest_f0(x)
        males.append((x, t, f0))
    sets["male_vctk"] = males
    for sname, items in sets.items():
        for st in (4, 8, 12):
            r = 2 ** (st / 12)
            rt, err, octv, lvl = [], [], 0, []
            for x, tf, f0 in items:
                _, a_sub, e = D.analyze(x, order)
                segs = D.pitch_marks(e, tf, f0)
                e1, segs1 = D.psola_shift(e, segs, r)
                y1 = D.synthesize(e1, a_sub)
                e2, _ = D.psola_shift(e1, segs1, 1 / r)
                y2 = D.synthesize(e2, a_sub)
                rt.append(spec_metrics(y2, x))
                lvl.append(float(np.sqrt((y2 ** 2).mean() / max((x ** 2).mean(), 1e-12))))
                f0a, _ = harvest_f0(np.clip(y1, -1, 1))
                f0b, _ = harvest_f0(x)
                if (f0a > 0).sum() > 10 and (f0b > 0).sum() > 10:
                    d = 12 * np.log2(np.median(f0a[f0a > 0]) / np.median(f0b[f0b > 0])) - st
                    err.append(abs(d))
                    octv += int(abs(d) > 6)
            rep[sname][f"+{st}st"] = {"roundtrip_logmel": round(float(np.mean([m["logmel"] for m in rt])), 4),
                                      "roundtrip_mrstft": round(float(np.mean([m["mrstft"] for m in rt])), 4),
                                      "roundtrip_rms_ratio": round(float(np.median(lvl)), 3),
                                      "oneway_f0_abs_err_st_median": round(float(np.median(err)), 3) if err else None,
                                      "octave_errors": octv, "n": len(items)}
            print("S0-5", sname, f"+{st}st", rep[sname][f"+{st}st"], flush=True)
    x, tf, f0 = males[0]
    _, a_sub, e = D.analyze(x, order)
    segs = D.pitch_marks(e, tf, f0)
    rng = np.random.default_rng(0)
    for st in (4, 12):
        r = 2 ** (st / 12)
        base, _ = D.psola_shift(e, segs, r)
        worst = 0
        for c in np.linspace(0.3, 0.8, 6) * len(e):
            c = int(c)
            e2 = e.copy()
            e2[c:] = rng.standard_normal(len(e) - c) * e.std()
            out2, _ = D.psola_shift(e2, segs, r)
            diff = np.nonzero(np.abs(out2 - base) > 1e-9)[0]
            if len(diff):
                worst = max(worst, c - int(diff[0]))
        rep["lookahead_ms"][f"male_+{st}st"] = round(worst / D.SR * 1000, 2)
    rep["criterion"] = "女声 held24 の +8 半音往復で logmel ≤ 0.3"
    rep["verdict"] = "PASS" if rep["female_held24"]["+8st"]["roundtrip_logmel"] <= 0.3 else "FAIL"
    rep["note"] = "マークは harvest f0+残差ピーク(非因果)で固定・往復の復路は往路の出力マークを再利用(マーク検出誤差は含まない)。VC は上げ方向のみなので往復は保守的。"
    return rep


def vctk_male_test() -> list[str]:
    import glob
    info = (ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/speaker-info.txt").read_text().splitlines()[1:]
    males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    rest = [m for m in males if m not in ("p226", "p227", "p232")][::3]
    fs = []
    for spk in rest:
        fs += sorted(glob.glob(str(ROOT / f"data/vctk/VCTK-Corpus/VCTK-Corpus/wav48/{spk}/*.wav")))[:2]
    return fs


def causal_forward(x: np.ndarray, order: int, r: float, D: int):
    _, a_sub, e = D_.analyze(x, order)
    f0c, _ = D_.causal_yin(x, voi_max=YIN_VOI)
    f0k, _ = D_.causal_yin(x, voi_max=YIN_VOI_CONT)
    runs = D_.causal_marks(e, f0c, f0_cont=f0k)
    e1, segs1, info = D_.causal_psola(e, runs, r, D, f0=f0c)
    return a_sub, e1, segs1, info, f0c, runs


def pipeline_lookahead(x: np.ndarray, order: int, r: float, D: int, n_edit: int = 5) -> dict:
    rng = np.random.default_rng(0)
    a_sub, e1, _, _, _, _ = causal_forward(x, order, r, D)
    y0 = D_.synthesize(e1, a_sub)
    worst, sens = -10 ** 9, False
    for c in (np.linspace(0.35, 0.9, n_edit) * len(x)).astype(int):
        xe = x.copy()
        xe[c:] = rng.standard_normal(len(x) - c) * x.std()
        a2, e2, _, _, _, _ = causal_forward(xe, order, r, D)
        y1 = D_.synthesize(e2, a2)
        d = np.nonzero(np.abs(y0 - y1) > 1e-9)[0]
        if len(d) == 0:
            continue
        sens = True
        worst = max(worst, c - int(d[0]))
    return {"lookahead_samples": int(max(worst, 0)) if sens else None, "inconclusive": not sens,
            "lookahead_ms": round(float(max(worst, 0)) / D_.SR * 1000, 2) if sens else None}


def causal_eval(items: list, order: int, D: int, sts=(4, 8, 12)) -> dict:
    out = {}
    for st in sts:
        r = 2 ** (st / 12)
        rt, err, octv, lags = [], [], 0, []
        for x, _, _ in items:
            a_sub, e1, segs1, info, _, _ = causal_forward(x, order, r, D)
            y1 = D_.synthesize(e1, a_sub)
            e2, _ = D_.psola_shift(e1, [q for q in segs1 if len(q) >= 2], 1 / r)
            y2 = D_.synthesize(e2, a_sub)
            rt.append(spec_metrics(y2, x)["logmel"])
            if info["grain_lag_ms_median"] is not None:
                lags.append(info["grain_lag_ms_median"])
            f0a, _ = harvest_f0(np.clip(y1, -1, 1))
            f0b, _ = harvest_f0(x)
            if (f0a > 0).sum() > 10 and (f0b > 0).sum() > 10:
                d = 12 * np.log2(np.median(f0a[f0a > 0]) / np.median(f0b[f0b > 0])) - st
                err.append(abs(d))
                octv += int(abs(d) > 6)
        out[f"+{st}st"] = {"roundtrip_logmel": round(float(np.mean(rt)), 4),
                           "oneway_f0_abs_err_st_median": round(float(np.median(err)), 3) if err else None,
                           "octave_errors": octv, "n": len(items),
                           "grain_lag_ms_median": round(float(np.median(lags)), 2) if lags else None}
    return out


def dev_items() -> list:
    items = []
    for p in vctk_male():
        x = load48(p).astype(np.float64)[: 8 * D.SR]
        f0, t = harvest_f0(x)
        items.append((x, t, f0))
    return items


def s07b(order: int = 24) -> dict:
    items = dev_items()
    rep = {"order": order, "set": "dev: VCTK p226/p227/p232 × 8 発話", "sweep": {}, "voicing": {}}
    agree, match = [], []
    for x, t, f0 in items:
        _, _, e = D.analyze(x, order)
        f0c, _ = D.causal_yin(x, voi_max=YIN_VOI)
        tc = (np.arange(len(f0c)) * D.F0_HOP + D.F0_HOP - 1 - D.YIN_W / 2) / D.SR
        fh = np.interp(tc, t, f0) * (np.interp(tc, t, (f0 > 0).astype(float)) > 0.5)
        agree.append(float(((f0c > 0) == (fh > 0)).mean()))
        both = (f0c > 0) & (fh > 0)
        if both.any():
            match.append(float((np.abs(12 * np.log2(f0c[both] / fh[both])) < 1.0).mean()))
        f0k, _ = D.causal_yin(x, voi_max=YIN_VOI_CONT)
        runs = D.causal_marks(e, f0c, f0_cont=f0k)
        off = np.concatenate(D.pitch_marks(e, t, f0)) if D.pitch_marks(e, t, f0) else np.array([])
        cm = np.array([m for run in runs for m, _ in run])
        if len(off) and len(cm):
            dist = np.abs(cm[:, None] - off[None, :]).min(1)
            rep["voicing"].setdefault("causal_mark_within_0.5ms_of_offline", []).append(round(float((dist <= 24).mean()), 3))
    rep["voicing"]["voiced_agreement_vs_harvest"] = round(float(np.mean(agree)), 3)
    rep["voicing"]["f0_within_1st_vs_harvest"] = round(float(np.mean(match)), 3)
    print("S0-7b voicing", rep["voicing"], flush=True)
    for dms in (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0):
        Dn = int(round(dms * D.SR / 1000))
        row = causal_eval(items, order, Dn)
        row["future_invariance"] = pipeline_lookahead(items[0][0], order, 2 ** (8 / 12), Dn)
        rep["sweep"][f"{dms}ms"] = row
        print("S0-7b", f"D={dms}ms", json.dumps(row, ensure_ascii=False), flush=True)
    ref = 0.205
    dstar = None
    for k, row in rep["sweep"].items():
        r8 = row["+8st"]
        fi = row["future_invariance"]
        ok = (r8["oneway_f0_abs_err_st_median"] is not None and r8["oneway_f0_abs_err_st_median"] <= 0.5
              and r8["octave_errors"] == 0 and r8["roundtrip_logmel"] <= ref + 0.03
              and not fi["inconclusive"] and fi["lookahead_ms"] <= float(k[:-2]) + 1e-6)
        row["meets_dstar_rule"] = ok
        if ok and dstar is None:
            dstar = float(k[:-2])
    rep["criterion"] = "§8.1: D* = +8半音で f0誤差中央 ≤0.5半音・オクターブ誤り0・往復logmel ≤ 0.205+0.03・未来不変性の先読み ≤ D を満たす最小D。D* ≤ 15ms で S0-7 確定 PASS"
    rep["D_star_ms"] = dstar
    rep["verdict"] = "PASS" if dstar is not None and dstar <= 15 else "FAIL"
    return rep


def s05b(order: int = 24) -> dict:
    s7 = json.loads((OUT / "s07b.json").read_text())
    dstar = s7["D_star_ms"]
    assert dstar is not None, "S0-7b で D* が決まっていない(S0-5b は走らせない)"
    items = []
    for p in vctk_male_test():
        x = load48(p).astype(np.float64)[: 8 * D.SR]
        items.append((x, None, None))
    Dn = int(round(dstar * D.SR / 1000))
    rep = {"order": order, "D_star_ms": dstar, "set": f"test: VCTK 男声 15 人 × 2 発話 = {len(items)}", "files": vctk_male_test()}
    rep.update(causal_eval(items, order, Dn))
    r8 = rep["+8st"]
    rep["criterion"] = "§8.1: +8半音で往復 logmel ≤ 0.3・片道 f0 誤差中央 ≤ 0.5 半音・オクターブ誤り ≤ 2/30"
    rep["verdict"] = "PASS" if (r8["roundtrip_logmel"] <= 0.3 and r8["oneway_f0_abs_err_st_median"] is not None
                                and r8["oneway_f0_abs_err_st_median"] <= 0.5 and r8["octave_errors"] <= 2) else "FAIL"
    print("S0-5b", json.dumps({k: rep[k] for k in ("+4st", "+8st", "+12st")}, ensure_ascii=False), flush=True)
    return rep


def spec_metrics_lp(y: np.ndarray, x: np.ndarray, fc: float) -> dict:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(8, fc, fs=D_.SR, output="sos")
    return spec_metrics(np.ascontiguousarray(sosfiltfilt(sos, y)), np.ascontiguousarray(sosfiltfilt(sos, x)))


def psola_ref_lp(items: list, order: int, st: float, la: int = 0) -> float:
    r = 2 ** (st / 12)
    vals = []
    for x, t, f0 in items:
        _, a_sub, e = D_.analyze(x, order, la=la)
        segs = D_.pitch_marks(e, t, f0)
        e1, segs1 = D_.psola_shift(e, segs, r)
        e2, _ = D_.psola_shift(e1, [q for q in segs1 if len(q) >= 2], 1 / r)
        vals.append(spec_metrics_lp(D_.synthesize(e2, a_sub), x, 0.95 * D_.SR / 2 / r)["logmel"])
    return float(np.mean(vals))


def shift_metrics(x: np.ndarray, y: np.ndarray, st: float, f0x: np.ndarray) -> dict:
    f0y, _ = harvest_f0(np.clip(y, -1, 1))
    k = min(len(f0y), len(f0x))
    vx, vy = f0x[:k] > 0, f0y[:k] > 0
    both = vx & vy
    ok = np.zeros(k, bool)
    ok[both] = np.abs(12 * np.log2(f0y[:k][both] / f0x[:k][both]) - st) <= 1.0
    med = (12 * np.log2(np.median(f0y[vy]) / np.median(f0x[vx])) - st) if vy.sum() > 10 and vx.sum() > 10 else None
    return {"shift_ok": float(ok[vx].mean()) if vx.any() else None,
            "false_voicing": float(vy[~vx].mean()) if (~vx).any() else None, "median_err_st": med}


def rrps_forward(x: np.ndarray, order: int, r: float, D: int, la: int = LA):
    assert D >= la, "解析窓の先読み la は出力遅延 D の中に収める"
    _, a_sub, e = D_.analyze(x, order, la=la)
    f0p, _ = D_.causal_yin(x, voi_max=YIN_VOI_CONT)
    v = D_.voiced_known(len(x), f0p, D, D_.F0_HOP)
    e1, info = D_.rrps(e, f0p, r, D, voiced=v)
    return a_sub, e, e1, f0p, v, info


def rrps_lookahead(x: np.ndarray, order: int, r: float, D: int, n_edit: int = 5) -> dict:
    rng = np.random.default_rng(0)
    a_sub, _, e1, _, _, _ = rrps_forward(x, order, r, D)
    y0 = D_.synthesize(e1, a_sub)
    worst, sens = -10 ** 9, False
    for c in (np.linspace(0.35, 0.9, n_edit) * len(x)).astype(int):
        xe = x.copy()
        xe[c:] = rng.standard_normal(len(x) - c) * x.std()
        a2, _, e2, _, _, _ = rrps_forward(xe, order, r, D)
        y1 = D_.synthesize(e2, a2)
        d = np.nonzero(np.abs(y0 - y1) > 1e-9)[0]
        if len(d) == 0:
            continue
        sens = True
        worst = max(worst, c - int(d[0]))
    return {"lookahead_samples": int(max(worst, 0)) if sens else None, "inconclusive": not sens,
            "lookahead_ms": round(float(max(worst, 0)) / D_.SR * 1000, 2) if sens else None}


def rrps_eval(items: list, order: int, D: int, sts=(4, 8, 12)) -> dict:
    out = {}
    for st in sts:
        r = 2 ** (st / 12)
        rt, err, octv, sok, fv = [], [], 0, [], []
        for x, t, f0 in items:
            a_sub, e, e1, f0p, v, _ = rrps_forward(x, order, r, D)
            y1 = D_.synthesize(e1, a_sub)
            e2, _ = D_.rrps(e1, np.where(f0p > 0, f0p * r, 0.0), 1 / r, D, voiced=v)
            y2 = D_.synthesize(e2, a_sub)
            rt.append(spec_metrics_lp(y2, x, 0.95 * D_.SR / 2 / r)["logmel"])
            f0x = f0 if f0 is not None else harvest_f0(x)[0]
            m = shift_metrics(x, y1, st, f0x)
            if m["median_err_st"] is not None:
                err.append(abs(m["median_err_st"]))
                octv += int(abs(m["median_err_st"]) > 6)
            if m["shift_ok"] is not None:
                sok.append(m["shift_ok"])
            if m["false_voicing"] is not None:
                fv.append(m["false_voicing"])
        out[f"+{st}st"] = {"roundtrip_logmel_lp": round(float(np.mean(rt)), 4),
                           "oneway_f0_abs_err_st_median": round(float(np.median(err)), 3) if err else None,
                           "octave_errors": octv, "n": len(items),
                           "shift_ok": round(float(np.mean(sok)), 3), "false_voicing": round(float(np.mean(fv)), 3)}
    return out


def s07c(order: int = 24) -> dict:
    items = dev_items()
    rep = {"order": order, "method": "RRPS(残差の再標本化シフト・有声ゲート・WSOLA 型の跳び先整列)", "set": "dev: VCTK p226/p227/p232 × 8 発話",
           "sweep": {}}
    ref = psola_ref_lp(items, order, 8, la=LA)
    rep["roundtrip_ref_lp"] = {"value": round(ref, 4), "how": "非因果 PSOLA(harvest マーク)往路+マーク再利用の復路を、両信号とも 0.95·(SR/2)/ratio で低域通過してから logmel(RRPS は上げで 24k/ratio 以上を捨てるため、往復の比較は方式に依らずこの帯域で行う)"}
    print("S0-7c ref_lp", rep["roundtrip_ref_lp"], flush=True)
    rep["analysis_lookahead_ms"] = LA / D.SR * 1000
    for dms in (10.0, 12.5, 15.0):
        Dn = int(round(dms * D.SR / 1000))
        row = rrps_eval(items, order, Dn)
        row["future_invariance"] = rrps_lookahead(items[0][0], order, 2 ** (8 / 12), Dn)
        rep["sweep"][f"{dms}ms"] = row
        print("S0-7c", f"D={dms}ms", json.dumps(row, ensure_ascii=False), flush=True)
    dstar = None
    for k, row in rep["sweep"].items():
        r8 = row["+8st"]
        fi = row["future_invariance"]
        ok = (r8["oneway_f0_abs_err_st_median"] is not None and r8["oneway_f0_abs_err_st_median"] <= 0.5
              and r8["octave_errors"] == 0 and r8["roundtrip_logmel_lp"] <= ref + 0.03
              and not fi["inconclusive"] and fi["lookahead_ms"] <= float(k[:-2]) + 1e-6)
        row["meets_dstar_rule"] = ok
        if ok and dstar is None:
            dstar = float(k[:-2])
    rep["criterion"] = "§8.1 と同じ規則(往復の比較だけ帯域制限 logmel・基準は同じ測り方の PSOLA 参照 +0.03): D* = +8半音で f0誤差中央 ≤0.5半音・オクターブ誤り0・往復 ≤ ref+0.03・先読み ≤ D を満たす最小D"
    rep["D_star_ms"] = dstar
    rep["verdict"] = "PASS" if dstar is not None and dstar <= 15 else "FAIL"
    return rep


def s06() -> dict:
    eb = ROOT / "results/earbattery"
    clips = {cid: eb / f"s16/{cid}_source.wav" for cid in ("dlg_low", "dlg_mid", "moan_high", "whis_0", "whis_1", "whis_2")}
    rep = {}
    for cid, p in clips.items():
        x = load48(str(p)).astype(np.float64)
        row = {}
        for order in (24, 48):
            lar, a32, e32 = D.analyze(x, order, np.float32)
            k = D.lar_to_k(lar)
            y = D.synthesize(e32, a32).astype(np.float64)
            row[f"p{order}"] = {"max_abs_k": round(float(np.abs(k).max()), 6),
                                "frac_k_gt_0.999": round(float((np.abs(k) > 0.999).mean()), 6),
                                "lar_abs_max": round(float(np.abs(lar).max()), 3),
                                "snr_float32_db": round(snr_db(x, y), 1),
                                "residual_crest": round(float(np.abs(e32).max() / (np.sqrt((e32.astype(np.float64) ** 2).mean()) + 1e-12)), 2)}
        rep[cid] = row
        print("S0-6", cid, row, flush=True)
    rep["criterion"] = "不安定フィルタ 0(|k|<1)・float32 で SNR ≥ 60dB"
    rep["verdict"] = {o: ("PASS" if all(rep[c][f"p{o}"]["max_abs_k"] < 1 and rep[c][f"p{o}"]["snr_float32_db"] >= 60
                                        for c in clips) else "FAIL") for o in (24, 48)}
    return rep


def s07() -> dict:
    from scipy.signal import butter, group_delay
    s01r = json.loads((OUT / "s01.json").read_text()) if (OUT / "s01.json").exists() else {}
    s05r = json.loads((OUT / "s05.json").read_text()) if (OUT / "s05.json").exists() else {}
    fs = D.SR / D.H
    lag = {}
    for fc in (10, 15):
        b, a = butter(2, fc / (fs / 2))
        w, gd = group_delay((b, a), w=[3.0, 8.0], fs=fs)
        lag[f"lar_lowpass_{fc}Hz_group_delay_ms_at_3Hz"] = round(float(gd[0]) / fs * 1000, 1)
    psola = s05r.get("lookahead_ms", {})
    total = max(psola.values()) if psola else None
    return {"lpc_analysis_lookahead": s01r.get("future_invariance"), "psola_lookahead_ms": psola,
            "lar_lowpass_lag(遅れであって先読みではない)": lag, "total_lookahead_ms": total,
            "criterion": "合計先読み ≤ 15ms", "verdict": (None if total is None else ("PASS" if total <= 15 else "FAIL"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("item", choices=["s01", "s02", "s03", "s05", "s06", "s07", "s07b", "s05b", "s07c"])
    ap.add_argument("--order", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fn = {"s01": s01, "s02": s02, "s03": s03, "s05": lambda: s05(a.order), "s06": s06, "s07": s07,
          "s07b": lambda: s07b(a.order), "s05b": lambda: s05b(a.order), "s07c": lambda: s07c(a.order)}[a.item]
    rep = fn()
    (OUT / f"{a.item}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print("->", OUT / f"{a.item}.json", "verdict:", rep.get("verdict"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
