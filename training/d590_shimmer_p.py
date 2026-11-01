"""D590(学習なし): 高さに依らない shimmer(shimmer_p)。
d109 の shimmer = 有声の 5 ms 格子で Hann 10 ms 窓(振幅に窓 = 力に窓²)の音量の隣の |Δ dB| の中央値。窓に入る周期が少ない低い声(+5 半音の男声 ≈ 1.5 周期)では、窓とパルスの位相のずれが音量の揺れに見える。
shimmer_p = 同じ格子・同じ有声・同じ正規化で、力 x² に長さ m·T0(T0 = SR / f0・m = max(2, round(0.010·f0)))の Hann を掛けた和の隣の |Δ dB| の中央値。
Hann は長さ / m ずらしの和が一定 = 周期的な信号では窓の位相に依らない(窓の長さは 8〜13 ms で旧の 10 ms と同程度)。
段: synth = 合成の母音(倍音の和・f0 100〜440 Hz・±1 半音のゆっくりした揺れ・一定の音量 / 15 Hz 以下の音量の揺れ)で単体検査。
    pool = d109 と同じ選び方の本物の女声(旧 shimmer が d109 のキャッシュと一致するかを確かめる)。
    sets = 耳のページの音(ear:pass / ear:fail)と候補の組(d570 の版)。判定 = d109 と同じ規則(外れ ≤ 15% かつ 中央値が本物の [p15, p85])を旧と新の両方で。
    uv run python d590_shimmer_p.py --s <scratch> --tag d590_dev2 --sets 'OURS5RY58G=<dir>/*_OURS5RY58G.wav:none' ... [--ear] [--synth]
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
SEC = 8.0
EAR = ROOT / "results/earbattery"


def ear_sets() -> list:
    out = []

    def page(name: str, conds: dict, trials: list | None = None) -> None:
        k = json.loads((EAR / name / "_key_聴取後に開く.json").read_text())["trials"]
        for tr, v in sorted(k.items()):
            if trials is not None and tr not in trials:
                continue
            for L, c in v["map"].items():
                if c in conds:
                    f = EAR / name / tr / f"{L}.wav"
                    if f.exists():
                        out.append((f"{name}:{c}", str(f), conds[c]))
            r = EAR / name / tr / "R.wav"
            if r.exists():
                out.append(("REAL_R", str(r), "real"))

    page("gabi_ab1", {c: "fail" for c in ("dynamic_periodicity", "base_product", "teacher_f0", "all_three_real")})
    page("warp_ab2", {c: "fail" for c in ("warp1.3_k30_g0.7", "mix0.5", "soft_mix0")})
    page("prodcand_ab1", {c: "fail" for c in ("cand_gamma05_color_ft360k", "cand_gamma07_color_ft360k")})
    page("env_ab1", {"table_envelope": "fail", "real_envelope": "pass"})
    page("tabenv_ab1", {"table_means_plus_real_dynamics": "pass", "same_utt_unit_means": "fail", "table_postfilter13": "fail", "table_soft": "fail"}, ["te_02"])
    page("fcopy_ab1", {"ora_teacher_analysis": "pass"}, ["fc_02", "fc_03"])
    for f in sorted((EAR / "ff_v3_ear").glob("t*_out.wav")):
        out.append(("ff_v3_ear:P43OP", str(f), "pass"))
    for f in sorted((EAR / "ff_v3_ear").glob("t*_tgt.wav")) + sorted((EAR / "ff_v3_ear").glob("t*_in.wav")):
        out.append(("REAL_ff", str(f), "real"))
    seen, uniq = set(), []
    for g, f, e in out:
        if f not in seen:
            seen.add(f)
            uniq.append((g, f, e))
    return uniq


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    return float(gt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_pool", type=int, default=150)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--ear", action="store_true")
    ap.add_argument("--synth", action="store_true")
    a = ap.parse_args()
    import soundfile as sf
    import f0est as FE
    import nvoc as N
    from d6_unit_cov import load48
    S = Path(a.s)
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())
    HOP, SR = N.HOP, N.SR

    def f0track(x: np.ndarray) -> np.ndarray:
        n = len(x) // HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * HOP])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        f = np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12))
        return np.where(vo[:m] & (c >= 0.5), f, 0.0)

    def active(x: np.ndarray) -> np.ndarray:
        n = len(x) // HOP
        e = 10 * np.log10((x[: n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-12)
        return e > e.max() - 40

    def stats(x: np.ndarray) -> dict:
        x = x[: int(SEC * SR)].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        f = f0track(x)
        act = active(x)[: len(f)]
        f = f[: len(act)]
        v = f > 0
        fw = np.lib.stride_tricks.sliding_window_view(np.r_[np.zeros(HOP, np.float32), x], 2 * HOP)[::HOP][: len(act)]
        ed = 10 * np.log10((fw * np.hanning(2 * HOP)).__pow__(2).mean(1) + 1e-12)
        vv = v[: len(ed)]
        both = vv[1:] & vv[:-1]
        de = np.abs(np.diff(ed))[both]
        x2 = x.astype(np.float64) ** 2
        ep = np.full(len(ed), np.nan)
        for t in np.nonzero(vv)[0]:
            T0 = SR / f[t]
            L = max(2, int(round(0.010 * f[t]))) * T0
            n = np.arange(int(np.ceil(-L / 2)), int(np.floor(L / 2)) + 1)
            w = 0.5 * (1 + np.cos(2 * np.pi * n / L))
            idx = t * HOP + n
            ok = (idx >= 0) & (idx < len(x2))
            ep[t] = 10 * np.log10((w[ok] * x2[idx[ok]]).sum() / L + 1e-12)
        dp = np.abs(np.diff(ep))[both]
        dp = dp[np.isfinite(dp)]
        return {"shimmer": float(np.median(de)) if len(de) else float("nan"), "shimmer_p": float(np.median(dp)) if len(dp) else float("nan"),
                "f0_med": float(np.median(f[v])) if v.any() else float("nan"), "vfrac": float(v.sum() / max(act.sum(), 1))}

    rep: dict = {"tag": a.tag}
    if a.synth:
        rng = np.random.default_rng(590)
        t = np.arange(int(SEC * SR)) / SR
        fm, bw = np.array([700.0, 1200.0, 2600.0, 3500.0]), np.array([90.0, 110.0, 160.0, 200.0])
        syn = {}
        from scipy.signal import butter, sosfiltfilt
        nz = rng.standard_normal(len(t))
        pad = 2 * SR
        am = sosfiltfilt(butter(4, 15, fs=SR, output="sos"), rng.standard_normal(len(t) + 2 * pad))[pad:-pad]
        am = (am - am.mean()) / am.std() * 2.0
        for f0 in (100, 130, 160, 200, 260, 330, 440):
            ft = f0 * 2 ** (np.sin(2 * np.pi * 0.5 * t) / 12)
            ph = 2 * np.pi * np.cumsum(ft) / SR
            y = np.zeros_like(t)
            for h in range(1, int(20000 / (f0 * 1.06))):
                hf = h * ft
                g = np.ones_like(hf)
                for F, B in zip(fm, bw):
                    g = g / np.sqrt((1 - (hf / F) ** 2) ** 2 + (hf * B / F ** 2) ** 2)
                y += g * np.cos(h * ph) / h ** 0.5
            for kind, yy in (("const", y), ("am", y * 10 ** (am / 20))):
                yy = yy / np.sqrt((yy ** 2).mean()) * 0.05 + 0.05 * 10 ** (-60 / 20) * nz
                syn[f"{kind}_{f0}"] = stats(yy.astype(np.float32))
                print("synth", kind, f0, {k: round(v, 3) for k, v in syn[f"{kind}_{f0}"].items()}, flush=True)
        cp = [syn[f"const_{f}"]["shimmer_p"] for f in (100, 130, 160, 200, 260, 330, 440)]
        amp = [syn[f"am_{f}"]["shimmer_p"] for f in (100, 130, 160, 200, 260, 330, 440)]
        rep["synth"] = syn
        rep["synth_check"] = {"const_max": round(max(cp), 3), "am_ratio": round(max(amp) / min(amp), 3), "pass": bool(max(cp) <= 0.10 and max(amp) / min(amp) <= 1.3)}
        print("synth_check", rep["synth_check"], flush=True)
        (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        if not rep["synth_check"]["pass"]:
            return 1

    cpath = S / "d590_cache.json"
    cache = json.loads(cpath.read_text()) if cpath.exists() else {}
    old = json.loads((S / "d109_cache.json").read_text()) if (S / "d109_cache.json").exists() else {}

    def cstats(key: str, xf) -> dict:
        if key not in cache:
            cache[key] = stats(xf())
        return cache[key]

    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(109).shuffle(pool)
    real, par = [], []
    for spk in pool:
        if len(real) >= a.n_pool:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or any(sf.info(str(w)).samplerate < 44100 for w in ws[:3]):
            continue
        key = f"pool|{spk}|{ws[len(ws) // 2].name}"
        if key not in cache:
            x = load48(ws[len(ws) // 2])
            if len(x) < 3 * SR:
                continue
            rest = sum(sf.info(str(w)).frames * SR / sf.info(str(w)).samplerate for w in [w for w in ws if w != ws[len(ws) // 2]][:6])
            if rest < 8 * SR:
                continue
        r = cstats(key, lambda: load48(ws[len(ws) // 2]))
        real.append(r)
        o = old.get(f"2|{key}")
        if o is not None:
            par.append(abs(o["shimmer"] - r["shimmer"]))
        if len(real) % 50 == 0:
            print("pool", len(real), flush=True)
            cpath.write_text(json.dumps(cache))
    cpath.write_text(json.dumps(cache))
    rep["pool_n"] = len(real)
    rep["pool_parity_old_shimmer"] = {"n": len(par), "max_abs": round(max(par), 5) if par else None}
    print("pool", len(real), "parity", rep["pool_parity_old_shimmer"], flush=True)
    R = {k: np.array([r[k] for r in real]) for k in ("shimmer", "shimmer_p", "f0_med")}
    rep["pool_pct"] = {k: [round(float(z), 3) for z in np.percentile(R[k][np.isfinite(R[k])], [2.5, 15, 50, 85, 97.5])] for k in R}

    def judge(sv: np.ndarray, rv: np.ndarray) -> dict:
        sv, rv = sv[np.isfinite(sv)], rv[np.isfinite(rv)]
        if len(sv) == 0:
            return {}
        lo, hi, l15, h15 = np.percentile(rv, [2.5, 97.5, 15, 85])
        md = float(np.median(sv))
        of = float(np.mean((sv < lo) | (sv > hi)))
        return {"median": round(md, 3), "real_p15_p85": [round(float(l15), 3), round(float(h15), 3)], "out_frac": round(of, 3), "pass": bool(of <= 0.15 and l15 <= md <= h15)}

    items = []
    if a.ear:
        items += ear_sets()
    for spec in a.sets:
        name, rest = spec.split("=", 1)
        pat, lab = rest.rsplit(":", 1)
        items += [(name, f, lab) for f in sorted(glob.glob(pat))]
    rows: dict = {}
    for g, f, lab in items:
        p = Path(f)
        r = cstats(f"set|{p}|{p.stat().st_mtime_ns}", lambda: load48(p))
        rows.setdefault(g, {"lab": lab, "rows": []})["rows"].append(r)
    cpath.write_text(json.dumps(cache))
    rep["sets"] = {}
    for g, d in rows.items():
        vals = {k: np.array([r[k] for r in d["rows"]]) for k in ("shimmer", "shimmer_p", "f0_med")}
        rep["sets"][g] = {"lab": d["lab"], "n": len(d["rows"]), "f0_med": round(float(np.nanmedian(vals["f0_med"])), 1),
                          "shimmer": judge(vals["shimmer"], R["shimmer"]), "shimmer_p": judge(vals["shimmer_p"], R["shimmer_p"])}
        s = rep["sets"][g]
        print(f"{g:40s} {d['lab']:5s} n {s['n']:3d} f0 {s['f0_med']:6.1f} | old {s['shimmer'].get('median')} out {s['shimmer'].get('out_frac')} {s['shimmer'].get('pass')} | new {s['shimmer_p'].get('median')} out {s['shimmer_p'].get('out_frac')} {s['shimmer_p'].get('pass')}", flush=True)
    if a.ear:
        allr = [(r, d["lab"]) for d in rows.values() for r in d["rows"] if d["lab"] in ("pass", "fail")]
        rep["ear_auc"] = {}
        for k in ("shimmer", "shimmer_p"):
            pos = np.array([r[k] for r, l in allr if l == "fail"])
            neg = np.array([r[k] for r, l in allr if l == "pass"])
            rep["ear_auc"][k] = {"auc": round(auc(pos, neg), 3), "n_fail": int(len(pos)), "n_pass": int(len(neg))}
        print("ear_auc", rep["ear_auc"], flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
