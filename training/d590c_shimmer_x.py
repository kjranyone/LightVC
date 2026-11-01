"""D590c(学習なし): 高さをそろえた shimmer の超過(shimmer_x)。統計は d109 の旧 shimmer のまま(耳の較正を保つ)・参照の側で高さをそろえる。
各発話: 隣り合う有声フレームの |Δ dB|(d109 と同じ 5 ms 格子・Hann 10 ms 窓・先頭 8 s・RMS 0.05)と、その対の f0(幾何平均)。
参照 = 本物の声のフレームの対: d109 と同じ選び方の実女声 150 人 + 実男声(VCTK 男声 devvctkm・外郎売 devreal の入力)= 低い高さの帯を埋める。
f0 の帯 = 80 Hz から 1/6 オクターブ。発話の超過 = その発話の |Δ| の中央値 − 参照のフレームを発話の帯の数に重みづけした重みつき中央値(参照のフレームが 200 未満の帯の発話のフレームは除き、除いた割合を記録)。
本物の分布 = 実女声 150 人の超過(自分の話者を参照から除く)。判定 = d109 と同じ規則(外れ ≤ 15% かつ 中央値が本物の [p15, p85])。
    uv run python d590c_shimmer_x.py --s <scratch> --tag d590c_dev2 --sets 'NAME=<glob>:none' ... [--ear]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import d590_shimmer_p as D

ROOT = Path(__file__).resolve().parent.parent
SEC = 8.0
HREF = Path("/home/kojirotanaka/lightvc_scratch_20261013")
EDGES = 80.0 * 2.0 ** (np.arange(0, 23) / 6.0)
MIN_REF = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_pool", type=int, default=150)
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--ear", action="store_true")
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

    def frames(x: np.ndarray) -> tuple:
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
        fl = f[: len(ed)]
        return np.abs(np.diff(ed))[both].astype(np.float32), np.sqrt(fl[1:] * fl[:-1])[both].astype(np.float32)

    cdir = S / "d590c_cache"
    cdir.mkdir(exist_ok=True)

    def cframes(key: str, xf) -> tuple:
        p = cdir / (hashlib.md5(key.encode()).hexdigest()[:16] + ".npz")
        if not p.exists():
            dd, fp = frames(xf())
            np.savez(p, dd=dd, fp=fp, key=key)
        z = np.load(p)
        assert str(z["key"]) == key
        return z["dd"], z["fp"]

    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(109).shuffle(pool)
    ref_units: list = []
    nf = 0
    for spk in pool:
        if nf >= a.n_pool:
            break
        ws = sorted((base / spk).glob("*.wav"))
        if len(ws) < 3 or any(sf.info(str(w)).samplerate < 44100 for w in ws[:3]):
            continue
        w0 = ws[len(ws) // 2]
        if sf.info(str(w0)).frames * SR / sf.info(str(w0)).samplerate < 3 * SR:
            continue
        rest = sum(sf.info(str(w)).frames * SR / sf.info(str(w)).samplerate for w in [w for w in ws if w != w0][:6])
        if rest < 8 * SR:
            continue
        dd, fp = cframes(f"pool|{spk}|{w0.name}", lambda: load48(w0))
        ref_units.append(("fem", spk, dd, fp))
        nf += 1
    for sname in ("devvctkm", "devreal"):
        for w in sorted((HREF / sname / "d56").glob("*_SRC.wav")):
            dd, fp = cframes(f"male|{sname}|{w.name}", lambda: load48(w))
            ref_units.append(("male", f"{sname}|{w.name}", dd, fp))
    rdd = np.concatenate([u[2] for u in ref_units])
    rbin = np.digitize(np.concatenate([u[3] for u in ref_units]), EDGES)
    rown = np.concatenate([np.full(len(u[2]), i) for i, u in enumerate(ref_units)])
    order = np.argsort(rdd)
    rdd, rbin, rown = rdd[order], rbin[order], rown[order]
    nb_all = np.bincount(rbin, minlength=len(EDGES) + 1)
    print("ref units", len(ref_units), "female", nf, "frames", len(rdd), "per bin", dict(zip([int(e) for e in EDGES], nb_all[1:len(EDGES)].tolist())), flush=True)

    def excess(dd: np.ndarray, fp: np.ndarray, skip: int = -1) -> tuple:
        b = np.digitize(fp, EDGES)
        nb = nb_all.copy()
        if skip >= 0:
            nb = nb - np.bincount(rbin[rown == skip], minlength=len(nb))
        keep = nb[b] >= MIN_REF
        if keep.sum() < 20:
            return float("nan"), float(1 - keep.mean()) if len(keep) else 1.0, float("nan")
        tb = np.bincount(b[keep], minlength=len(nb)).astype(np.float64)
        wb = np.where(nb > 0, tb / np.maximum(nb, 1), 0.0)
        w = wb[rbin]
        if skip >= 0:
            w = np.where(rown == skip, 0.0, w)
        cw = np.cumsum(w)
        mref = float(rdd[np.searchsorted(cw, cw[-1] / 2)])
        md = float(np.median(dd[keep]))
        return md - mref, float(1 - keep.mean()), mref

    real = []
    for i, u in enumerate(ref_units):
        if u[0] == "fem":
            real.append(excess(u[2], u[3], skip=i)[0])
    real = np.array(real)
    lo, hi, l15, h15 = np.percentile(real[np.isfinite(real)], [2.5, 97.5, 15, 85])
    rep: dict = {"tag": a.tag, "ref_units": len(ref_units), "ref_frames": int(len(rdd)), "real_pct": [round(float(z), 3) for z in (lo, l15, np.median(real), h15, hi)]}
    print("real excess pct", rep["real_pct"], flush=True)

    def judge(sv: np.ndarray) -> dict:
        sv = sv[np.isfinite(sv)]
        if len(sv) == 0:
            return {}
        md = float(np.median(sv))
        of = float(np.mean((sv < lo) | (sv > hi)))
        return {"median": round(md, 3), "real_p15_p85": [round(float(l15), 3), round(float(h15), 3)], "out_frac": round(of, 3), "pass": bool(of <= 0.15 and l15 <= md <= h15)}

    items = []
    if a.ear:
        items += D.ear_sets()
    for spec in a.sets:
        name, rest_ = spec.split("=", 1)
        pat, lab = rest_.rsplit(":", 1)
        items += [(name, f, lab) for f in sorted(glob.glob(pat))]
    rows: dict = {}
    per_file: dict = {}
    for g, f, lab in items:
        p = Path(f)
        dd, fp = cframes(f"set|{p}|{p.stat().st_mtime_ns}", lambda: load48(p))
        ex, drop, mref = excess(dd, fp)
        r = {"x": ex, "drop": drop, "old": float(np.median(dd)) if len(dd) else float("nan"), "f0": float(np.median(fp)) if len(fp) else float("nan")}
        rows.setdefault(g, {"lab": lab, "rows": []})["rows"].append(r)
        per_file[str(p)] = r
    rep["sets"] = {}
    for g, d in rows.items():
        X = np.array([r["x"] for r in d["rows"]])
        rep["sets"][g] = {"lab": d["lab"], "n": len(X), "f0_med": round(float(np.nanmedian([r["f0"] for r in d["rows"]])), 1),
                          "drop_med": round(float(np.median([r["drop"] for r in d["rows"]])), 3), "old_med": round(float(np.nanmedian([r["old"] for r in d["rows"]])), 3), "shimmer_x": judge(X)}
        s = rep["sets"][g]
        print(f"{g:40s} {d['lab']:5s} n {s['n']:3d} f0 {s['f0_med']:6.1f} drop {s['drop_med']:.3f} | old {s['old_med']} | x {s['shimmer_x'].get('median')} out {s['shimmer_x'].get('out_frac')} {s['shimmer_x'].get('pass')}", flush=True)
    if a.ear:
        keep = []
        for g, f, lab in items:
            if g == "ff_v3_ear:P43OP" and Path(f).name[:3] not in ("t01", "t02", "t03"):
                continue
            if lab in ("pass", "fail"):
                keep.append((lab, per_file[str(Path(f))]))
        pos = {k: np.array([r[k] for l, r in keep if l == "fail"]) for k in ("old", "x")}
        neg = {k: np.array([r[k] for l, r in keep if l == "pass"]) for k in ("old", "x")}
        au = {k: D.auc(pos[k], neg[k]) for k in pos}
        rng = np.random.default_rng(590)
        ds = []
        for _ in range(4000):
            ip = rng.integers(0, len(pos["old"]), len(pos["old"]))
            ineg = rng.integers(0, len(neg["old"]), len(neg["old"]))
            ds.append(D.auc(pos["x"][ip], neg["x"][ineg]) - D.auc(pos["old"][ip], neg["old"][ineg]))
        rep["ear_auc"] = {"old": round(au["old"], 3), "x": round(au["x"], 3), "diff": round(au["x"] - au["old"], 3), "ci95": [round(float(z), 3) for z in np.percentile(ds, [2.5, 97.5])],
                          "n_fail": int(len(pos["old"])), "n_pass": int(len(neg["old"]))}
        print("ear_auc", rep["ear_auc"], flush=True)
    rep["per_file"] = per_file
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
