"""D237(学習なし・results/conv_p0/d237_prereg.yaml): 気持ち良さの目標関数 v0 = 心理音響の感覚的快さ(Zwicker & Fastl)の粗さ・鋭さが既存の耳のラベルを再現するか。
r_ecma = ECMA-418-2 の粗さ・r_dw = Daniel & Weber の粗さ(有音の時間中央値)・s_din = DIN 45692 の鋭さ(有音の時間中央値)・p_rs = exp(−0.7 r_dw)·exp(−1.08 s_din)。
音量 = 有音の RMS 0.05 FS → 65 dB SPL に固定。本物の女声 = D236 と同じ 150 人。mosqito 1.2.1(Apache-2.0・評価専用)。
    uv run --with mosqito==1.2.1 python d237_psy_ear.py --s <dev の親> --out ../results/conv_p0/d237.json
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
SR = 48000
SEC = 8.0
PA = 20e-6 * 10 ** (65 / 20) / 0.05
BAD = {"r_ecma": "hi", "r_dw": "hi", "s_din": "both", "p_rs": "lo"}
FAIL_GROUPS = ("FAIL_cand_ab1", "FAIL_prodcand", "FAIL_convg1", "FAIL_conv_ab1", "dev:G03S10ZH", "dev:RRQSDC19LE")
DEVF = {"KNN": ("d135_real", "KNN"), "G03S10ZH": ("d114", "G03S10ZH"), "RRQSDC19LE": ("d139", "RRQSDC19LE")}


def load48(p: str) -> np.ndarray:
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    x, sr = sf.read(p, dtype="float64", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x


def measure(job: tuple) -> dict:
    warnings.filterwarnings("ignore")
    from mosqito.sq_metrics import roughness_dw, roughness_ecma, sharpness_din_tv
    grp, ident, path, form = job
    x = load48(path)[: int(SEC * SR)]
    n = len(x) // 480
    e = 10 * np.log10((x[: n * 480].reshape(n, 480) ** 2).mean(1) + 1e-20)
    act = e > e.max() - 40
    x = x / (np.sqrt((x[: n * 480].reshape(n, 480)[act] ** 2).mean()) + 1e-12) * 0.05
    p = x * PA
    on = lambda t: act[np.clip((np.asarray(t) * 100).astype(int), 0, n - 1)]
    R, _, _, _, _ = roughness_ecma(p, SR)
    Rd, _, _, td = roughness_dw(p, SR, overlap=0.5)
    S, ts = sharpness_din_tv(p, SR, skip=0.2)
    rdw = float(np.median(Rd[on(td)])) if on(td).sum() else float(np.median(Rd))
    sdn = float(np.median(S[on(ts)])) if on(ts).sum() else float(np.median(S))
    r = {"grp": grp, "id": ident, "form": form, "r_ecma": float(R), "r_dw": rdw, "s_din": sdn}
    r["p_rs"] = float(np.exp(-0.7 * rdw) * np.exp(-1.08 * sdn))
    return r


def jobs(S: Path) -> list:
    J = []
    d236 = json.loads((ROOT / "results/conv_p0/d236.json").read_text())["rows"]
    base = ROOT / "female-dataset"
    for r in d236:
        if r["grp"] == "REAL_POOL":
            ws = sorted((base / r["id"]).glob("*.wav"))
            J.append(("REAL_POOL", r["id"], str(ws[len(ws) // 2]), ""))
        elif not r["grp"].startswith("dev:"):
            d, t, L = r["id"].split("/")
            J.append((r["grp"], r["id"], str(EB / d / t / f"{L}.wav"), r.get("form", "R")))
    for nm, (dd, c) in DEVF.items():
        for f in sorted((S / "dev" / dd).glob(f"*_{c}.wav")):
            J.append((f"dev:{nm}", f"dev/{f.name}", str(f), nm))
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cache = Path(a.out).with_suffix(".rows.jsonl")
    done = {json.loads(l)["id"]: json.loads(l) for l in cache.read_text().splitlines()} if cache.exists() else {}
    J = [j for j in jobs(Path(a.s)) if j[1] not in done]
    print("jobs", len(J), "cached", len(done), flush=True)
    with Pool(a.workers) as pool, cache.open("a") as fo:
        for i, r in enumerate(pool.imap_unordered(measure, J)):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            done[r["id"]] = r
            if (i + 1) % 25 == 0:
                print("done", i + 1, flush=True)
    rows = list(done.values())
    rep: dict = {}
    real = {m: np.array([r[m] for r in rows if r["grp"] == "REAL_POOL"], float) for m in BAD}
    rep["real_p15_p85"] = {m: [round(float(np.percentile(v, 15)), 4), round(float(np.percentile(v, 85)), 4)] for m, v in real.items()}
    grps = sorted({r["grp"] for r in rows})
    rep["medians"] = {g: {m: round(float(np.median([r[m] for r in rows if r["grp"] == g])), 4) for m in BAD} | {"n": sum(r["grp"] == g for r in rows)} for g in grps}

    def bad(m: str, v: float) -> bool:
        lo, hi = rep["real_p15_p85"][m]
        return v > hi if BAD[m] == "hi" else v < lo if BAD[m] == "lo" else (v < lo or v > hi)

    rep["axis"] = {}
    for m in BAD:
        pass_ok = all(not bad(m, rep["medians"][g][m]) for g in ("PASS_rvoc", "PASS_bigvgan"))
        det = {g: bad(m, rep["medians"][g][m]) for g in FAIL_GROUPS}
        rep["axis"][m] = {"pass_ok": pass_ok, "fail_detected": det, "axis_ok": pass_ok and all(det.values())}
    rep["trial1"] = {r["form"]: {m: round(r[m], 4) for m in BAD} for r in rows if r["id"].startswith("cand_ab1/cand_01/")}
    ok = [m for m in BAD if rep["axis"][m]["axis_ok"]]
    rep["verdict"] = "PSY_EXPLAINS" if any(m != "s_din" for m in ok) else "SHARP_ONLY" if ok else "NONE"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep | {"rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
