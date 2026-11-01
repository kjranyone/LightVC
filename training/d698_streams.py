"""D698(paper_mf E-A1b0・規則は results/conv_p0/d698/prereg.yaml): N65 の診断の流れ(H/d695/stream の c0 の列)に、F1 の f0・有声の代わりに入力の非因果の教師
(f0hi.teacher_f0 = 生成器の学習の f0 と同じ作り方)× 2^{k/12} を入れた流れを作る → H/d698/stream/<組>_k<k>/{pi}_{ti}.f32([c0, 出力の f0 Hz(0 = 無声)])。
時刻は D694 の書き出し(H/d694/dump・Rust の F1)の有声のフレームで ±4 フレームの探索(対数の差の分散が最小)。F1 と教師の食い違い(有声・cent)を記録する。
--check <wav の書き出し>: 通常のモードの LIGHTVC_DUMP_COND が流れ(c0 = 0 行・有声 = 26 行・log(f0/200) = 25 行)と一致することを確かめる。
    uv run python d698_streams.py
    uv run python d698_streams.py --check <dump.f32> --stream <stream.f32>
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import f0hi as FH
from d6_unit_cov import load48

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
DC = 295
JOBS = [("d487_uiro", None, (12, 5)), ("devvctkm", "ME", (12, 5)), ("devff2", None, (0,))]


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def build(job: tuple) -> dict:
    s, r, ks = job
    key = f"p{r['pi']:04d}_t{r['ti']:03d}"
    x = np.fromfile(H / "d694" / "dump" / s / f"{r['pi']}_{r['ti']}.f32", dtype="<f4").reshape(-1, DC)
    T = len(x)
    st = np.fromfile(H / "d695" / "stream" / s / f"{r['pi']}_{r['ti']}.f32", dtype="<f4").reshape(-1, 2)
    assert len(st) == T, (s, key, len(st), T)
    v1 = x[:, 26] > 0.5
    l1 = x[:, 25]
    ft, _ = FH.teacher_f0(load48(H / s / "d56" / "_work" / f"pre_{key}.wav").astype(np.float64), n=T + 8)
    best = None
    for L in range(-4, 5):
        fs = np.zeros(T, np.float32)
        lo, hi = max(0, -L), min(T, len(ft) - L)
        fs[lo:hi] = ft[lo + L:hi + L]
        m = v1 & (fs > 0)
        if m.sum() < 50:
            continue
        d = l1[m] - np.log(fs[m] / 200.0)
        if best is None or float(np.var(d)) < best[0]:
            best = (float(np.var(d)), L, fs, float(np.median(d)))
    _, L, fs, off = best
    vt = fs > 0
    m = v1 & vt
    cent = 1200.0 / np.log(2.0) * (l1[m] - np.log(fs[m] / 200.0) - off)
    for k in ks:
        o = H / "d698" / "stream" / f"{s}_k{k}" / f"{r['pi']}_{r['ti']}.f32"
        o.parent.mkdir(parents=True, exist_ok=True)
        np.stack([st[:, 0], np.where(vt, fs * 2.0 ** (k / 12.0), 0.0).astype(np.float32)], 1).astype("<f4").tofile(o)
    return {"set": s, "key": key, "lag": L, "k_dump_est": round(off * 12.0 / np.log(2.0), 2), "vfrac_f1": float(v1.mean()), "vfrac_teacher": float(vt.mean()),
            "voicing_disagree": float((v1 != vt).mean()), "f1_only": float((v1 & ~vt).mean()), "teacher_only": float((vt & ~v1).mean()),
            "cent_abs_median": float(np.median(np.abs(cent))), "octave_err": float((np.abs(cent) > 600).mean())}


def check(dump: Path, stream: Path) -> int:
    x = np.fromfile(dump, dtype="<f4").reshape(-1, DC)
    st = np.fromfile(stream, dtype="<f4").reshape(-1, 2)
    f = st[:len(x), 1]
    v = f > 0
    c0s = st[:len(x), 0]
    fin = np.isfinite(c0s)
    rep = {"frames_dump": len(x), "frames_stream": len(st), "c0_from_stream_frames": int(fin.sum()), "max_c0_diff": float(np.abs(x[fin, 0] - c0s[fin]).max()) if fin.any() else 0.0,
           "voicing_mismatch": int(((x[:, 26] > 0.5) != v).sum()), "max_logf0_diff": float(np.abs(x[v, 25] - np.log(f[v] / 200.0)).max())}
    ok = rep["frames_dump"] == rep["frames_stream"] and rep["max_c0_diff"] == 0.0 and rep["voicing_mismatch"] == 0 and rep["max_logf0_diff"] < 1e-5
    print(json.dumps(rep), "OK" if ok else "MISMATCH", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", default="")
    ap.add_argument("--stream", default="")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.check:
        return check(Path(a.check), Path(a.stream))
    jobs = [(s, r, ks) for s, kd, ks in JOBS for r in pairs(s, kd)]
    with Pool(a.workers) as pl:
        res = pl.map(build, jobs)
    rep: dict = {}
    for s, *_ in JOBS:
        rs = [x for x in res if x["set"] == s]
        rep[s] = {"n": len(rs)} | {k: round(float(np.median([x[k] for x in rs])), 4) for k in ("lag", "k_dump_est", "vfrac_f1", "vfrac_teacher", "voicing_disagree", "f1_only",
                                                                                         "teacher_only", "cent_abs_median", "octave_err")}
    out = ROOT / "results/conv_p0/d698"
    out.mkdir(parents=True, exist_ok=True)
    (out / "streams.json").write_text(json.dumps({"summary": rep, "per": res}, indent=1))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
