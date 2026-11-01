"""D700(腕 67・paper_mf E-A1b・規則は results/diag_ppgvoc67/prereg.yaml): 推論の流れ = [c0(H/d695/stream の列 = 腕 65 の c0 の表), 中立の f0]。
中立の f0 = D694 の書き出し(Rust の F1 の f0・k_書き出し の半音つき)と事後の argmax に train_ppgvoc.f0v_np(学習と同じ式・腕 67 の有声の表)を掛け、2^{(k − k_書き出し)/12} を掛ける
→ H/d700/stream/<組>_k<k>/{pi}_{ti}.f32。
    uv run python d700_streams.py --vtab ../results/diag_ppgvoc67/vv_tab.f32
E-S1(製品の界面・c0 は元のまま): --keep_c0 --out_root H/d702/stream → [NaN, 中立の f0]。
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import train_ppgvoc as P

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
DC, Q0, Q1 = 295, 31, 231
JOBS = [("d487_uiro", None, 12, (12, 5)), ("devvctkm", "ME", 12, (12, 5)), ("devff2", None, 0, (0,))]


def pairs(s: str, kind: str | None) -> list:
    return [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kind is None or r["kind"] == kind]


def build(job: tuple) -> dict:
    s, r, kd, ks, vtab, keep_c0, root = job
    P.F0V.update(on=True, vtab=vtab)
    x = np.fromfile(H / "d694" / "dump" / s / f"{r['pi']}_{r['ti']}.f32", dtype="<f4").reshape(-1, DC)
    if keep_c0:
        st = np.full((len(x), 2), np.nan, np.float32)
    else:
        st = np.fromfile(H / "d695" / "stream" / s / f"{r['pi']}_{r['ti']}.f32", dtype="<f4").reshape(-1, 2)
    assert len(st) == len(x)
    f1 = np.where(x[:, 26] > 0.5, 200.0 * np.exp(x[:, 25].astype(np.float64)), 0.0)
    fn = P.f0v_np(f1, x[:, Q0:Q1].argmax(1), vtab)
    for k in ks:
        o = Path(root) / f"{s}_k{k}" / f"{r['pi']}_{r['ti']}.f32"
        o.parent.mkdir(parents=True, exist_ok=True)
        np.stack([st[:, 0], (fn * 2.0 ** ((k - kd) / 12.0)).astype(np.float32)], 1).astype("<f4").tofile(o)
    v1, vn = f1 > 0, fn > 0
    m = v1 & vn
    return {"set": s, "vfrac_f1": float(v1.mean()), "vfrac_neutral": float(vn.mean()), "voicing_disagree": float((v1 != vn).mean()),
            "cent_abs_median": float(np.median(np.abs(1200 * np.log2(fn[m] / f1[m])))) if m.any() else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtab", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--keep_c0", action="store_true")
    ap.add_argument("--out_root", default=str(H / "d700" / "stream"))
    ap.add_argument("--report", default=str(ROOT / "results/diag_ppgvoc67/streams.json"))
    ap.add_argument("--sets", nargs="*", default=[], help="JOBS の代わり: <組>=<種類と k を写す元の組>(例 devvctkm_mp3=devvctkm)")
    a = ap.parse_args()
    vtab = np.fromfile(a.vtab, dtype="<f4").astype(np.float64)
    J = [(n.split("=")[0], *next(j[1:] for j in JOBS if j[0] == n.split("=")[1])) for n in a.sets] if a.sets else JOBS
    jobs = [(s, r, kd, ks, vtab, a.keep_c0, a.out_root) for s, kn, kd, ks in J for r in pairs(s, kn)]
    with Pool(a.workers) as pl:
        res = pl.map(build, jobs)
    rep = {s: {k: round(float(np.median([x[k] for x in res if x["set"] == s and x[k] is not None])), 4) for k in ("vfrac_f1", "vfrac_neutral", "voicing_disagree", "cent_abs_median")}
           for s, *_ in J}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
