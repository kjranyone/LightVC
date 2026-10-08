"""f0hi.teacher_f0(v2)の実データの監査: 標本ファイル(manifest の既存 f0 = A をそのまま使う)で、置換したフレームを層別に数え、
(1) A が有声で A ≈ B/2 (2) A が有声で無関係 (3) A が無声 (4) 区間内の隣接フレームの跳躍 > 300cent (5) A との階段(区間の縁)(6) pYIN(fmax 2400)との一致(一部)
を出し、層ごとに無作為の区間のスペクトログラムを保存する(標本抽出の seed 固定・盲検ではない)。
    uv run python f0hi_audit.py --n_real 300 --n_tts 100 --out ../results/f0hi/audit_v2.json --fig_dir <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import gcd
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import f0hi as H

ROOT = Path(__file__).resolve().parent.parent


def load48(p: str, sec: float = 12.0) -> np.ndarray:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 48000:
        g = gcd(sr, 48000)
        x = resample_poly(x, 48000 // g, sr // g).astype(np.float32)
    return x[:int(sec * 48000)]


def runs(b):
    d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def one(job):
    row, kind = job
    try:
        x = load48(row["wav"])
        n = len(x) // H.HOP
        A = np.load(ROOT / row["f0"]).astype(np.float32)[:n]
        f, rep = H.teacher_f0(x, n, A)
        assert np.array_equal(f[~rep], A[~rep]), "区間の外が A とビット一致しない"
        rec = {"wav": row["wav"], "kind": kind, "voiced": int((A > 0).sum()), "B": int(rep.sum())}
        if not rep.any():
            return rec
        B = f
        half = rep & (A > 0) & (np.abs(H.cents(A, B / 2)) < 100)
        other = rep & (A > 0) & ~half
        unv = rep & (A == 0)
        jump = 0
        step_edge = []
        for a, b in runs(rep):
            d = np.abs(H.cents(B[a + 1:b], B[a:b - 1]))
            jump += int((d > 300).sum())
            for e, o in ((a, a - 1), (b - 1, b)):
                if 0 <= o < n and not rep[o]:
                    step_edge.append(float(abs(H.cents(np.array([B[e]]), np.array([max(f[o], 1e-3)]))[0])) if f[o] > 0 else -1.0)
        rec.update(A_is_B_over_2=int(half.sum()), A_valid_unrelated=int(other.sum()), A_unvoiced=int(unv.sum()), jump_gt300=jump,
                   segs=runs(rep), edge_step_cent=step_edge)
        return rec
    except Exception as e:
        return {"wav": row["wav"], "kind": kind, "err": f"{type(e).__name__}: {e}"}


def pyin_agree(rec):
    try:
        import librosa
        x = load48(rec["wav"])
        n = len(x) // H.HOP
        ROWS = rec["_row"]
        A = np.load(ROOT / ROWS["f0"]).astype(np.float32)[:n]
        f, rep = H.teacher_f0(x, n, A)
        x16 = resample_poly(x, 1, 3)
        py, vf, _ = librosa.pyin(x16, fmin=60, fmax=2400, sr=16000, frame_length=1024, hop_length=80)
        py = np.nan_to_num(py)[:n]
        m = min(len(py), len(f))
        sel = rep[:m] & (py[:m] > 0)
        agree = (np.abs(H.cents(f[:m], np.where(py[:m] > 0, py[:m], 1))) < 100) & sel
        return int(rep[:m].sum()), int(sel.sum()), int(agree.sum())
    except Exception:
        return 0, 0, 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_real", type=int, default=300)
    ap.add_argument("--n_tts", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig_dir", default="")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--pyin", type=int, default=40)
    a = ap.parse_args()
    m = json.loads((ROOT / "data/rvoc_f0/manifest.json").read_text())
    real = [r for r in m["rows"] if r.get("keep") and r["src"] == "real_female"]
    tts = [r for r in m["rows"] if r.get("keep") and r["src"].startswith("tts")]
    rng = random.Random(11)
    jobs = [(r, "real") for r in rng.sample(real, a.n_real)] + [(r, "tts") for r in rng.sample(tts, a.n_tts)]
    with Pool(a.workers) as pool:
        res = pool.map(one, jobs, chunksize=4)
    rep = {}
    for kind in ("real", "tts"):
        r = [x for x in res if x["kind"] == kind and "err" not in x]
        voiced = sum(x["voiced"] for x in r)
        B = sum(x["B"] for x in r)
        rep[kind] = {"files": len(r), "files_with_B": sum(1 for x in r if x["B"]), "voiced_frames": voiced, "B_frames": B,
                     "B_frac_of_voiced_pct": round(100 * B / max(1, voiced), 3),
                     "A_is_B_over_2": sum(x.get("A_is_B_over_2", 0) for x in r), "A_valid_unrelated": sum(x.get("A_valid_unrelated", 0) for x in r),
                     "A_unvoiced": sum(x.get("A_unvoiced", 0) for x in r), "jump_gt300cent_inside_runs": sum(x.get("jump_gt300", 0) for x in r),
                     "errors": sum(1 for x in res if x["kind"] == kind and "err" in x)}
    steps = [s for x in res for s in x.get("edge_step_cent", [])]
    rep["run_edges"] = {"n": len(steps), "A_unvoiced_at_edge": sum(1 for s in steps if s < 0),
                        "A_voiced_median_step_cent": round(float(np.median([s for s in steps if s >= 0])), 0) if any(s >= 0 for s in steps) else None,
                        "A_voiced_step_gt300cent": sum(1 for s in steps if s > 300)}
    with_B = [x for x in res if x.get("B")]
    mp = {r["wav"]: r for r in m["rows"]}
    sel = random.Random(2).sample([x for x in with_B if x["kind"] == "real"], min(a.pyin, len(with_B)))
    with Pool(a.workers) as pool:
        pr = pool.map(pyin_agree, [{**x, "_row": mp[x["wav"]]} for x in sel], chunksize=2)
    nB = sum(p[0] for p in pr); nsel = sum(p[1] for p in pr); nag = sum(p[2] for p in pr)
    rep["pyin_fmax2400"] = {"files": len(sel), "B_frames": nB, "pyin_voiced_on_B": nsel, "agree_within_100cent": nag,
                             "agree_frac_of_pyin_voiced": round(nag / max(1, nsel), 3), "pyin_voiced_frac_of_B": round(nsel / max(1, nB), 3)}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": rep, "files": with_B}, ensure_ascii=False, indent=1))
    if a.fig_dir:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Path(a.fig_dir).mkdir(parents=True, exist_ok=True)
        rr = random.Random(5)
        groups = {"HALF_A(A が B/2)": ("A_is_B_over_2", 12), "UNRELATED(A 有声・無関係)": ("A_valid_unrelated", 12), "A_UNVOICED": ("A_unvoiced", 8)}
        for gi, (gname, (key, k)) in enumerate(groups.items()):
            pool_ = [x for x in with_B if x.get(key, 0) >= 8]
            picks = rr.sample(pool_, min(k, len(pool_)))
            nr = (len(picks) + 1) // 2
            fig, axs = plt.subplots(nr, 2, figsize=(14, 3.1 * nr)); axs = np.array(axs).ravel()
            for ax, x in zip(axs, picks):
                w = load48(x["wav"]); n = len(w) // H.HOP
                A = np.load(ROOT / mp[x["wav"]]["f0"]).astype(np.float32)[:n]
                f, rp = H.teacher_f0(w, n, A)
                seg = max(x["segs"], key=lambda s: s[1] - s[0]); c = (seg[0] + seg[1]) // 2
                t0 = max(0, c * 0.005 - 0.35); t1 = t0 + 0.7
                ax.specgram(w[int(t0 * 48000):int(t1 * 48000)], NFFT=1024, Fs=48000, noverlap=768, cmap="magma", vmin=-120, vmax=-30)
                tt = np.arange(len(f)) * 0.005; mk = (tt >= t0) & (tt < t1)
                ax.plot(tt[mk] - t0, np.where(rp[mk], f[mk], np.nan), "c.", ms=4)
                ax.plot(tt[mk] - t0, np.where((A[:len(tt)][mk] > 0) & ~rp[mk], A[:len(tt)][mk], np.nan), "w.", ms=2)
                ax.set_ylim(0, 5000); ax.set_title(f"{Path(x['wav']).name[-22:]}  B={x['B']}", fontsize=7)
            fig.suptitle(gname)
            plt.tight_layout(); plt.savefig(Path(a.fig_dir) / f"audit_{gi}.png", dpi=48); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
