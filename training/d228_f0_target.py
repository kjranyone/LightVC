import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("/home/kojirotanaka/kjranyone/LightVC")
sys.path.insert(0, str(ROOT / "training"))
S = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2e1a59c3-e906-4c48-bba4-03855ea95b2e/scratchpad")
RT = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def f0med(path: str) -> float:
    import librosa
    x, _ = librosa.load(path, sr=16000)
    x = x[: 16000 * 12]
    f, v, _ = librosa.pyin(x, fmin=90, fmax=900, sr=16000, frame_length=1024)
    f = f[v & np.isfinite(f)]
    return float(np.median(f)) if len(f) > 20 else float("nan")


rows = json.load(open(S / "diag_ltas_target.json"))
jobs = []
for r in rows:
    P = RT / r["set"]
    cen = P / ("lda181ind" if r["set"] == "dev" else "lda_ind") / f"cen_{r['t']:03d}.wav"
    ks = sorted(p.name.split("_RRQSDC19L")[0] for p in (P / "d139").glob(f"p*_t{r['t']:03d}_RRQSDC19L.wav"))
    jobs.append((str(cen), [str(P / "d139" / f"{k}_RRQSDC19L.wav") for k in ks], [str(P / "d135_real" / f"{k}_KNN.wav") for k in ks],
                 [str(P / "d56" / f"{k}_SRC.wav") for k in ks], str(P / "d37/_work" / f"ref_{r['t']:03d}.wav")))
flat = sorted({p for j in jobs for p in [j[0], j[4]] + j[1] + j[2] + j[3]})
with Pool(6) as pool:
    med = dict(zip(flat, pool.map(f0med, flat, chunksize=4)))
out = []
for r, (cen, rr, kn, src, ref) in zip(rows, jobs):
    mu = float(np.exp(json.loads((RT / r["set"] / "d85_work" / f"en_{r['t']:03d}" / "enroll.json").read_text())["mu"]))
    out.append(r | {"mu": mu, "f0_cen": med[cen], "f0_ref": med[ref], "f0_rr": float(np.nanmean([med[p] for p in rr])), "f0_knn": float(np.nanmean([med[p] for p in kn])),
                    "f0_src": float(np.nanmean([med[p] for p in src]))})
json.dump(out, open(S / "diag_f0_target.json", "w"))
a = {k: np.array([o[k] for o in out], float) for k in ("mu", "f0_cen", "f0_ref", "f0_rr", "f0_knn", "f0_src")}
print("pct f0_cen", np.nanpercentile(a["f0_cen"], [10, 50, 90]).round(0), "mu", np.nanpercentile(a["mu"], [10, 50, 90]).round(0))
for nm in ("mu", "f0_ref", "f0_rr", "f0_knn"):
    rt = 12 * np.log2(a[nm] / a["f0_cen"])
    print(f"{nm:7s} − cen (半音): median {np.nanmedian(rt):+.2f}  |>6| {np.nanmean(np.abs(rt) > 6):.3f}  >+9 {np.nanmean(rt > 9):.3f}")
for m in ("LDA", "ECAPA"):
    rr = np.array([o[f"{m}_RRQSDC19L"] for o in out]); kn = np.array([o[f"{m}_KNN"] for o in out])
    e = np.abs(12 * np.log2(a["f0_rr"] / a["f0_cen"]))
    ek = np.abs(12 * np.log2(a["f0_knn"] / a["f0_cen"]))
    ok = np.isfinite(e) & np.isfinite(ek)
    print(m, "corr(RR rank, |f0_rr − cen|)", np.corrcoef(rr[ok], e[ok])[0, 1].round(3), "corr(diff, |f0_rr−cen| − |f0_knn−cen|)", np.corrcoef((rr - kn)[ok], (e - ek)[ok])[0, 1].round(3))
    for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 99)):
        s = ok & (e >= lo) & (e < hi)
        print(f"   |f0_rr − cen| {lo}-{hi} 半音: n {s.sum():2d}  RR {rr[s].mean():5.2f}  KNN {kn[s].mean():5.2f}  diff {(rr - kn)[s].mean():+5.2f}")
