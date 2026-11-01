import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal as ss

ROOT = Path("/home/kojirotanaka/kjranyone/LightVC")
sys.path.insert(0, str(ROOT / "training"))
import d181_verdict as V
import nvoc as N
from d6_unit_cov import load48

S = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2e1a59c3-e906-4c48-bba4-03855ea95b2e/scratchpad")
RT = Path("/home/kojirotanaka/lightvc_scratch_20261013")
cf = 1000.0 * 2.0 ** (np.arange(-10, 11) / 3.0)
cf = cf[(cf >= 100) & (cf <= 10000)]


def active(x):
    n = len(x) // N.HOP
    e = 10 * np.log10((x[: n * N.HOP].reshape(n, N.HOP) ** 2).mean(1) + 1e-12)
    return e > e.max() - 40


def ltas(path):
    x = load48(path).astype(np.float64)
    x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
    act = active(x)
    fq, _, Z = ss.stft(x, N.SR, nperseg=2048, noverlap=2048 - N.HOP)
    P = np.abs(Z[:, : len(act)]) ** 2
    P = P[:, act[: P.shape[1]]].mean(1)
    d = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() for c in cf]) + 1e-20)
    return d - d.mean()


FORMS = {"RRQSDC19L": "d139", "G03S10ZH": "d114", "KNN": "d135_real"}
rows = []
for D in ("dev", "dev2", "dev3"):
    P = RT / D
    R = V.ranks(S / f"rr19{D}_lda.npz", ROOT / f"results/conv_p0/rr19{D}_tri.json")
    cen = P / ("lda181ind" if D == "dev" else "lda_ind")
    tg = {}
    for k in sorted(R["LDA"]["KNN"]):
        tg.setdefault(int(k.split("_t")[1]), []).append(k)
    for ti, ks in tg.items():
        Lref = ltas(P / "d37/_work" / f"ref_{ti:03d}.wav")
        Lcen = ltas(cen / f"cen_{ti:03d}.wav")
        r = {"set": D, "t": ti, "ref_cen": float(np.sqrt(((Lref - Lcen) ** 2).mean()))}
        for f, d in FORMS.items():
            Ls = [ltas(P / d / f"{k}_{f}.wav") for k in ks]
            r[f"ref_{f}"] = float(np.mean([np.sqrt(((L - Lref) ** 2).mean()) for L in Ls]))
            r[f"cen_{f}"] = float(np.mean([np.sqrt(((L - Lcen) ** 2).mean()) for L in Ls]))
            for m in ("LDA", "ECAPA"):
                r[f"{m}_{f}"] = float(np.mean([R[m][f][k] for k in ks]))
        rows.append(r)
json.dump(rows, open(S / "diag_ltas_target.json", "w"))
a = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("set", "t")}
print("n", len(rows))
for f in FORMS:
    print(f, "ltas_d to ref", a[f"ref_{f}"].mean().round(2), "to cen", a[f"cen_{f}"].mean().round(2))
print("ref vs cen (same speaker, other recording)", a["ref_cen"].mean().round(2))
for m in ("LDA", "ECAPA"):
    dr = a[f"{m}_RRQSDC19L"] - a[f"{m}_KNN"]
    dl = a["cen_RRQSDC19L"] - a["cen_KNN"]
    print(m, "corr(rank diff RR-KNN, ltas_d(cen) RR-KNN)", np.corrcoef(dr, dl)[0, 1].round(3),
          "corr(RR rank, ltas_d(cen) RR)", np.corrcoef(a[f"{m}_RRQSDC19L"], a["cen_RRQSDC19L"])[0, 1].round(3),
          "corr(KNN rank, ltas_d(cen) KNN)", np.corrcoef(a[f"{m}_KNN"], a["cen_KNN"])[0, 1].round(3),
          "corr(RR rank, ref_cen)", np.corrcoef(a[f"{m}_RRQSDC19L"], a["ref_cen"])[0, 1].round(3))
