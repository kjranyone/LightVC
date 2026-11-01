"""D692(論文 current/paper_mf.md の E-B1・規則は results/conv_p0/d692/prereg.yaml): 目標の登録の表を低い声区の形へ写す(条件の側で低い声区の情報を与えるオラクル)。
Δ_reg(u) = 声の低い学習の女声(rvoc_tables_real の exp(mu) が 170〜250 Hz)の表の平均 − 目標と同じ高さの女声(|log(f0 / f0_T)| < 0.12)の表の平均(単位ごと・24 係数の空間 = にじみなし)。
B1R = T + Δ_reg・B1X = T − Δ_reg(逆向きの対照)。表現の検査(f0 より下だけを 24 係数で変えられるか・重みつきの最小二乗)も記録する。
出力 = <set>/rr_en_b1r/en_TTT・<set>/rr_en_b1x/en_TTT(enroll.json の写し + 書き換えた enroll.bin)・統計 = results/conv_p0/d692/build.json。
    uv run python d692_lowband.py
"""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import numpy as np

import nvoc as N
import zsvc as Z

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
ROOT = Path(__file__).resolve().parent.parent
SETS = ("d487_uiro", "devvctkm")
LO, HI, BW = 170.0, 250.0, 0.12


def wls(B: np.ndarray, L: np.ndarray, L2: np.ndarray, w: np.ndarray) -> np.ndarray:
    return B.T @ np.linalg.solve((B * w) @ B.T, (B * w) @ L2)


def main() -> int:
    B = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)[1:25]
    fb = N.mel_fb().numpy()
    fc = fb.argmax(1) * (N.SR / 2) / (fb.shape[1] - 1)
    Ts, F = [], []
    for f in sorted(glob.glob(str(ROOT / "data/rvoc_tables_real/*.npz"))):
        z = np.load(f)
        Ts.append(z["T"].astype(np.float64))
        F.append(float(np.exp(float(z["mu"]))))
    Ts, F = np.array(Ts), np.array(F)
    low = Ts[(F >= LO) & (F < HI)].mean(0)
    Pl = B.T @ low
    rep: dict = {"low_tables": int(((F >= LO) & (F < HI)).sum()), "band_hz": [LO, HI], "bw_log": BW, "targets": {}, "representation": {}}
    rows = []
    for s in SETS:
        for ti in sorted({r["ti"] for r in json.loads((H / s / "devset.json").read_text())["pairs"]}):
            src = H / s / "rr_en" / f"en_{ti:03d}"
            j = json.loads((src / "enroll.json").read_text())
            T = np.fromfile(src / "enroll.bin", dtype="<f4").reshape(j["D"], j["K"]).astype(np.float64)
            f0t = float(np.exp(j["mu"]))
            g = np.abs(np.log(F / f0t)) < BW
            dR = low - Ts[g].mean(0)
            pr = (B.T @ dR).mean(1) * 8.686
            rep["targets"][f"{s}/{ti:03d}"] = {"f0_T": round(f0t, 1), "n_same_register": int(g.sum()),
                                               "delta_db_lt300": round(float(pr[fc < 300].mean()), 2), "delta_db_600_1300": round(float(pr[(fc > 600) & (fc < 1300)].mean()), 2),
                                               "delta_db_1300_4000": round(float(pr[(fc > 1300) & (fc < 4000)].mean()), 2)}
            for nm, sg in (("b1r", 1.0), ("b1x", -1.0)):
                dst = H / s / f"rr_en_{nm}" / f"en_{ti:03d}"
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(src / "enroll.json", dst / "enroll.json")
                (T + sg * dR).astype("<f4").tofile(dst / "enroll.bin")
            L = B.T @ T
            ic = int(np.searchsorted(fc, 0.9 * f0t))
            if ic:
                L2 = L.copy()
                L2[:ic] = L[ic][None] + (Pl[:ic] - Pl[ic][None])
                mid = (np.arange(len(fc)) >= ic) & (fc <= 1200)
                r_ = {}
                for wm in (1.0, 10.0, 100.0):
                    w = np.ones(len(fc))
                    w[fc > 1200] = 1000.0
                    w[mid] = wm
                    R = wls(B, L, L2, w)
                    r_[wm] = ((R[:ic] - L[:ic]).mean() * 8.686, (L2[:ic] - L[:ic]).mean() * 8.686, np.abs(R[mid] - L[mid]).max() * 8.686)
                rows.append(r_)
    for wm in (1.0, 10.0, 100.0):
        a = np.array([r[wm] for r in rows])
        rep["representation"][f"w_mid_{wm:g}"] = {"below_fc_achieved_db": round(float(a[:, 0].mean()), 2), "below_fc_intended_db": round(float(a[:, 1].mean()), 2),
                                                 "fc_to_1200_max_abs_change_db": round(float(a[:, 2].max()), 2)}
    v = list(rep["targets"].values())
    rep["summary"] = {k: round(float(np.mean([t[k] for t in v])), 2) for k in ("delta_db_lt300", "delta_db_600_1300", "delta_db_1300_4000")}
    rep["summary"]["n_same_register_min"] = min(t["n_same_register"] for t in v)
    out = ROOT / "results/conv_p0/d692"
    out.mkdir(parents=True, exist_ok=True)
    (out / "build.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({"low_tables": rep["low_tables"], "summary": rep["summary"], "representation": rep["representation"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
