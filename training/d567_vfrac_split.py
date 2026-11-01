"""D567(推論だけ・診断・新しい描画なし): 男声 → 女声の出力の有声の割合(d109 vfrac)が入力より上がる分をフレームで分ける。
vo / vi = 出力 / 入力の d109 の有声(F1 の有声 > 0.5 かつ 高さの事後の山の質量 ≥ 0.5)・vi_s = 入力の F1 の有声だけ・v_inf = 推論の経路の有声(書き出しの条件 26 列)・vi_w = 入力の WORLD harvest の有声。
超過 E = 出力の発声中で vo = 1 かつ vi = 0。規則は OVERNIGHT の D567。
    uv run python d567_vfrac_split.py --name P57S0 --sets dev2 dev3 devreal devvctkm --out ../results/conv_p0/d567.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyworld as pw
import scipy.signal as ss
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f0est as FE
import nvoc as N
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
PRE = 12.0
SEC = 8.0
NP = 2400


def harvest(x: np.ndarray) -> np.ndarray:
    y = ss.resample_poly(x.astype(np.float64), 1, 3)
    f0, _ = pw.harvest(y, 16000, f0_floor=50.0, f0_ceil=800.0, frame_period=1000.0 * N.HOP / N.SR)
    return f0 > 0


def agree(a_: np.ndarray, b_: np.ndarray, L: int) -> float:
    aa, bb = (a_[L:], b_) if L >= 0 else (a_, b_[-L:])
    n = min(len(aa), len(bb))
    return float((aa[:n] == bb[:n]).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", action="store_true", help="関門の組の構成(出力 = gate/CV/<k>_CV.wav・入力 = d56/_work/pre_<k>.wav)。判定の後の記録だけ")
    ap.add_argument("--gate_name", default="CV", help="--gate の出力の版の名前(gate/<名前>/<k>_<名前>.wav)")
    ap.add_argument("--d570", action="store_true", help="開発の組の描画(<組>/d570/<名前>/<k>_<名前>.wav)を読む")
    a = ap.parse_args()
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)

    def f1v(x: np.ndarray) -> tuple:
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // N.HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * N.HOP].astype(np.float32))[None].to(dev)], 1), n))
            vs = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vs), pr.shape[1])
        k = pr[:, :m].argmax(0)
        ii = np.clip(k[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        c = np.take_along_axis(pr[:, :m], ii, 0).sum(0)
        e = 10 * np.log10((x[: m * N.HOP].reshape(m, N.HOP) ** 2).mean(1) + 1e-12)
        return vs[:m] & (c >= 0.5), vs[:m], e > e.max() - 40, e - e.max()

    n0 = int(PRE * N.SR)
    res = {}
    for s in a.sets:
        S = H / s
        E = H / "e2e" / f"eval_{s}_pre"
        od = S / "gate" / a.gate_name if a.gate else (S / "d570" / a.name if a.d570 else ROOT / "results/conv_p0/d558" / a.name / s)
        rows, xs = [], []
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            fo = od / f"{k}_{a.gate_name if a.gate else a.name}.wav"
            if not fo.exists() or not (E / f"{k}.cond").exists():
                continue
            y = load48(fo).astype(np.float32)[: int(SEC * N.SR)]
            x = load48(S / "d56/_work" / f"pre_{k}.wav" if a.gate else S / "d467" / f"{k}.wav").astype(np.float32)[n0:n0 + int(SEC * N.SR)]
            c = np.fromfile(E / f"{k}.cond", "<f4").reshape(-1, 31)
            rows.append((k, y, x, c[NP:, 26] > 0.5))
            xs.append(x)
        with Pool(6) as pl:
            hv = pl.map(harvest, xs)
        F = []
        lagsc = np.zeros(7)
        for (k, y, x, vinf), vw in zip(rows, hv):
            vo, vos, ao, eo = f1v(y)
            vi, vis, ai, _ = f1v(x)
            F.append((k, vo, vi, vis, vinf, vw, ao, ai, eo, vos))
            for j, L in enumerate(range(-3, 4)):
                lagsc[j] += agree(vinf, vis, L)
        best = int(np.argmax(lagsc)) - 3
        acc = {}
        per = []
        for k, vo, vi, vis, vinf, vw, ao, ai, eo, vos in F:
            vinf = vinf[best:] if best >= 0 else np.concatenate([np.zeros(-best, bool), vinf])
            n = min(len(vo), len(vi), len(vinf), len(vw))
            d = {"vo": vo[:n], "vi": vi[:n], "vis": vis[:n], "vinf": vinf[:n], "vw": vw[:n], "act": ao[:n], "ai": ai[:n], "eo": eo[:n], "vos": vos[:n]}
            for kk, vv in d.items():
                acc.setdefault(kk, []).append(vv)
            m = d["act"]
            per.append({"k": k, "vfrac_out": float(d["vo"][m].mean()), "vfrac_in": float(d["vi"][m].mean()), "vfrac_in_s": float(d["vis"][m].mean()),
                        "vfrac_inf": float(d["vinf"][m].mean()), "vfrac_in_w": float(d["vw"][m].mean()),
                        "d109s_out": float(d["vos"].sum() / max(m.sum(), 1)), "d109s_in": float(d["vis"].sum() / max(d["ai"].sum(), 1)),
                        "d109_out": float(d["vo"].sum() / max(m.sum(), 1)), "d109_in": float(d["vi"].sum() / max(d["ai"].sum(), 1)),
                        "vfrac_out_s": float(d["vos"][m].mean()),
                        "A_out": float((d["vo"] & m).sum() / max(m.sum(), 1)), "B_out": float((d["vo"] & ~m).sum() / max(m.sum(), 1)),
                        "A_in": float((d["vi"] & d["ai"]).sum() / max(d["ai"].sum(), 1)), "B_in": float((d["vi"] & ~d["ai"]).sum() / max(d["ai"].sum(), 1))})
        A = {kk: np.concatenate(vv) for kk, vv in acc.items()}
        m = A["act"]
        Ex = m & A["vo"] & ~A["vi"]
        ne = int(Ex.sum())

        def fr(z: np.ndarray) -> float:
            return round(float(z[Ex].mean()), 3) if ne else float("nan")

        out = {"n_utt": len(rows), "n_act": int(m.sum()), "lag_vinf_vs_vis_best": best,
               "vfrac_median": {q: round(float(np.median([p[q] for p in per])), 3) for q in ("vfrac_out", "vfrac_in", "vfrac_in_s", "vfrac_inf", "vfrac_in_w")},
               "E_frac_of_act": round(ne / max(int(m.sum()), 1), 4),
               "E_share": {"vis1_vw1": fr(A["vis"] & A["vw"]), "vis1": fr(A["vis"]), "vw1": fr(A["vw"]), "vw0_vinf1": fr(~A["vw"] & A["vinf"]), "vinf0": fr(~A["vinf"]), "vinf1": fr(A["vinf"])},
               "gen_P_vo1_given_vinf0": round(float(A["vo"][m & ~A["vinf"]].mean()), 3), "gen_P_vo0_given_vinf1": round(float((~A["vo"])[m & A["vinf"]].mean()), 3),
               "deficit_frac_of_act": round(float((m & ~A["vo"] & A["vi"]).sum()) / max(int(m.sum()), 1), 4)}
        out["loose_vs_strict_median"] = {q: round(float(np.median([p[q] for p in per])), 3) for q in ("d109_out", "d109_in", "d109s_out", "d109s_in", "vfrac_out_s", "vfrac_in_s")}
        for q in ("A_out", "B_out", "A_in", "B_in"):
            out.setdefault("AB_median", {})[q] = round(float(np.median([p[q] for p in per])), 3)
        out["AB_mean"] = {q: round(float(np.mean([p[q] for p in per])), 4) for q in ("A_out", "B_out", "A_in", "B_in")}
        Bo = ~A["act"] & A["vo"]
        nb = int(Bo.sum())
        out["Bout"] = {"n": nb, "in_inactive": round(float((~A["ai"])[Bo].mean()), 3) if nb else None, "vinf1": round(float(A["vinf"][Bo].mean()), 3) if nb else None,
                       "vi1": round(float(A["vi"][Bo].mean()), 3) if nb else None, "vw1": round(float(A["vw"][Bo].mean()), 3) if nb else None,
                       "e_rel_db_p10_p50_p90": [round(float(np.percentile(A["eo"][Bo], q)), 1) for q in (10, 50, 90)] if nb else None}
        res[s] = out
        print(s, json.dumps(out, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
