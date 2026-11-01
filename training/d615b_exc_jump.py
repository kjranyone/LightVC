"""D615b(D615 の追加): 励起の高さ(条件の列 25 × 2^(k/12))で jump を分け直す。D615: k5 に残る jump(d109 と同じ: F1 の高さの隣り合う有声フレームの差 > 7.2 半音)の一つ一つを分ける。
立ち上がりからの位置(F1 の有声の連続の何フレーム目か)・倍読みか(跳びの片側の F1 が基準の高さ r から 9 半音超ずれる)・受け継ぎか(r 自身も 7.2 半音超跳ぶ・r が片側 0)・
跳びのフレームの H1 − H2(D607 の最小二乗)。r = 入力の harvest × 2^(k/12)(遅れ合わせ)。規則は results/conv_p0/d615/prereg.yaml。
    uv run python d615b_exc_jump.py --vers OURS5RY58H4G OURS5RY58G --out ../results/conv_p0/d615/exc_jump.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import NPROC, H
from d607_h1h2 import job

ONSET = 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vers", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    its = []
    for s in ("dev2", "dev3"):
        for nm in a.vers:
            k = int(re.match(r"OURS(\d+)", nm).group(1))
            for f in sorted((H / s / "d570" / nm).glob("p*_t*.wav")):
                key, ti = re.match(r"(p\d+_t(\d+))_", f.name).groups()
                its.append((nm, s, key, int(ti), k, f))
    with Pool(NPROC) as pl:
        res = pl.map(job, its, chunksize=2)
    import torch
    import f0est as FE
    ROOT = Path(__file__).resolve().parent.parent
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    lhz = np.log(FE.bin_hz().numpy())

    def f1track(x: np.ndarray) -> np.ndarray:
        x = x[: 8 * 48000].astype(np.float32)
        x = x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05
        n = len(x) // 240
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * 240])[None].to(dev)], 1), n))
            vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
            pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
        m = min(len(vo), pr.shape[1])
        kk = pr[:, :m].argmax(0)
        ii = np.clip(kk[None] + np.arange(-4, 5)[:, None], 0, pr.shape[0] - 1)
        w = np.take_along_axis(pr[:, :m], ii, 0)
        c = w.sum(0)
        return np.where(vo[:m] & (c >= 0.5), np.exp((w * lhz[ii]).sum(0) / np.maximum(c, 1e-12)), 0.0)

    import train_ppgvoc as P
    rep: dict = {"onset_frames": ONSET, "versions": {}}
    for nm in a.vers:
        ev, vsec, allh = [], 0.0, []
        cnt = {"exr_oct_onset": 0, "exr_n_onset": 0, "exr_oct_mid": 0, "exr_n_mid": 0}
        for (n_, s, key, ti, k, _), x, r, d12 in res:
            if n_ != nm:
                continue
            fv = f1track(x)
            c, _ = P.load_dump(P.H / "e2e" / f"eval_{s}_pre_nof0", key)
            c = c[2400:]
            n = min(len(fv), len(r), len(d12), len(c))
            fv, rr, dd = fv[:n], r[:n], d12[:n]
            ex = np.where(c[:n, 26] > 0.5, 200.0 * np.exp(c[:n, 25]) * 2 ** (k / 12), 0.0)
            erun = np.zeros(n, int)
            for i in range(n):
                erun[i] = erun[i - 1] + 1 if ex[i] > 0 and i > 0 else int(ex[i] > 0)
            okr = (ex > 0) & (rr > 0)
            oc = okr & (np.abs(12 * np.log2(np.where(okr, ex, 1.0) / np.where(okr, rr, 1.0))) > 9)
            cnt["exr_oct_onset"] += int((oc & (erun <= ONSET)).sum()); cnt["exr_n_onset"] += int((okr & (erun <= ONSET)).sum())
            cnt["exr_oct_mid"] += int((oc & (erun > ONSET)).sum()); cnt["exr_n_mid"] += int((okr & (erun > ONSET)).sum())
            v = fv > 0
            vsec += v.sum() / 200.0
            run = np.zeros(n, int)
            for i in range(n):
                run[i] = run[i - 1] + 1 if v[i] and i > 0 else int(v[i])
            allh.extend(dd[v & np.isfinite(dd)][::4].tolist())
            for i in range(1, n):
                if not (v[i] and v[i - 1]):
                    continue
                dst = 12 * np.log2(fv[i] / fv[i - 1])
                if abs(dst) <= 7.2:
                    continue
                e0 = 12 * np.log2(fv[i - 1] / rr[i - 1]) if rr[i - 1] > 0 else np.nan
                e1 = 12 * np.log2(fv[i] / rr[i]) if rr[i] > 0 else np.nan
                inh = (rr[i - 1] <= 0 or rr[i] <= 0) or abs(12 * np.log2(rr[i] / rr[i - 1])) > 7.2
                octv = (np.nan_to_num(abs(e0)) > 9) or (np.nan_to_num(abs(e1)) > 9)
                exj = (ex[i - 1] <= 0 or ex[i] <= 0) or abs(12 * np.log2(ex[i] / ex[i - 1])) > 7.2
                g0 = 12 * np.log2(fv[i - 1] / ex[i - 1]) if ex[i - 1] > 0 else np.nan
                g1 = 12 * np.log2(fv[i] / ex[i]) if ex[i] > 0 else np.nan
                ev.append({"t": f"{s}|{ti}", "up": bool(dst > 0), "pos": int(run[i - 1]), "onset": bool(run[i - 1] <= ONSET), "octave": bool(octv),
                           "inherited": bool(inh), "ex_jump": bool(exj), "f1_vs_ex": [None if not np.isfinite(g0) else round(float(g0), 1), None if not np.isfinite(g1) else round(float(g1), 1)], "e_up_side": float(np.nanmax([e0, e1])) if np.isfinite([e0, e1]).any() else None,
                           "h12": float(np.nanmin([dd[i - 1], dd[i]])) if np.isfinite([dd[i - 1], dd[i]]).any() else None})
        E = ev
        nE = max(len(E), 1)
        cls = {"onset_octave": sum(e["onset"] and e["octave"] and not e["inherited"] for e in E), "mid_octave": sum((not e["onset"]) and e["octave"] and not e["inherited"] for e in E),
               "inherited": sum(e["inherited"] for e in E), "other": sum((not e["octave"]) and not e["inherited"] for e in E)}
        hv = [e["h12"] for e in E if e["h12"] is not None]
        rep["versions"][nm] = {"events": len(E), "voiced_sec": round(vsec, 1), "jumps_per_voiced_sec": round(len(E) / max(vsec, 1e-3), 3),
                               "share_onset": round(sum(e["onset"] for e in E) / nE, 3), "share_octave": round(sum(e["octave"] for e in E) / nE, 3),
                               "share_inherited": round(sum(e["inherited"] for e in E) / nE, 3),
                               "E1_share_exjump_among_octave": round(sum(e["ex_jump"] for e in E if e["octave"]) / max(sum(e["octave"] for e in E), 1), 3),
                               "share_exjump_among_onset": round(sum(e["ex_jump"] for e in E if e["onset"]) / max(sum(e["onset"] for e in E), 1), 3),
                               "f1_matches_ex_both_sides": round(sum(1 for e in E if None not in e["f1_vs_ex"] and max(abs(e["f1_vs_ex"][0]), abs(e["f1_vs_ex"][1])) < 3) / nE, 3),
                               "ex_vs_r_octave_rate": {"onset": round(cnt["exr_oct_onset"] / max(cnt["exr_n_onset"], 1), 4), "mid": round(cnt["exr_oct_mid"] / max(cnt["exr_n_mid"], 1), 4)}, "share_up": round(sum(e["up"] for e in E) / nE, 3), "classes": cls,
                               "h12_at_events_median": round(float(np.median(hv)), 2) if hv else None, "h12_all_voiced_median": round(float(np.median(allh)), 2),
                               "pos_hist": np.bincount(np.minimum([e["pos"] for e in E], 40) if E else [0], minlength=41)[:41].tolist()}
        print(nm, json.dumps({k_: v_ for k_, v_ in rep["versions"][nm].items() if k_ != "pos_hist"}, ensure_ascii=False), flush=True)
        rep["versions"][nm]["events_list"] = E
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
