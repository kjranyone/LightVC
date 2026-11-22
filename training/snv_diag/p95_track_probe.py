"""段 P95 step a(results/snv_p95/prereg.yaml): 前段 v3 の高さの読みを、本物の男声(devA の設計の側)と学習の族(snv_e1 の男声と女声)で、物差しの pYIN と比べる(学習なし)。
量 = 声の水準のフレームのうち f0 が話者の中央値の 2.2 倍を超える率 R_hi・0.6 倍を下回る率 R_lo(前段と pYIN)・両方が有声のフレームの食い違いの率 G。
出力 = results/snv_p95/track_probe.json。
    cd training && uv run --no-sync python snv_diag/p95_track_probe.py [n=64] [fam=40]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
FEAT = ROOT / "data/snv_p_feat"
H = 240


def one(npz: Path, wav: Path) -> dict:
    f0 = np.load(npz)["f0"].astype(np.float64)
    x, sr = sf.read(str(wav), dtype="float32")
    assert sr == 48000
    t = min(len(f0), len(x) // H)
    f0 = f0[:t]
    r = 20 * np.log10(np.sqrt((x[: t * H].reshape(t, H) ** 2).mean(1)) + 1e-9)
    lvl = np.percentile(r, 90)
    loud = r >= lvl - 25
    fp, _, _ = librosa.pyin(x.astype(np.float64), fmin=50, fmax=1000, sr=48000, frame_length=2048, hop_length=H, center=True)
    fp = np.nan_to_num(fp[1 : t + 1] if len(fp) > t else np.pad(fp, (0, t - len(fp))))[:t]
    return {"f0": f0, "fp": fp, "loud": loud}


def rates(us: list[dict]) -> dict:
    f0 = np.concatenate([u["f0"] for u in us])
    fp = np.concatenate([u["fp"] for u in us])
    loud = np.concatenate([u["loud"] for u in us])
    out = {}
    for nm, f in (("front", f0), ("pyin", fp)):
        m = loud & (f > 0)
        med = float(np.median(f[m])) if m.sum() else 0.0
        out[nm] = {"med": med, "n": int(m.sum()), "r_hi": float((f[m] > 2.2 * med).mean()) if m.sum() else 0.0, "r_lo": float((f[m] < 0.6 * med).mean()) if m.sum() else 0.0}
    both = loud & (f0 > 0) & (fp > 0)
    out["g"] = float((np.abs(np.log2(f0[both] / fp[both])) > 0.2).mean()) if both.sum() else 0.0
    out["g_hi"] = float((np.log2(f0[both] / fp[both]) > 0.2).mean()) if both.sum() else 0.0
    out["v_front_only"] = float(((f0 > 0) & (fp == 0) & loud).sum() / max(loud.sum(), 1))
    return out


def med(rows: dict, f) -> float:
    return float(np.median([f(r) for r in rows.values()]))


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, fam = int(opt.get("n", 64)), int(opt.get("fam", 40))
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dev = [s for s in man["speakers"] if s["split"] == "devA"][:n]
    res = {"devA": {}, "fam_M": {}, "fam_F": {}}
    for s in dev:
        us = [one(FEAT / f"cv/{s['speaker']}_{c['name']}.npz", FEAT / f"cv/{s['speaker']}_{c['name']}.wav") for c in s["src"][:2]]
        res["devA"][s["speaker"]] = rates(us)
    e1 = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]
    for sex in ("M", "F"):
        for v in [v for v in e1 if v["sex"] == sex][:fam]:
            us = [one(FEAT / f"e1/{v['voice']}/{t}.npz", ROOT / f"data/snv_e1/{v['voice']}/{t}.wav") for t in ("t01", "t02")]
            res[f"fam_{sex}"][v["voice"]] = rates(us)
    bad = lambda r, k: r[k]["r_hi"] + r[k]["r_lo"]
    summ = {}
    for g, rows in res.items():
        summ[g] = {"n": len(rows), "front_bad_median": med(rows, lambda r: bad(r, "front")), "front_bad_mean": float(np.mean([bad(r, "front") for r in rows.values()])),
                   "front_r_hi_median": med(rows, lambda r: r["front"]["r_hi"]), "front_r_lo_median": med(rows, lambda r: r["front"]["r_lo"]),
                   "pyin_bad_median": med(rows, lambda r: bad(r, "pyin")), "pyin_bad_mean": float(np.mean([bad(r, "pyin") for r in rows.values()])),
                   "g_median": med(rows, lambda r: r["g"]), "g_hi_median": med(rows, lambda r: r["g_hi"]), "v_front_only_median": med(rows, lambda r: r["v_front_only"])}
    rows = json.loads((ROOT / "results/snv_p93/dev_rows_devA_s0_k3g.json").read_text())["rows"]
    rows = next(iter(rows.values()))
    cer = {}
    for r in rows:
        if r["group"] == "devA":
            cer.setdefault(r["spk"], []).append(r["cer"])
    sp = [s for s in res["devA"] if s in cer]
    rho = {k: float(spearmanr([f(res["devA"][s]) for s in sp], [np.mean(cer[s]) for s in sp]).statistic) for k, f in
           (("front_bad", lambda r: bad(r, "front")), ("g", lambda r: r["g"]), ("front_r_hi", lambda r: r["front"]["r_hi"]), ("front_r_lo", lambda r: r["front"]["r_lo"]))}
    out = {"summary": summ, "rho_cer_k3g": rho, "n_rho": len(sp), "speakers": res}
    p = ROOT / "results/snv_p95/track_probe.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"summary": summ, "rho_cer_k3g": rho, "n_rho": len(sp)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
