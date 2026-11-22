"""段 P95 step c3 の (c3-a)(c3-b)(results/snv_p95/prereg.yaml): 高さの番(snv_f0_guard.clean)を、devA の設計の側と学習の族で、物差しの pYIN と比べる(学習なし)。
bad = 声の水準のフレームのうち、前段が pYIN より 0.2 オクターブを超えて高い(両方が有声)か、前段だけが有声で f0 > 2.2 med。fixed = そのうち番の後に無声か pYIN から 0.2 オクターブ以内。
good = 前段と pYIN が 0.2 オクターブ以内で合うフレーム。touched = そのうち番が値を変えた率。pYIN は results/snv_p95/pyin に残す。
    cd training && uv run --no-sync python snv_diag/p95_f0guard_probe.py [n=64] [fam=40]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import snv_f0_guard as FG

ROOT = Path(__file__).resolve().parent.parent.parent
FEAT = ROOT / "data/snv_p_feat"
CACHE = ROOT / "results/snv_p95/pyin"
H = 240


def pyin(wav: Path, key: str, t: int) -> np.ndarray:
    p = CACHE / f"{key}.npy"
    if p.exists():
        return np.load(p)
    x, sr = sf.read(str(wav), dtype="float32")
    assert sr == 48000
    fp, _, _ = librosa.pyin(x.astype(np.float64), fmin=50, fmax=1000, sr=48000, frame_length=2048, hop_length=H, center=True)
    fp = np.nan_to_num(fp[1:t + 1])
    fp = np.pad(fp, (0, t - len(fp))).astype(np.float32)
    np.save(p, fp)
    return fp


def one(npz: Path, wav: Path, key: str, med: float) -> dict:
    f0 = np.load(npz)["f0"].astype(np.float32)
    x, _ = sf.read(str(wav), dtype="float32")
    t = min(len(f0), len(x) // H)
    f0 = f0[:t]
    r = 20 * np.log10(np.sqrt((x[: t * H].reshape(t, H) ** 2).mean(1)) + 1e-9)
    loud = r >= np.percentile(r, 90) - 25
    fp = pyin(wav, key, t)
    fc = FG.clean(f0, med)
    both = loud & (f0 > 0) & (fp > 0)
    lr = np.zeros(t)
    lr[both] = np.log2(f0[both] / fp[both])
    bad = (both & (lr > 0.2)) | (loud & (f0 > 0) & (fp == 0) & (f0 > 2.2 * med))
    fixed = bad & ((fc == 0) | ((fp > 0) & (np.abs(np.log2(np.maximum(fc, 1) / np.maximum(fp, 1))) <= 0.2)))
    good = both & (np.abs(lr) <= 0.2)
    return {"bad": int(bad.sum()), "fixed": int(fixed.sum()), "good": int(good.sum()), "touched": int((good & (fc != f0)).sum()), "voiced": int((loud & (f0 > 0)).sum()),
            "changed": int((loud & (fc != f0)).sum()), "to_unvoiced": int((loud & (f0 > 0) & (fc == 0)).sum())}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, fam = int(opt.get("n", 64)), int(opt.get("fam", 40))
    CACHE.mkdir(parents=True, exist_ok=True)
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    res = {"devA": {}, "fam_M": {}, "fam_F": {}}
    add = lambda a, b: {k: a.get(k, 0) + b[k] for k in b}
    for s in [s for s in man["speakers"] if s["split"] == "devA"][:n]:
        med = FG.median_f0([np.load(FEAT / f"cv/{s['speaker']}_{c['name']}.npz")["f0"] for c in s["enroll"]])
        acc = {}
        for c in s["src"][:2]:
            acc = add(acc, one(FEAT / f"cv/{s['speaker']}_{c['name']}.npz", FEAT / f"cv/{s['speaker']}_{c['name']}.wav", f"cv_{s['speaker']}_{c['name']}", med))
        res["devA"][s["speaker"]] = acc | {"med": med}
    e1 = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]
    for sex in ("M", "F"):
        for v in [v for v in e1 if v["sex"] == sex][:fam]:
            med = FG.median_f0([np.load(FEAT / f"e1/{v['voice']}/{t}.npz")["f0"] for t in ("t00", "t03", "t04", "t05", "t06", "t07", "t08", "t09")])
            acc = {}
            for t in ("t01", "t02"):
                acc = add(acc, one(FEAT / f"e1/{v['voice']}/{t}.npz", ROOT / f"data/snv_e1/{v['voice']}/{t}.wav", f"e1_{v['voice']}_{t}", med))
            res[f"fam_{sex}"][v["voice"]] = acc | {"med": med}
    summ = {}
    for g, rows in res.items():
        tot = {k: sum(r[k] for r in rows.values()) for k in ("bad", "fixed", "good", "touched", "voiced", "changed", "to_unvoiced")}
        summ[g] = tot | {"fixed_rate": tot["fixed"] / max(tot["bad"], 1), "touched_rate": tot["touched"] / max(tot["good"], 1), "bad_share_of_voiced": tot["bad"] / max(tot["voiced"], 1),
                         "changed_share_of_voiced": tot["changed"] / max(tot["voiced"], 1),
                         "fixed_rate_spk_median": float(np.median([r["fixed"] / max(r["bad"], 1) for r in rows.values()])),
                         "touched_rate_spk_max": float(max(r["touched"] / max(r["good"], 1) for r in rows.values()))}
    (ROOT / "results/snv_p95/f0guard_probe.json").write_text(json.dumps({"summary": summ, "speakers": res}, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
