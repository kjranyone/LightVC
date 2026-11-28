import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FEAT = ROOT / "data/snv_p_feat"


def runs(v: np.ndarray, m: int) -> list[tuple[int, int]]:
    out, s = [], None
    for i, b in enumerate(np.append(v, False)):
        if b and s is None:
            s = i
        elif not b and s is not None:
            if i - s >= m:
                out.append((s + 2, i - 2))
            s = None
    return out


def stats(files: list[Path]) -> dict:
    d2c, d2f, d2h, df, flips, nfr, cc = [], [], [], [], 0, 0, []
    for p in files:
        d = np.load(p)
        f0, env = d["f0"].astype(np.float64), d["env"].astype(np.float64)
        v = f0 > 0
        flips += int(np.abs(np.diff(v.astype(int))).sum())
        nfr += len(v)
        lf = np.log2(np.maximum(f0, 1.0))
        for a, b in runs(v, 14):
            c0, l = env[0, a:b], lf[a:b]
            x = c0[1:-1] - 0.5 * (c0[:-2] + c0[2:])
            y = l[1:-1] - 0.5 * (l[:-2] + l[2:])
            h = env[1:9, a:b]
            z = np.sqrt(((h[:, 1:-1] - 0.5 * (h[:, :-2] + h[:, 2:])) ** 2).sum(0))
            d2c.append(x)
            d2f.append(y)
            d2h.append(z)
            df.append(np.diff(l))
    x, y, z, g = (np.concatenate(q) for q in (d2c, d2f, d2h, df))
    return {"files": len(files), "voiced_run_frames": len(x), "c0_alt_rms": round(float(np.sqrt((x**2).mean())), 4), "c0_alt_p95": round(float(np.percentile(np.abs(x), 95)), 4),
            "shape_alt_rms": round(float(np.sqrt((z**2).mean())), 4), "lf0_alt_rms_cent": round(float(np.sqrt((y**2).mean()) * 1200), 2), "lf0_alt_p95_cent": round(float(np.percentile(np.abs(y), 95) * 1200), 2),
            "f0_jump_rate": round(float((np.abs(g) > 0.15).mean()), 5), "flips_per_s": round(flips / nfr * 200, 2), "corr_c0alt_f0alt": round(float(np.corrcoef(np.abs(x), np.abs(y))[0, 1]), 3)}


def main() -> None:
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    cv = [FEAT / f"cv/{s['speaker']}_{c['name']}.npz" for s in [x for x in man["speakers"] if x["split"] == "devA"][:64] for c in s["src"][:2]]
    val = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    sets = {"devA(本物・CV)": cv, "E1 検証・女": [FEAT / f"e1/{v}/{t}.npz" for v in val if v.startswith("ef") for t in ("t03", "t06", "t09")][:120],
            "E1 検証・男": [FEAT / f"e1/{v}/{t}.npz" for v in val if v.startswith("em") for t in ("t01", "t02", "t03", "t04", "t05", "t06", "t07", "t08", "t09")]}
    for k in ("jvnv",):
        fs = sorted((FEAT / k).glob("*.npz")) or sorted((FEAT / k).glob("*/*.npz"))
        sets["JVNV 男"] = [p for p in fs if "M1" in p.name or "M2" in p.name][:120]
        sets["JVNV 女"] = [p for p in fs if "F1" in p.name or "F2" in p.name][:120]
    res = {k: stats(v) for k, v in sets.items() if v}
    for k, v in res.items():
        print(k, json.dumps(v, ensure_ascii=False))
    json.dump(res, open(ROOT / "results/snv_p98/feat_rough.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
