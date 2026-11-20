"""段 D5(paper_snv rev41 §6・results/snv_d5/prereg.yaml lever_b の Pb_sig・学習なし): 生成の痕跡が前段 v3 の特徴を動かす大きさを、録音の鎖 v1 の乱しの大きさと比べる。
痕跡 = 同じ発話の対(JVNV の元の音 対 Irodori の符号器を通した音 jvnv_codec・+ 透かし jvnv_codecwm・data/snv_p_feat)。鎖 = E1 のきれいな音 対 鎖の版(data/snv_p_feat/d5_cache/<fam>・既定 f2v2)。
推論の E が読む 29 次元(包絡 c0〜c24 / 10・周期性 4)ごとに、両方で有声のフレームの差の RMS(c0 は発話ごとの平均の差を除く = 音量の差を数えない)。Pb_sig = 全ての次元で 鎖の RMS ≥ 痕跡の RMS。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_sig_probe.py [fam=f2v2] [out=results/snv_d5/sig_probe_<fam>.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/snv_p_feat"


def pair(a: Path, b: Path) -> tuple | None:
    da, db = np.load(a), np.load(b)
    n = min(len(da["f0"]), len(db["f0"]))
    ya, yb = T.to_y(da["env"][:, :n], da["per"][:, :n], da["f0"][:n]), T.to_y(db["env"][:, :n], db["per"][:, :n], db["f0"][:n])
    m = (da["f0"][:n] > 0) & (db["f0"][:n] > 0)
    if m.sum() < 50:
        return None
    d = (yb - ya)[:, m]
    d[0] -= d[0].mean()
    flip = float(((da["f0"][:n] > 0) != (db["f0"][:n] > 0)).mean())
    cents = float(np.median(np.abs(1200 * np.log2(db["f0"][:n][m] / da["f0"][:n][m]))))
    return (d ** 2).mean(1), flip, cents


def group(pairs: list) -> dict:
    R = [r for r in (pair(a, b) for a, b in pairs if a.exists() and b.exists()) if r is not None]
    ms = np.sqrt(np.mean([r[0] for r in R], 0))
    return {"n": len(R), "rms": ms.round(4).tolist(), "voicing_flip": round(float(np.mean([r[1] for r in R])), 4), "f0_abs_cents_median": round(float(np.median([r[2] for r in R])), 1)}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    names = sorted(p.stem for p in (FEAT / "jvnv_codec").glob("*.npz"))
    res = {"codec": group([(FEAT / "jvnv" / f"{n}.npz", FEAT / "jvnv_codec" / f"{n}.npz") for n in names]),
           "codec_wm": group([(FEAT / "jvnv" / f"{n}.npz", FEAT / "jvnv_codecwm" / f"{n}.npz") for n in names])}
    fam = opt.get("fam", "f2v2")
    ch = sorted((FEAT / "d5_cache" / fam).glob("*_k0.npz"))
    res["chain"] = group([(FEAT / "e1" / p.stem.split("_")[0] / f"{p.stem.split('_')[1]}.npz", p) for p in ch])
    rc, rk = np.array(res["codec_wm"]["rms"]), np.array(res["chain"]["rms"])
    res["ratio_chain_over_codecwm"] = (rk / (rc + 1e-9)).round(2).tolist()
    res["Pb_sig"] = bool((rk >= rc).all())
    dims = ["c0"] + [f"c{i}" for i in range(1, 25)] + [f"per{i}" for i in range(4)]
    for k in ("codec", "codec_wm", "chain"):
        r = res[k]
        print(k, "n", r["n"], "有声の反転", r["voicing_flip"], "高さの差(セント)", r["f0_abs_cents_median"], "RMS c0", r["rms"][0], "c1〜c12 の平均", round(float(np.mean(r["rms"][1:13])), 4), "c13〜c24", round(float(np.mean(r["rms"][13:25])), 4), "per", r["rms"][25:])
    print("鎖 / 痕跡 の比の最小", min(res["ratio_chain_over_codecwm"]), dims[int(np.argmin(res["ratio_chain_over_codecwm"]))], "Pb_sig", res["Pb_sig"])
    (ROOT / opt.get("out", f"results/snv_d5/sig_probe_{fam}.json")).write_text(json.dumps(res | {"dims": dims, "fam": fam}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
