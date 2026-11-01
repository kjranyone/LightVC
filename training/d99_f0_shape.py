"""D99(学習なし): 変換の f0 の形(関門 (3))。製品経路の変換(<dir>/pNNNN_tNNN_<cond>.wav)と元の男声(SRC)を比べる。
  F1 の誤り = 元の男声で F1(製品の f0)対 教師 f0(harvest): オクターブ誤り(|Δ log2| > 0.5)・有声の取りこぼし / 誤検出。
  出力の形 = 出力の教師 f0 と、元の教師 f0 を一定の比で移したもの: 両方有声のフレームで log f0 の相関(形)・オクターブ外れ(|log2 比 − 中央値| > 0.5)の率・有声の一致率・
  有声区間内の 0.6 oct 超の跳び / 秒(出力・元・本物の女声 cen で比べる)。
出力 = results/conv_p0/<tag>.json。
    uv run python d99_f0_shape.py --dir <scratchpad>/d85_360k --conds C07 --tag d99
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--conds", default="C07")
    ap.add_argument("--tag", default="d99")
    a = ap.parse_args()
    import f0est as FE
    import f0hi as H
    import nvoc as N
    dev = "cuda"
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    fr = N.SR / N.HOP
    d = Path(a.dir)

    def tf0(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        f, _ = H.teacher_f0(x[: n * N.HOP].astype(np.float32), n)
        return f[:n].astype(np.float64)

    def jumps(f: np.ndarray) -> float:
        v = f > 0
        lf = np.log2(np.where(v, f, 1.0))
        dv = np.abs(np.diff(lf))[(v[1:] & v[:-1])]
        return float((dv > 0.6).sum() / (len(f) / fr))

    rows = []
    for src in sorted(d.glob("p*_SRC.wav")):
        k = src.name[: -len("_SRC.wav")]
        ti = int(re.search(r"t(\d+)", k).group(1))
        x, _ = sf.read(src, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        ft = tf0(x)
        with torch.no_grad():
            xt = torch.from_numpy(x)[None].to(dev)
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
            ff = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
        ff = np.pad(ff, (0, n - len(ff)))
        both = (ft > 0) & (ff > 0)
        oct_err = float(np.mean(np.abs(np.log2(ff[both] / ft[both])) > 0.5)) if both.any() else 0.0
        miss = float(np.mean((ft > 0) & (ff <= 0)) / max(np.mean(ft > 0), 1e-9))
        fa = float(np.mean((ft <= 0) & (ff > 0)) / max(np.mean(ft <= 0), 1e-9))
        row = {"key": k, "f1_oct_err": oct_err, "f1_voicing_miss": miss, "f1_voicing_fa": fa, "src_jumps": jumps(ft)}
        cen = d / f"cen_{ti:03d}.wav"
        if cen.exists():
            row["real_fem_jumps"] = jumps(tf0(sf.read(cen, dtype="float32")[0]))
        for c in a.conds.split(","):
            p = d / f"{k}_{c}.wav"
            if not p.exists():
                continue
            y, _ = sf.read(p, dtype="float32")
            fo = tf0(y)
            m = min(len(fo), len(ft))
            fo, fs_ = fo[:m], ft[:m]
            bb = (fo > 0) & (fs_ > 0)
            r = np.log2(fo[bb] / fs_[bb])
            med = float(np.median(r)) if len(r) else 0.0
            row[f"{c}_shape_corr"] = float(np.corrcoef(np.log2(fo[bb]), np.log2(fs_[bb]))[0, 1]) if bb.sum() > 10 else float("nan")
            row[f"{c}_octave_off"] = float(np.mean(np.abs(r - med) > 0.5)) if len(r) else float("nan")
            row[f"{c}_voicing_agree"] = float(np.mean((fo > 0) == (fs_ > 0)))
            row[f"{c}_jumps"] = jumps(fo)
            row[f"{c}_shift_oct"] = med
        rows.append(row)
        print(k, {kk: round(vv, 3) for kk, vv in row.items() if kk != "key"}, flush=True)
    keys = [kk for kk in rows[0] if kk != "key"]
    summ = {kk: round(float(np.nanmean([r.get(kk, np.nan) for r in rows])), 4) for kk in keys}
    print("SUMMARY", json.dumps(summ), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
