"""D600(推論だけ・腕 61 の起動前の検査): psd_disc.psgram の細かい構造 z が (T1) 高さだけの違いで動かず (T2) 偶数倍音の強調と (T3) 候補の奇数倍音の不足を見るか。
高さ: REFKC / EVEN = 描画の輪郭(cen の harvest × 狙い / 中央値)・候補 = 製品の指令 FE.decode(F1(SRC), 0.5) × 2^(k/12)(harvest(出力)との遅れ ±10)。
    uv run python d600_psd_check.py --sets dev2 dev3 --out ../results/diag_ppgvoc61/psd_check.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from d592_refkc import H, SR, boot, harvest

ROOT = Path(__file__).resolve().parent.parent
VERS = ["REFKC0", "REFKC5", "REFKC12", "EVEN5B00", "EVEN5B30", "EVEN5B60", "OURS5RY58G", "OURS5RY60G", "OURS12RY58G"]
PAIRS = [("REFKC12", "REFKC5"), ("REFKC0", "REFKC5"), ("EVEN5B30", "EVEN5B00"), ("EVEN5B60", "EVEN5B00"), ("OURS5RY58G", "REFKC5"), ("OURS5RY60G", "OURS5RY58G"), ("OURS12RY58G", "REFKC12")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    import f0est as FE
    import nvoc as N
    import psd_disc as PS
    from d6_unit_cov import load48
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(s1["net"])
    f1.eval()
    fr_ = FE.Front().to(dev)
    taps = PS.lpf_taps().to(dev)

    def dec(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        with torch.no_grad():
            v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x[: n * N.HOP])[None].to(dev)], 1), n))
            return FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)

    def prep(x: np.ndarray) -> np.ndarray:
        x = x[: int(8 * SR)].astype(np.float32)
        return x / (np.sqrt((x ** 2).mean()) + 1e-9) * 0.05

    def shift(r0: np.ndarray, lag: int) -> np.ndarray:
        r = np.roll(r0, lag)
        if lag > 0:
            r[:lag] = 0
        elif lag < 0:
            r[lag:] = 0
        return r

    def best_lag(r0: np.ndarray, h: np.ndarray) -> int:
        best = (0, -1.0)
        for lag in range(-10, 11):
            r = shift(r0, lag)
            n = min(len(r), len(h))
            b = (r[:n] > 0) & (h[:n] > 0)
            if b.sum() < 50:
                continue
            ag = float((np.abs(12 * np.log2(h[:n][b] / r[:n][b])) < 1).mean())
            if ag > best[1]:
                best = (lag, ag)
        return best[0]

    def zst(x: np.ndarray, f0w: np.ndarray) -> dict:
        f0 = f0w[1:]
        T = min(len(f0), len(x) // N.HOP)
        xt = torch.from_numpy(x[: T * N.HOP])[None].to(dev)
        ft = torch.from_numpy(f0[:T].astype(np.float32))[None].to(dev)
        with torch.no_grad():
            z, ok = PS.psgram(xt, ft, 0, None, taps=taps)
        return PS.z_stats(z, ok)

    rows: dict = {v: {} for v in VERS}
    for s in a.sets:
        S = H / s
        cache: dict = {}
        for v in VERS:
            k = int(re.match(r"(?:OURS|REFKC|EVEN)(\d+)", v).group(1))
            for f in sorted((S / "d570" / v).glob("p*_t*.wav")):
                m = re.match(r"(p\d+_t(\d+))_", f.name)
                key, ti = m.group(1), int(m.group(2))
                if key not in cache:
                    xs = prep(load48(S / "d56" / f"{key}_SRC.wav"))
                    fc = harvest(load48(S / "cen_reg" / f"cen_{ti:03d}.wav").astype(np.float64)[: 8 * SR])
                    hs = harvest(xs)
                    cache[key] = (fc, float(np.median(hs[hs > 0])), dec(xs))
                fc, mo, dc = cache[key]
                x = prep(load48(f))
                if v.startswith("OURS"):
                    c = dc * 2 ** (k / 12)
                    lag = best_lag(np.concatenate([[0.0], c]), harvest(x))
                    f0w = shift(np.concatenate([[0.0], c]), lag)
                else:
                    f0w = fc * mo * 2 ** (k / 12) / float(np.median(fc[fc > 0]))
                st = zst(x, f0w)
                if st["nf"] >= 20:
                    rows[v][(s, key)] = {"t": ti, **st}
            print(s, v, len(rows[v]), flush=True)
    rng = np.random.default_rng(600)
    rep: dict = {"sets": {v: {m: round(float(np.median([r[m] for r in rows[v].values()])), 4) for m in ("oe_z", "valley", "nf")} | {"n": len(rows[v])} for v in VERS}, "pairs": {}}
    for v in VERS:
        print(v, rep["sets"][v], flush=True)
    for A_, B_ in PAIRS:
        ks = sorted(set(rows[A_]) & set(rows[B_]))
        tg = np.array([f"{key[0]}|{rows[A_][key]['t']}" for key in ks])
        for m in ("oe_z", "valley"):
            d = np.array([rows[A_][key][m] - rows[B_][key][m] for key in ks])
            rep["pairs"][f"{A_}-{B_}:{m}"] = {"n": len(ks), "diff": round(float(d.mean()), 4), "ci95": boot(d, tg, rng)}
            print(f"{A_}-{B_}:{m}", rep["pairs"][f"{A_}-{B_}:{m}"], flush=True)
    P_ = rep["pairs"]
    dc_ = abs(P_["OURS5RY58G-REFKC5:oe_z"]["diff"])
    rep["T1"] = bool(abs(P_["REFKC12-REFKC5:oe_z"]["diff"]) < 0.5 * dc_ and abs(P_["REFKC0-REFKC5:oe_z"]["diff"]) < 0.5 * dc_)
    rep["T2"] = bool(P_["EVEN5B30-EVEN5B00:oe_z"]["ci95"][1] < 0)
    rep["T3"] = bool(P_["OURS5RY58G-REFKC5:oe_z"]["ci95"][1] < 0 and P_["OURS5RY60G-OURS5RY58G:oe_z"]["diff"] < 0)
    print("T1", rep["T1"], "T2", rep["T2"], "T3", rep["T3"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0 if rep["T1"] and rep["T2"] and rep["T3"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
