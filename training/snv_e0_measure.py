"""E0 の測定(results/snv_e0/prereg.yaml): 種の結合 P1 の対 (Y, XC) と対照 (Y, XI) の時刻・高さ・有声・大きさの一致。
    uv run python snv_e0_measure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import soxr
import torch

import c1_content as CC
import nvoc as N

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_e0"
WAV = OUT / "wav"
WIN_F, STEP_F, MAXLAG = 80, 20, 30
LAG_OK = 2


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(1)
    return (x if sr == N.SR else soxr.resample(x, sr, N.SR)).astype(np.float32)


@torch.no_grad()
def post(x: np.ndarray, c1n, mfc, dev: str) -> np.ndarray:
    T = len(x) // N.HOP
    a = torch.from_numpy(np.concatenate([np.zeros(N.WIN - N.HOP, np.float32), x[: T * N.HOP]]))[None].to(dev)
    return c1n(mfc(a))[..., -T:].softmax(1)[0].float().cpu().numpy().T


def energy_db(x: np.ndarray, T: int) -> np.ndarray:
    e = np.convolve(x ** 2, np.ones(N.HOP) / N.HOP, "same")[:: N.HOP][:T]
    return 10 * np.log10(e + 1e-10)


def f0(x: np.ndarray, T: int) -> np.ndarray:
    x16 = soxr.resample(x, N.SR, 16000).astype(np.float64)
    f, t = pw.harvest(x16, 16000, f0_floor=60.0, f0_ceil=900.0, frame_period=1000.0 * N.HOP / N.SR)
    f = f[:T]
    return np.pad(f, (0, T - len(f)))


def local_lags(qy: np.ndarray, qx: np.ndarray, act: np.ndarray) -> list:
    qy = qy / np.linalg.norm(qy, axis=1, keepdims=True)
    qx = qx / np.linalg.norm(qx, axis=1, keepdims=True)
    T = len(qy)
    out = []
    for s in range(MAXLAG, T - WIN_F - MAXLAG, STEP_F):
        if act[s: s + WIN_F].mean() < 0.5:
            continue
        sims = [float((qy[s: s + WIN_F] * qx[s + l: s + l + WIN_F]).sum(1).mean()) for l in range(-MAXLAG, MAXLAG + 1)]
        b = int(np.argmax(sims))
        out.append({"lag": b - MAXLAG, "peak": sims[b], "at0": sims[MAXLAG]})
    return out


def pair_stats(y: np.ndarray, x: np.ndarray, qy, qx) -> dict:
    T = min(len(qy), len(qx))
    ey, ex = energy_db(y, T), energy_db(x, T)
    act = ey > ey.max() - 35
    lags = local_lags(qy[:T], qx[:T], act)
    fy, fx = f0(y, T), f0(x, T)
    both = act & (fy > 0) & (fx > 0)
    d = 12 * np.log2(fy[both] / fx[both]) if both.sum() > 20 else np.array([])
    k = float(np.median(d)) if len(d) else float("nan")
    res = float(1.4826 * np.median(np.abs(d - k))) if len(d) else float("nan")
    au = act | (ex > ex.max() - 35)
    return {"lags": lags, "k": k, "f0_res": res, "n_both": int(both.sum()),
            "vuv_agree": float(((fy > 0) == (fx > 0))[act].mean()),
            "e_corr": float(np.corrcoef(ey[au], ex[au])[0, 1]),
            "f0y_med": float(np.median(fy[act & (fy > 0)])) if (act & (fy > 0)).sum() else float("nan"),
            "f0x_med": float(np.median(fx[act & (fx > 0)])) if (act & (fx > 0)).sum() else float("nan")}


def pool(rows: list, key: str) -> dict:
    L = np.array([w["lag"] for r in rows for w in r[key]["lags"]])
    P = np.array([w["peak"] for r in rows for w in r[key]["lags"]])
    A0 = np.array([w["at0"] for r in rows for w in r[key]["lags"]])
    g = lambda f: float(np.nanmedian([r[key][f] for r in rows]))
    return {"n_win": int(len(L)), "frac_lag_ok": float((np.abs(L) <= LAG_OK).mean()), "frac_y_leads": float((L > LAG_OK).mean()),
            "frac_x_leads": float((L < -LAG_OK).mean()), "lag_ms_p05_p50_p95": [float(np.percentile(L, q)) * 5.0 for q in (5, 50, 95)],
            "peak_med": float(np.median(P)), "at0_med": float(np.median(A0)), "f0_res_med": g("f0_res"), "k_med": g("k"),
            "vuv_agree_med": g("vuv_agree"), "e_corr_med": g("e_corr"), "f0x_med": g("f0x_med"), "f0y_med": g("f0y_med")}


def main() -> int:
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"])
    c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    its = json.loads((WAV / "items.json").read_text())
    rows = []
    for it in its:
        p = WAV / f"i{it['item']:03d}"
        if not Path(f"{p}_XI.wav").exists():
            continue
        y, xc, xi = (load(Path(f"{p}_{s}.wav")) for s in ("Y", "XC", "XI"))
        qy, qc, qi = (post(a, c1n, mfc, dev) for a in (y, xc, xi))
        rows.append({"item": it["item"], "style": it["style"] or "plain", "XC": pair_stats(y, xc, qy, qc), "XI": pair_stats(y, xi, qy, qi)})
    res = {"n_items": len(rows), "all": {k: pool(rows, k) for k in ("XC", "XI")}}
    for st in sorted({r["style"] for r in rows}):
        rs = [r for r in rows if r["style"] == st]
        res[st] = {k: pool(rs, k) for k in ("XC", "XI")}
    a = res["all"]
    res["A1_pass"] = bool(a["XC"]["frac_lag_ok"] >= 0.9 and a["XC"]["frac_lag_ok"] - a["XI"]["frac_lag_ok"] >= 0.3)
    (OUT / "e0_measure.json").write_text(json.dumps({"summary": res, "rows": rows}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
