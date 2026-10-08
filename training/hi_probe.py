"""1kHz 超の評価区間(data/hi_eval・hi_eval_set.py)での出力部の写し合成の評価(current/f0_range.md §3-4)。
区間ごとに、f0 = 新しい教師(f0hi)と f0 = 既存の教師(A のみ・1kHz 超を取りこぼす)の 2 通りで描画し、元音声と比べる:
  level_gap_db: 置換フレーム(f0hi が B を採った所)の出力 RMS − 元の RMS(dB・平均)
  band_gap_db: 置換フレームの帯域(0–0.5・0.5–1・1–2.5・2.5–5・5–10kHz)のパワーの出力 − 元(dB)
  logmel・pesq・am_db・hf_db: 区間全体(eval_nvoc.metrics)
    uv run python hi_probe.py --ckpt ../results/rvoc2am2/snap/ema_300k.pt --out ../results/rvoc2am2/hi_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
import pae as PA
import rvoc as R
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
BANDS = ((0, 500), (500, 1000), (1000, 2500), (2500, 5000), (5000, 10000))


def item(x: np.ndarray, f0: np.ndarray) -> dict:
    n = len(x) // N.HOP
    x = x[:n * N.HOP].astype(np.float32)
    f0 = np.pad(f0[:n], (0, max(0, n - len(f0)))).astype(np.float32)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    return {"x": x, "xa": xa, "f0_h": f0, "env": PA.envelope(xa, f0)}


def gaps(y: np.ndarray, x: np.ndarray, frames: np.ndarray) -> tuple[float, list]:
    n = min(len(x), len(y)) // N.HOP * N.HOP
    fr = frames[frames < n // N.HOP]
    if len(fr) == 0:
        return float("nan"), [float("nan")] * len(BANDS)
    idx = fr[:, None] * N.HOP + np.arange(N.HOP)[None]
    rms = lambda a: 10 * np.log10((a[idx] ** 2).mean() + 1e-12)
    seg = lambda a: np.abs(np.fft.rfft(a[idx[:, :240]] * np.hanning(240), n=2048, axis=1)) ** 2
    f = np.fft.rfftfreq(2048, 1 / N.SR)
    Sx, Sy = seg(x).mean(0), seg(y).mean(0)
    return float(rms(y) - rms(x)), [float(10 * np.log10(Sy[(f >= lo) & (f < hi)].sum() / (Sx[(f >= lo) & (f < hi)].sum() + 1e-20) + 1e-20)) for lo, hi in BANDS]


_MEL = None


def logmel_hi(y: np.ndarray, x: np.ndarray, frames: np.ndarray, dev: str) -> float:
    """置換フレーム(1kHz 超の有声)だけの |log-mel(出力) − log-mel(元)| の平均(128 帯・1024 点・hop 240)。区間全体の logmel は無声・無音が混ざって 1kHz 超の改善を拾いにくい。"""
    global _MEL
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(N.SR, 1024, 1024, N.HOP, n_mels=128, power=1.0).to(dev)
    n = min(len(x), len(y)) // N.HOP * N.HOP
    lx = (_MEL(torch.from_numpy(x[:n]).to(dev)) + 1e-5).log()
    ly = (_MEL(torch.from_numpy(y[:n].astype(np.float32)).to(dev)) + 1e-5).log()
    fr = frames[frames < lx.shape[-1]]
    return float((lx[:, fr] - ly[:, fr]).abs().mean()) if len(fr) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env_smooth", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    g = R.RVoc(ch=st["cfg"]["ch"], kernels=tuple(st["cfg"]["kernels"]), dils=tuple(st["cfg"]["dils"]), d_cond=st["cfg"]["d_cond"]).to(dev)
    g.load_state_dict(st["ema"])
    g.eval()
    fr = TR.Front("pae", TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
    idx = json.loads((ROOT / "data/hi_eval/index.json").read_text())["items"]
    rows = {"f0_new": [], "f0_old": []}
    for r in idx:
        z = np.load(ROOT / "data/hi_eval" / f"{r['name']}.npz")
        x, fnew, fold = z["x"].astype(np.float32), z["f0"], z["f0_old"]
        rep = np.where((fnew != fold) & (fnew >= 950))[0]
        for key, f0 in (("f0_new", fnew), ("f0_old", fold)):
            it = item(x, f0)
            with torch.no_grad():
                y = TR.render(g, fr, it, it["f0_h"], dev)[N.DELAY:]
            m = E.metrics(y, it["x"], 0, dev)
            lg, bg = gaps(y, it["x"], rep)
            rows[key].append({"name": r["name"], "n_rep": int(len(rep)), **{k: float(v) for k, v in m.items()}, "level_gap_db": lg, "band_gap_db": bg, "logmel_hi": logmel_hi(y, it["x"], rep, dev)})
    rep_ = {}
    for key, rs in rows.items():
        rep_[key] = {"n": len(rs), **{k: round(float(np.nanmean([r[k] for r in rs])), 3) for k in ("logmel", "logmel_hi", "pesq", "am_db", "hf_db", "level_gap_db")},
                     "band_gap_db(0-.5,.5-1,1-2.5,2.5-5,5-10k)": [round(float(np.nanmean([r["band_gap_db"][i] for r in rs])), 2) for i in range(len(BANDS))],
                     "level_gap_ge_-3dB_frac": round(float(np.mean([r["level_gap_db"] >= -3 for r in rs if np.isfinite(r["level_gap_db"])])), 3)}
    print(json.dumps(rep_, ensure_ascii=False, indent=1))
    Path(a.out).write_text(json.dumps({"summary": rep_, "rows": rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
