"""converter.md C0-9: 製品の f0(生の因果 YIN)の代価を、誤りの種類ごとに分解する(出力部 R1 A の EMA 重み・held21 の写し合成・学習なし)。

包絡と周期性は harvest の f0 で分析した値に固定し(製品では変換器が出す)、パルスと log f0 だけを替える:
  harvest        正解(上限)
  yin            生の因果 YIN(製品の f0 推定)
  yin_voice_h    有声判定だけ harvest(YIN が無声で harvest が有声のフレームは harvest の値・逆は 0)= 有声判定の誤りを除く
  yin_oct_h      YIN の値を harvest の ±0.5 オクターブへ折り返す = オクターブ誤りを除く
  yin_oct_voice  両方を除く = 残りは遅れと細かい誤差
  h_lag1 / h_lag2  harvest を 1 / 2 フレーム遅らせる = 遅れだけの代価
  h_jit20        harvest に 20 cent の白色の揺れ = 細かい誤差だけの代価

    uv run python c09_f0_cost.py --ckpt ../results/diag_rvoc_r/snap/ema_20k.pt --out ../results/conv_c0/c09_f0_cost.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import rvoc as R
import train_rvoc as TR


def variants(fh: np.ndarray, fy: np.ndarray, rng: np.random.Generator) -> dict:
    vh, vy = fh > 0, fy > 0
    fold = fy.copy()
    both = vh & vy
    d = np.log2(np.where(both, fy, 1.0) / np.where(both, fh, 1.0))
    fold[both] = fy[both] * 2.0 ** (-np.round(d[both]))
    yv = np.where(vh, np.where(vy, fy, fh), 0.0)
    yov = np.where(vh, np.where(vy, fold, fh), 0.0)
    lag = lambda f, k: np.concatenate([np.zeros(k), f[:-k]])
    jit = np.where(vh, fh * 2.0 ** (rng.standard_normal(len(fh)) * 20 / 1200), 0.0)
    return {"harvest": fh, "yin": fy, "yin_voice_h": yv, "yin_oct_h": np.where(vy, fold, 0.0), "yin_oct_voice": yov,
            "h_lag1": lag(fh, 1), "h_lag2": lag(fh, 2), "h_jit20": jit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env_smooth", type=float, default=None, help="旧 ckpt は明示が要る")
    a = ap.parse_args()
    import eval_nvoc as E
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=st["cfg"]["ch"], d_cond=st["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(st["ema"])
    gen.eval()
    front = TR.Front(st.get("cond", "pae"), TR.ckpt_env_smooth(st, getattr(a, "env_smooth", None))).to(dev)
    items = TR.held(dev)
    rng = np.random.default_rng(0)
    rows: dict = {}
    stats = {"oct_err_frac": [], "voice_miss_frac": [], "voice_false_frac": []}
    with torch.no_grad():
        for it in items:
            fh, fy = it["f0_h"].astype(np.float64), it["f0_y"].astype(np.float64)
            vh, vy = fh > 0, fy > 0
            both = vh & vy
            if both.any():
                stats["oct_err_frac"].append(float((np.abs(np.log2(fy[both] / fh[both])) > 0.5).mean()))
            stats["voice_miss_frac"].append(float((vh & ~vy).sum() / max(1, vh.sum())))
            stats["voice_false_frac"].append(float((~vh & vy).sum() / max(1, (~vh).sum())))
            for name, f in variants(fh, fy, rng).items():
                y = TR.render(gen, front, it, f.astype(np.float32), dev)
                m = E.metrics(y, it["x"], N.DELAY, dev)
                rows.setdefault(name, []).append(m["pesq"])
    rep = {"ckpt": a.ckpt, "n": len(items), "pesq": {k: round(float(np.nanmean(v)), 4) for k, v in rows.items()},
           "yin_errors": {k: round(float(np.mean(v)), 4) for k, v in stats.items()}}
    base = rep["pesq"]["harvest"]
    rep["cost_vs_harvest"] = {k: round(base - v, 4) for k, v in rep["pesq"].items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
