"""段 P114(results/snv_p114/prereg.yaml)の準備(学習なし): 各発話の「周期ごとの強さ」を作る。教師の音の波形から、前段の高さ f0(フレームごと)の周期 T0 の 1/2・1・2・3 倍の遅れでの周期性(1 − 正規化した差分関数・0〜1)。
先読みなし(前段の高さの追跡と同じ左寄せの窓)。無声のフレームと、遅れが範囲の外のものは 0。出力 = data/snv_p114_sal/<e1|e1nv>/<声>/<発話>.npy [4, T] float16。
    cd training && SNV_DEV=cuda uv run --no-sync python snv_p114_sal.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import snv_p0_front as P
import train_snv_p1 as T
from train_snv_p97 import items_for

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p114_sal"
MULT = (0.5, 1.0, 2.0, 3.0)
W = 600


def salience(x: np.ndarray, f0: np.ndarray, dev: str = "cpu") -> np.ndarray:
    n = len(f0)
    xt = torch.from_numpy(x[: n * P.H].astype(np.float32))
    dn, _ = P.cmndf(P.lowpass(xt).to(dev), W)
    dn = dn[:n].float()
    f = torch.from_numpy(f0.astype(np.float32)).to(dn.device)
    out = torch.zeros(len(MULT), n, device=dn.device)
    t0 = P.SR / f.clamp(min=1.0)
    for i, m in enumerate(MULT):
        lag = (t0 * m).round().long()
        ok = (f > 0) & (lag >= P.TMIN + 2) & (lag <= P.TMAX - 2)
        idx = lag.clamp(P.TMIN + 2, P.TMAX - 2)
        mn = torch.stack([dn.gather(1, (idx + o)[:, None])[:, 0] for o in (-2, -1, 0, 1, 2)], 0).min(0).values
        out[i] = torch.where(ok, (1.0 - mn).clamp(0.0, 1.0), torch.zeros_like(mn))
    return out.cpu().numpy().astype(np.float32)


def main() -> int:
    import os
    dev = os.environ.get("SNV_DEV", "cpu")
    sp = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    items = items_for(sp["train"] + sp["val"])
    n = 0
    for w, f in items:
        dst = OUT / f.relative_to(T.FEAT).with_suffix(".npy")
        if dst.exists():
            continue
        x, sr = sf.read(str(w), dtype="float32")
        assert sr == P.SR
        f0 = np.load(f)["f0"]
        m = min(len(f0), len(x) // P.H)
        s = np.zeros((len(MULT), len(f0)), np.float32)
        s[:, :m] = salience(x, f0[:m], dev)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst.with_suffix(".tmp.npy"), s.astype(np.float16))
        dst.with_suffix(".tmp.npy").rename(dst)
        n += 1
        if n % 1000 == 0:
            print(n, "/", len(items), flush=True)
    (OUT / "manifest.json").write_text(json.dumps({"stage": "P114", "mult": MULT, "w": W, "what": "教師の音の、前段の高さの周期の 1/2・1・2・3 倍の遅れでの周期性(0〜1)"}, ensure_ascii=False))
    print("done", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
