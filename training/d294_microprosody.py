"""D294: 男声 → 女声の 0.5 秒の局所の芯(D293・v1⊥ 約 0.9・全ての腕で不動)の通り道を、推論の入力の韻律の細かい構造を因果に滑らかにして切り分ける(学習なし)。
base / f0s5 = 有声の log f0 を因果な 5 フレーム平均(約 25 ms の揺れを均す)/ f0ema = 有声の log f0 を因果な走行平均(1 フレーム 0.05・約 100 ms より速い動きを消す)/
c0s5 = c0n を因果な 5 フレーム平均。目標ごとに 1 本(30 本)を書き出し、d286_window.py --perp --eq_dir で 0.5 / 2 / 8 秒の v1⊥ を測る。
    uv run python d294_microprosody.py --ckpt ../results/diag_ppgvoc6/last_ema.pt --out <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def causal_mean(x: np.ndarray, w: int) -> np.ndarray:
    xp = np.r_[np.full(w - 1, x[0]), x]
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="base,f0s5,f0ema,c0s5")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"] = bool(st.get("ctx", False))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    print("ckpt step", st["step"], "ctx", P.CTX["on"], flush=True)
    _, pv_items = P.eval_items(P.e2e_held())
    seen, items = set(), []
    for it in pv_items:
        t = it[0].split("_t")[1][:3]
        if t not in seen:
            seen.add(t)
            items.append(it)
    for v in a.variants.split(","):
        od = O / v
        od.mkdir(parents=True, exist_ok=True)
        for nm, c, lg, spk in items:
            c = c.copy()
            vo = c[:, 26] > 0.5
            if v == "f0s5":
                buf: list = []
                for t in range(len(c)):
                    if vo[t]:
                        buf = (buf + [c[t, 25]])[-5:]
                        c[t, 25] = float(np.mean(buf))
                    else:
                        buf = []
            elif v == "f0ema":
                m = None
                for t in range(len(c)):
                    if vo[t]:
                        m = c[t, 25] if m is None else 0.95 * m + 0.05 * c[t, 25]
                        c[t, 25] = m
                    else:
                        m = None
            elif v == "c0s5":
                c[:, 0] = causal_mean(c[:, 0], 5)
            sf.write(od / f"{nm}.wav", P.render(g, c, lg, spk, dev), N.SR)
        print("rendered", v, len(items), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
