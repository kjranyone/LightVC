"""D472(results/conv_p0/d472/prereg.yaml): 音量の条件 c0n を因果な 1 次の平滑(a 0.6・時定数 約 10 ms)にして、実男声の入力の変換の音量の揺れ(shimmer)が本物の範囲に入るか。
版 = NONE / C0S を候補(P43OP)で描画 → <out>/<set>/<版>/(対の評価)・<scratch>/<set>/d472/<版>/(d109)。組 = devreal・devvctkm の男声・devff2。
    uv run python d472_c0smooth.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d472
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

A = 0.6


def smooth(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    acc = x[0]
    for t in range(len(x)):
        acc = A * acc + (1 - A) * x[t]
        y[t] = acc
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    sets = {"devreal": [r for r in json.loads((P.H / "devreal" / "devset.json").read_text())["pairs"]],
            "devvctkm": [r for r in json.loads((P.H / "devvctkm" / "devset.json").read_text())["pairs"] if r["kind"].startswith("M")],
            "devff2": [r for r in json.loads((P.H / "devff2" / "devset.json").read_text())["pairs"]]}
    for s, rows in sets.items():
        ed = P.H / "e2e" / f"eval_{s}"
        n = 0
        for r in rows:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            for v in ("NONE", "C0S"):
                cc = c.copy()
                if v == "C0S":
                    cc[:, 0] = smooth(c[:, 0])
                y = P.render(g, cc, lg, P.load_enroll(P.H / s / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(P.H / s, k) if P.OPSLOW["on"] else None)
                for d in (O / s / v, P.H / s / "d472" / v):
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(d / f"{k}_{v}.wav", y, N.SR)
            n += 1
        print("rendered", s, n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
