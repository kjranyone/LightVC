"""D546(推論だけ・D545 の高域の倍音の崩れ): P43OP の周期性の行(目標の per_p75 × 有声・P43 は定数)の高域 2 帯(2.5〜5 kHz・5〜10 kHz)に固定の較正の足し算 δ を入れて描画する(製品の設定・定常状態・devff5)。
δ = 0.15・0.3(上限 1.5)。評価は d545(高域の倍音の明瞭さ)・d467_paired_v1 で別に。
    uv run python d546_perhf_knob.py --deltas 0.15 0.3 --out ../results/conv_p0/d546
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

PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", nargs="+", type=float, required=True)
    ap.add_argument("--bands", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--ckpt", default="../results/diag_ppgvoc43/product_002500_m0.pt")
    ap.add_argument("--set", default="devff5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192))
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    for dl in a.deltas:
        nm = f"HF{int(round(dl * 100)):02d}"
        od = Path(a.out) / nm
        od.mkdir(parents=True, exist_ok=True)
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            T, pp = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
            pp = pp.copy()
            pp[a.bands] = np.minimum(pp[a.bands] + dl, 1.5)
            y = P.render(g, c, lg, (T, pp), dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", nm, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
