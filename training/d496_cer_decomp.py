"""D496(診断・gate_ff_v2b の C FAIL の分解): devff23x の定常状態の条件(eval_devff23x_pre・d467_steady の書き出し)を使い、候補(P43OP)の目標の表だけを操作者自身の登録の表に替えた SELF を描画する。
f0(目標の声域へ移した軌跡)・事後・c0n・遅い揺らぎは CAND と同一 = CAND と SELF の差は目標の表だけ。CP(写し合成)と SELF の差は事後 → 表の経路。CER は d23_cer で別に測る。
    uv run python d496_cer_decomp.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d496
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff23x")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    od = Path(a.out) / "SELF"
    od.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"op_{r['pi']:04d}"), dev, T_op=P.op_tab(S, k))
        y = y[n0:n0 + 8 * N.SR]
        sf.write(od / f"{k}_SELF.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        n += 1
    print("rendered SELF", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
