"""D536(推論だけ): 候補の重みを、別の Rust の設定で書き出した定常状態の条件(<H>/e2e/eval_<set>_pre_<cfg>・d534_voicing_sweep.py)で描画する(製品の設定・遅い揺らぎ OPSLOW 1.5・元の 8 s)。
有声の判定の設定(H75 = --voicing-hyst 0.7,0.5・V70 = 0.7,0.7)で息の端のピッチ(d533)と関門の軸を測るため。
    uv run python d536_render_dump.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --cfgs H75 V70 --out ../results/conv_p0/d536v
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
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff5")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    S = P.H / a.set
    n0 = int(PRE * N.SR)
    for cfg in a.cfgs:
        ed = P.H / "e2e" / f"eval_{a.set}_pre_{cfg}"
        od = Path(a.out) / cfg
        od.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{cfg}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            n += 1
        print("rendered", cfg, n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
