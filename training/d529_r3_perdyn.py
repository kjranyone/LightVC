"""D529(推論だけの検査・学習なし・学び直しの起動前の I の検査): rvoc3hi の重み(事後の列 = 0 = 表だけの条件)に、変換の経路と同じ条件(目標の表の包絡・遅い揺らぎ OPSLOW 1.5・定常状態)を与え、
周期性の行を DYN = 書き出しの列 27..30(目標の水準 + 操作者の F4 の動き)・CONST = 目標の per_p75 × 有声 の 2 通りで devff23x に描画する。
DYN ≫ CONST なら rvoc3hi は周期性の行の動きを出力の息っぽさに写す = 周期性の動きを最初から入れて rvoc3hi から学び直す根拠。DYN ≈ CONST なら行だけでは追従は来ない(写し合成 CP の 0.66 は操作者の包絡が運ぶ)。
評価は d524_aper_track2.py・d527_clear_breathy.py で別に。
    uv run python d529_r3_perdyn.py --out ../results/conv_p0/d529
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff23x")
    ap.add_argument("--slow_ckpt", default="../results/diag_ppgvoc54/product_002500_m0.pt")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.slow_ckpt, map_location="cpu", weights_only=False)
    P.PPG = "soft"
    P.CTX.update(on=False)
    g = P.init_gen(dev, "table")
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    O = Path(a.out)
    for nm, dyn in (("R3DYN", True), ("R3CONST", False)):
        P.PERDYN["on"] = dyn
        od = O / nm
        od.mkdir(parents=True, exist_ok=True)
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
            y = P.render(g, c, lg, spk, dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", nm, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
