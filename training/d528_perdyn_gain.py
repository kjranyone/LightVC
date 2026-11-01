"""D528(推論だけの検査・学習なし): 周期性の行の動きの幅 κ。腕 54(--per_dyn・周期性の一致の損失)の生成器は最初の層の周期性の列の重みが rvoc3hi の約半分(1.43 → 0.77・事後の 200 列に重みが分かれた)=
周期性の動きへの感度が弱い。推論で 行 = clamp(目標の per_p75 + κ·(書き出しの列 27..30 − 目標の per_p75), 0, 1.5) × 有声(κ = 1 が 54 の製品の設定)を κ = 2・3 で devff23x に描画する
(製品の設定・定常状態・遅い揺らぎ 1.5)。評価は d524(追従)・d527(はっきり ⇔ かすれ)・d467_paired_v1(自然さ)で別に。
    uv run python d528_perdyn_gain.py --ckpt ../results/diag_ppgvoc54/product_002500_m0.pt --kappas 2 3 --out ../results/conv_p0/d528
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
    ap.add_argument("--kappas", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff23x")
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
    assert P.PERDYN["on"], "--per_dyn の重みを与える"
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    O = Path(a.out)
    for kap in a.kappas:
        nm = f"K{int(round(kap * 10)):02d}"
        od = O / nm
        od.mkdir(parents=True, exist_ok=True)
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
            tg = np.clip(spk[1], 0.0, 1.5)
            c = c.copy()
            v = c[:, 26] > 0.5
            c[:, 27:31] = np.where(v[:, None], np.clip(tg[None] + kap * (c[:, 27:31] - tg[None]), 0.0, 1.5), 0.0)
            y = P.render(g, c, lg, spk, dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", nm, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
