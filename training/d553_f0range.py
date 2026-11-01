"""D553(推論だけの診断・D552 の続き): 男声 → 女声の差 MT − FT は入力の高さ・声道の水準では動かない(D552)= 残りの候補は抑揚(高さの動きの幅)。
P43OP の dev(TTS 男声)の書き出しの log f0 の行を、有声のフレームで発話の前置き(操作者の参照の末尾 12 s)の有声の中央値のまわりに κ 倍(κ = 0.8・0.6)して描画(CPU・製品の設定)。
前置きの中央値は目標の声域へ移した後の値(列 25)。診断のみ(κ を製品に入れるなら因果な定数で)。比べるのは FT・MT(d467_paired_v1)。
    uv run python d553_f0range.py --kappas 0.8 0.6 --out ../results/conv_p0/d553
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

ROOT = Path(__file__).resolve().parent.parent
PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kappas", nargs="+", type=float, required=True)
    ap.add_argument("--set", default="dev")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cpu"
    torch.set_num_threads(6)
    st = torch.load(ROOT / "results/diag_ppgvoc43/product_002500_m0.pt", map_location="cpu", weights_only=False)
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
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        v = c[:, 26] > 0.5
        med = float(np.median(c[:2400][v[:2400], 25])) if v[:2400].sum() > 50 else float(np.median(c[v, 25]))
        spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        for kap in a.kappas:
            nm = f"K{int(round(kap * 100)):03d}"
            od = Path(a.out) / nm
            od.mkdir(parents=True, exist_ok=True)
            c2 = c.copy()
            c2[:, 25] = np.where(v, med + kap * (c[:, 25] - med), c[:, 25])
            y = P.render(g, c2, lg, spk, dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", k, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
