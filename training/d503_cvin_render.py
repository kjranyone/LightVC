"""D503: 連続の内容の特徴の入力(diag_ppgvoc49 / 49c・CVIN)の生成器を devff23x の定常状態の条件で描画する(製品の設定 = 遅い揺らぎ 利得 1.5)。
特徴 = train_ppgvoc.cvin_frames(C1 と同じ規約の前置き + d467 の音声)。版 = FF(目標の表)・SELF(操作者の表)。ckpt に cvin が無ければ特徴なし(49c の対照)。
    uv run python d503_cvin_render.py --ckpt ../results/diag_ppgvoc49/product_002500_m0.pt --name P49 --out ../results/conv_p0/d503
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
import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48

PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff23x")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if st.get("cvin"):
        P.cvin_setup(str(Path(__file__).parent / st["cvin"]) if not Path(st["cvin"]).is_absolute() else st["cvin"], dev)
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
    O = Path(a.out)
    n = 0
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        cvf = None
        if P.CVIN["on"]:
            x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
            cvf = P.cvin_frames(torch.from_numpy(TC.prime(x))[None].to(dev), len(lg))
        for mode in ("FF", "SELF"):
            spk = P.load_enroll(S / "d85_work" / (f"en_{r['ti']:03d}" if mode == "FF" else f"op_{r['pi']:04d}"))
            y = P.render(g, c, lg, spk, dev, T_op=P.op_tab(S, k), cvf=cvf)[n0:n0 + 8 * N.SR]
            od = O / f"{a.name}{mode}"
            od.mkdir(parents=True, exist_ok=True)
            sf.write(od / f"{k}_{a.name}{mode}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        n += 1
    print("rendered", a.name, n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
