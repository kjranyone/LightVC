"""D530(推論だけの検査・学習なし・D529 の続き): rvoc3hi(事後の列 = 0)+ 変換の経路の条件 + 周期性の動き(R3DYN = 0.477)に、操作者の包絡の残差 res = E_op − Σp T_op / 10 を
選んだ DCT の次数だけ「そのまま」与える(その次数は遅い揺らぎ OPSLOW の代わりに res・残りの次数は OPSLOW 1.5 のまま)。
次数の組 = c1(傾き)・c1..c2・c1..c4・c1..c24(全部 = 操作者の包絡 + 表の話者差 ≈ 写し合成 CP 0.658)。写し合成との差 0.18 を運ぶのがどの次数かを見る。
評価は d524_aper_track2.py で別に。
    uv run python d530_r3_envres.py --out ../results/conv_p0/d530
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
import train_rvoc as TR

PRE = 12.0
SELS = {"E1": [0], "E2": [0, 1], "E4": [0, 1, 2, 3], "E24": list(range(24))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff23x")
    ap.add_argument("--slow_ckpt", default="../results/diag_ppgvoc54/product_002500_m0.pt")
    ap.add_argument("--sels", nargs="+", default=list(SELS))
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.slow_ckpt, map_location="cpu", weights_only=False)
    P.PPG = "soft"
    P.CTX.update(on=False)
    g = P.init_gen(dev, "table")
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    P.PERDYN["on"] = True
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    O = Path(a.out)
    for nm in a.sels:
        sel = SELS[nm]
        od = O / nm
        od.mkdir(parents=True, exist_ok=True)
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
            T_op = P.op_tab(S, k)
            with torch.no_grad():
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                stt = P.spk_tensor(spk, dev)
                stt = (stt[0], ct[:, 27:31])
                res = ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10
                zo = P.OPSLOW["gain"] * P.opslow_causal(res, P.OPSLOW["tau"])
                zo[:, sel] = res[:, sel]
                cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, zo)
                f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
            y = y[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", nm, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
