"""D498(診断・D497 の続き): 包絡を本人の実測にしても CER が写し合成に届かない(SELF ALL 0.128 対 CP 0.034)。残る差 = 周期性(目標の定数 × 有声)と生成器。
版 = SELF_ALLP(行 = 操作者の包絡・周期性 = 操作者の実測)・FF_MP(製品の行・周期性 = 操作者の実測)・FF_LP20P(TAB + 20 Hz 以下の res・周期性 = 操作者の実測)・
CP_CPER(写し合成 rvoc3hi の周期性だけ目標の定数 × 有声)・CP_SPER(写し合成の周期性を操作者自身の有声フレームの 75 パーセンタイル × 有声 = 定数化だけの効果)。
    uv run python d498_cer_per.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d498
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
import rvoc as R
import train_ppgvoc as P
import train_rvoc as TR
from d497_cer_band import band

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
    s_ = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gcp = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
    gcp.load_state_dict(s_["ema"])
    gcp.eval()
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
        T_op = P.op_tab(S, k)
        out = {}
        with torch.no_grad():
            p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
            ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
            lk = torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10
            res = (ct[:, 1:25] - lk)[0].cpu().numpy()
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            exc = TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            vv = ct[:, 26:27]
            sp_op = P.spk_tensor(P.load_enroll(S / "d85_work" / f"op_{r['pi']:04d}"), dev)
            sp_tg = P.spk_tensor(P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev)
            cond = P.cond_ppg(p, ct[:, 0:1], ct[:, 25:26], vv, *sp_op)
            cond = torch.cat([cond[:, :1], ct[:, 1:25], cond[:, 25:27], ct[:, 27:31], cond[:, 31:]], 1)
            out["SELF_ALLP"] = g(cond, exc, sp_op)[0]
            zo = P.OPSLOW["gain"] * P.opslow_causal(ct[:, 1:25] - lk, P.OPSLOW["tau"])
            cond = g.make(p, ct[:, 0:1], ct[:, 25:26], vv, sp_tg, zo)
            out["FF_MP"] = g(torch.cat([cond[:, :27], ct[:, 27:31], cond[:, 31:]], 1), exc, sp_tg)[0]
            cond = P.cond_ppg(p, ct[:, 0:1], ct[:, 25:26], vv, *sp_tg)
            add = torch.from_numpy(band(res, "LP20").astype(np.float32)).to(dev)[None]
            out["FF_LP20P"] = g(torch.cat([cond[:, :1], cond[:, 1:25] + add, cond[:, 25:27], ct[:, 27:31], cond[:, 31:]], 1), exc, sp_tg)[0]
            out["CP_CPER"] = gcp(torch.cat([ct[:, :27], sp_tg[1][:, :, None] * vv], 1), exc)[0]
            pv = ct[0, 27:31][:, ct[0, 26] > 0.5]
            q = torch.quantile(pv, 0.75, dim=1) if pv.shape[1] else torch.zeros(4, device=dev)
            out["CP_SPER"] = gcp(torch.cat([ct[:, :27], q[None, :, None] * vv], 1), exc)[0]
        for v, y in out.items():
            y = y.cpu().numpy()[N.DELAY:][n0:n0 + 8 * N.SR]
            od = O / v
            od.mkdir(parents=True, exist_ok=True)
            sf.write(od / f"{k}_{v.replace('_', '')}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        n += 1
    print("rendered", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
