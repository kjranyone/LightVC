"""D497(診断・明瞭さの損失の運び手): devff23x の定常状態の条件で、包絡の行を作り替えて描画する。res = 操作者の包絡 − Σp T_操作者 / 10(単位の平均からのずれ)。
版 = TAB(行 = Σp T / 10 のみ・文脈の符号器と遅い揺らぎなし)・LP3 / LP8 / LP20(TAB + 時間方向の低域通過した res・零位相)・ALL(TAB + res)・MHP8(製品の行 + 8 Hz 以上の res)。
SELF = 表も操作者自身(TAB + ALL = 操作者の包絡そのもの)・FF = 目標の表。零位相の濾波は非因果 = 診断のみ(製品の手ではない)。CER は d23_cer で別に測る。
    uv run python d497_cer_band.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d497
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
import train_rvoc as TR

PRE = 12.0
VARS = ("TAB", "LP3", "LP8", "LP20", "ALL", "MHP8")


def band(res: np.ndarray, v: str) -> np.ndarray:
    fps = N.SR / N.HOP
    if v == "ALL":
        return res
    if v == "TAB":
        return np.zeros_like(res)
    if v.startswith("LP"):
        b, a = butter(2, float(v[2:]) / (fps / 2))
        return filtfilt(b, a, res, axis=-1)
    b, a = butter(2, float(v[3:]) / (fps / 2), btype="high")
    return filtfilt(b, a, res, axis=-1)


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
    O = Path(a.out)
    n = 0
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        T_op = P.op_tab(S, k)
        with torch.no_grad():
            p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
            ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
            res = (ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10)[0].cpu().numpy()
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            exc = TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            for mode in ("SELF", "FF"):
                spk = P.load_enroll(S / "d85_work" / (f"op_{r['pi']:04d}" if mode == "SELF" else f"en_{r['ti']:03d}"))
                sp = P.spk_tensor(spk, dev)
                for v in VARS:
                    add = torch.from_numpy(band(res, v).astype(np.float32)).to(dev)[None]
                    if v == "MHP8":
                        zo = P.OPSLOW["gain"] * P.opslow_causal(ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10, P.OPSLOW["tau"])
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], sp, zo)
                    else:
                        cond = P.cond_ppg(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], *sp)
                    cond = torch.cat([cond[:, :1], cond[:, 1:25] + add, cond[:, 25:]], 1)
                    y = g(cond, exc, sp)[0].cpu().numpy()[N.DELAY:][n0:n0 + 8 * N.SR]
                    od = O / f"{mode}_{v}"
                    od.mkdir(parents=True, exist_ok=True)
                    sf.write(od / f"{k}_{mode}{v}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        n += 1
    print("rendered", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
