"""D343: ppgvoc21 の SR で、本物の遅い揺らぎ(z = 本物・SR 0.06)の上に残る写し合成(−0.73)までの差の運び手を探す天井の検査。
変種: ZR = z 本物 / ZR_F05・ZR_F1 = + 速い残差(残差 − 0.5 s 平均)× 0.5 / 1 / ZR_PER = + 周期性の行を本物 × 有声 / ENVR = 包絡の行を本物そのもの(文脈のずれと z なし)/ ENVR_PER = それ + 周期性。診断。
    uv run python d343_oracle21.py --ckpt ../results/diag_ppgvoc21/last_ema.pt --out ../results/conv_p0/d343
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=["ZR", "ZR_F05", "ZR_F1", "ZR_PER", "ENVR", "ENVR_PER"], help="+ ZR_PSLOW / ZR_PSLOW15 = 周期性の行に本物の遅い揺らぎ(有声の 0.5 s 平均 − 定数の per_p75 の区間平均の差)× 1 / 1.5・ZR_MID / ZR_MID05 = 包絡の中間の帯(0.15 s 平均 − 0.5 s 平均 = 約 1〜3 Hz)の本物の残差 × 1 / 0.5")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    sr, _ = P.eval_items(P.e2e_held(), "table", dev)
    groups = []
    with torch.no_grad():
        for v in a.variants:
            od = O / v
            od.mkdir(parents=True, exist_ok=True)
            for nm, c, lg, spk in sr:
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                s_ = P.spk_tensor(spk, dev)
                base = torch.einsum("bdk,bkt->bdt", s_[0], p) / 10
                res = ct[:, 1:25] - base
                fast = res - F.avg_pool1d(F.pad(res, (50, 49), mode="replicate"), 100, 1)
                if v.startswith("ENVR"):
                    on = P.SLOW["on"]
                    P.SLOW["on"] = False
                    cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], s_)
                    P.SLOW["on"] = on
                    cond = torch.cat([cond[:, :1], ct[:, 1:25], cond[:, 25:]], 1)
                else:
                    cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], s_, P.slow_real(ct[:, 1:25], base))
                    if "_MID" in v:
                        mid = F.avg_pool1d(F.pad(res, (15, 14), mode="replicate"), 30, 1) - F.avg_pool1d(F.pad(res, (50, 49), mode="replicate"), 100, 1)
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + (0.5 if v.endswith("05") else 1.0) * mid, cond[:, 25:]], 1)
                    if "_F" in v:
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + (0.5 if v.endswith("F05") else 1.0) * fast, cond[:, 25:]], 1)
                if "PSLOW" in v:
                    vv = ct[:, 26:27]
                    pr = ct[:, 27:31]
                    num = F.avg_pool1d(F.pad(pr * vv, (50, 49), mode="replicate"), 100, 1)
                    den = F.avg_pool1d(F.pad(vv, (50, 49), mode="replicate"), 100, 1).clamp(min=1e-3)
                    ps = num / den
                    m_ = (ps * vv).sum(-1, keepdim=True) / vv.sum(-1, keepdim=True).clamp(min=1)
                    cond = torch.cat([cond[:, :27], cond[:, 27:31] + (1.5 if v.endswith("15") else 1.0) * (ps - m_) * vv, cond[:, 31:]], 1)
                if v.endswith("PER"):
                    cond = torch.cat([cond[:, :27], ct[:, 27:31] * ct[:, 26:27], cond[:, 31:]], 1)
                f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = P.TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                y = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            groups.append(f"{v}={od}/h*_SR.wav")
            print("rendered", v, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
