"""D547(D545 の高域の倍音の崩れの帰属): rvoc3hi を教師の条件(非因果: 教師の f0 = f0hi・PA の包絡(env_smooth は ckpt)・PA の周期性)で devff5 の入力 SRC(8 s)を写し合成(TEACH)。
Rust の因果の条件の写し合成 CP(gate_ff_v3)と d545 で比べる: TEACH も高域が崩れるならボコーダの限界・崩れないなら因果の前段(CT・F4・F1)が原因。
    uv run python d547_teacher_copy.py --out ../results/conv_p0/d547
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
import f0hi as FH
import nvoc as N
import pae as PA
import rvoc as R
import train_ppgvoc as P
import train_rvoc as TR
from d6_unit_cov import load48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff5")
    a = ap.parse_args()
    dev = "cuda"
    s_ = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    g = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
    g.load_state_dict(s_["ema"])
    g.eval()
    front = TR.Front("pae", TR.ckpt_env_smooth(s_, None)).to(dev)
    S = P.H / a.set
    od = Path(a.out) / "TEACH"
    od.mkdir(parents=True, exist_ok=True)
    done = {}
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        src = S / "d56" / f"{k}_SRC.wav"
        x = load48(src)[: 8 * N.SR].astype(np.float32)
        key = hash(x.tobytes())
        if key not in done:
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            f0, _ = FH.teacher_f0(x.astype(np.float64), n=n)
            f0 = f0.astype(np.float32)
            xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
            env = PA.envelope(xa, f0.astype(np.float64)).astype(np.float32)
            with torch.no_grad():
                fa = torch.from_numpy(f0)[None].to(dev)
                cond = front(torch.from_numpy(xa)[None].to(dev), fa, torch.from_numpy(env)[None].to(dev))
                y = g(cond, TR.excitation(fa, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
            done[key] = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
        sf.write(od / f"{k}_TEACH.wav", done[key], N.SR)
    print("rendered", len(done), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
