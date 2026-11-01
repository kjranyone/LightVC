"""D344: ppgvoc の初期値(rvoc3hi の重み・事後の入力は 0・文脈の符号器なし)の SR: 本物の 31 行(RV_REAL)・表の条件(RV_TAB)・表 + 乱数の遅い揺らぎ σ 1 / 1.5(RV_TAB_N1 / N15)。
写し合成 CP(Rust の前段 + rvoc3hi = −0.73)の再現と、表の条件の出発点を測る(描き方の劣化の仮説の較正)。診断。
    uv run python d344_rvoc3hi.py --out ../results/conv_p0/d344
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=str(P.ROOT / "results/diag_ppgvoc21/slow_stats.npz"))
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    P.PPG = "soft"
    P.CTX.update(on=False, ch=96, adapt=False)
    P.SLOW["on"] = False
    g = P.init_gen(dev, "table")
    g.eval()
    sr, _ = P.eval_items(P.e2e_held(), "table", dev)
    groups = []
    with torch.no_grad():
        for v in ("RV_REAL", "RV_TAB", "RV_TAB_N1", "RV_TAB_N15"):
            if v.startswith("RV_TAB_N"):
                P.slow_load(a.stats, 1.0 if v.endswith("N1") else 1.5)
            else:
                P.SLOW["on"] = False
            od = O / v
            od.mkdir(parents=True, exist_ok=True)
            for nm, c, lg, spk in sr:
                cond, exc, s_ = P.cond_exc(g, c, lg, spk, dev)
                if v == "RV_REAL":
                    ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                    cond = torch.cat([ct[:, :27], ct[:, 27:31] * ct[:, 26:27], cond[:, 31:]], 1)
                y = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            groups.append(f"{v}={od}/h*_SR.wav")
            print("rendered", v, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
