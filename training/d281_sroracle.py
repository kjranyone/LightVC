"""D281: diag_ppgvoc1 の生成の欠け(保留の実女声の自己再構成 SR 2.1 対 写し合成 CP −0.7・D280)の中身を本物の値への差し替えで見積もる(学習なし)。
SR の入力の行を、同じクリップの Rust の写し合成の条件(--causal-ct)の値に差し替える: env = 包絡 c1..c24(表 Σ p T の代わり)・per = 周期性 4 行(目標の定数の代わり)・both。
    uv run python d281_sroracle.py --ckpt ../results/diag_ppgvoc1/last_ema.pt --out ../results/conv_p0/d281
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
import train_rvoc as TR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="base,env,per,both")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    held = P.e2e_held()
    sr_items, _ = P.eval_items(held)
    groups = []
    for v in a.variants.split(","):
        od = O / v
        od.mkdir(parents=True, exist_ok=True)
        for nm, c, lg, spk in sr_items:
            with torch.no_grad():
                cond, exc, stt = P.cond_exc(g, c, lg, spk, dev)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                if v in ("env", "both"):
                    cond[:, 1:25] = ct[:, 1:25]
                if v in ("per", "both"):
                    cond[:, 27:31] = ct[:, 27:31]
                y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
            sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        groups.append(f"{v}={od}/h*_SR.wav")
        print("rendered", v, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
