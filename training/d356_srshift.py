"""D356: 自己再構成(SR・本人の表と事後)で log f0 だけを ± s 半音動かすと v1⊥ はどう変わるか(D355 後の誤差の分析: FF の発話ごとの決定値が声域の移動の向きと相関 −0.53 / −0.62 = 下げる変換ほど偽物らしい)。
下げ(−4)だけ悪化 = 生成器が自然より低い高さを描くのが苦手(出力の側)・両方とも同程度 = 高さの不一致一般・どちらも不変 = FF の向きの効果は入力の側(操作者の事後・音量)。診断・ppgvoc24 の候補(因果な揺らぎ σ 1.5)。
    uv run python d356_srshift.py --ckpt ../results/diag_ppgvoc24/product_s15_007500.pt --out ../results/conv_p0/d356
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", nargs="+", type=float, default=[0.0, -4.0, 4.0, -8.0])
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
        for s in a.shifts:
            name = f"S{s:+.0f}"
            od = O / name
            od.mkdir(parents=True, exist_ok=True)
            for nm, c, lg, spk in sr:
                c2 = c.copy()
                v = c2[:, 26] > 0.5
                c2[v, 25] += s * np.log(2.0) / 12.0
                sf.write(od / f"{nm}.wav", P.render(g, c2, lg, spk, dev), N.SR)
            groups.append(f"{name}={od}/h*_SR.wav")
            print("rendered", name, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
