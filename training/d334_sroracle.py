"""D334: 今の生成器(ppgvoc18)で自己再構成(SR・実女声 30 本・本人の表)の偽物らしさのうち、包絡の動きの分はどれだけか(D281 を今の系で測り直す)。
SR の包絡の行に本人の本物の包絡の残差 γ·(包絡 − Σp T / 10)を足して描画(γ = 0 / 0.5 / 1)・ENV = 包絡の行を本物の包絡そのものに差し替え(文脈の符号器のずれも外す)・PER = 周期性の定数の行を本物のフレームごとの周期性 × 有声に・ENVPER = 両方(最初の 31 行 = rvoc3hi の条件が全て本物)・S / F = 残差の遅い成分(0.5 s の移動平均・ゼロ位相・診断)/ 速い成分(残差 − 遅い)を γ 倍で足す。
CP(写し合成・results/diag_e2e1/eval/h*_CP.wav)と同じ C2ST(eval_v1perp・各 30 本)で並べる。診断専用(本物の包絡は推論で得られない)。
    uv run python d334_sroracle.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d334
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
    ap.add_argument("--variants", nargs="+", default=["G0", "G05", "G1", "ENV"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    P.CTX["adapt"] = bool(st.get("adapt", False))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    sr, _ = P.eval_items(P.e2e_held(), st.get("spk", "table"), dev)
    groups = []
    with torch.no_grad():
        for name, gam in [(n_, v_) for n_, v_ in (("G0", 0.0), ("G05", 0.5), ("G1", 1.0), ("ENV", "env"), ("PER", "per"), ("ENVPER", "envper"), ("S05", "s0.5"), ("F05", "f0.5"), ("S1", "s1.0"), ("F1", "f1.0")) if n_ in a.variants]:
            od = O / name
            od.mkdir(parents=True, exist_ok=True)
            for nm, c, lg, spk in sr:
                if isinstance(gam, str) and gam[0] in "sf":
                    cond, exc, s_ = P.cond_exc(g, c, lg, spk, dev)
                    pp = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                    res = torch.from_numpy(c[:, 1:25].T.copy()).float().to(dev)[None] - torch.einsum("dk,bkt->bdt", torch.from_numpy(spk[0]).float().to(dev), pp) / 10
                    slow = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(res, (50, 49), mode="replicate"), 100, 1)
                    part = slow if gam[0] == "s" else res - slow
                    cond = torch.cat([cond[:, :1], cond[:, 1:25] + float(gam[1:]) * part, cond[:, 25:]], 1)
                    y = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
                    y = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
                elif isinstance(gam, str):
                    cond, exc, s_ = P.cond_exc(g, c, lg, spk, dev)
                    ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                    if "env" in gam:
                        cond = torch.cat([cond[:, :1], ct[:, 1:25], cond[:, 25:]], 1)
                    if "per" in gam:
                        cond = torch.cat([cond[:, :27], ct[:, 27:31] * ct[:, 26:27], cond[:, 31:]], 1)
                    y = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
                    y = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
                else:
                    y = P.render(g, c, lg, spk, dev, gam, spk[0])
                sf.write(od / f"{nm}.wav", y, N.SR)
            groups.append(f"{name}={od}/h*_SR.wav")
            print("rendered", name, len(sr), flush=True)
    groups.append(f"CP={P.ROOT / 'results/diag_e2e1/eval'}/h*_CP.wav")
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
