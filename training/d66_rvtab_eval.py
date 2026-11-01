"""D66: 出力部の追加学習(diag_rvtab: 表の包絡を条件に)の評価。評価の 30 目標(学習に入れていない)の女声の実音声(D37 の cen)を、製品と同じ条件で作り直す:
包絡 = Σp T(本人の Rust 登録 d53_en・C1 の事後・prime)・c0 = F2 + 因果 3 平均・周期性 = 登録の 75 パーセンタイルの定数 × 有声・f0 = 教師。
出力部を 2 つ(BASE = rvoc3hi 330k EMA・FT = 追加学習後の EMA)で同じ励起の種で鳴らす。出力 = d4b の形式(cen・pNNNN_tNNN_{BASE,FT}・SRC)。
    uv run python d66_rvtab_eval.py --s <scratchpad> --ft ../results/diag_rvtab/snap/ema_335k.pt --out_dir <scratchpad>/d66
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max", type=int, default=30)
    a = ap.parse_args()
    import c1_content as CC
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)

    def load_gen(p):
        st = torch.load(p, map_location="cpu", weights_only=False)
        g = R.RVoc(ch=st["cfg"]["ch"], kernels=tuple(st["cfg"]["kernels"]), dils=tuple(st["cfg"]["dils"]), d_cond=st["cfg"]["d_cond"]).to(dev)
        g.load_state_dict(st["ema"]); g.eval().requires_grad_(False)
        return g

    gens = {"BASE": load_gen(ROOT / "results/rvoc3hi/snap/ema_330k.pt"), "FT": load_gen(Path(a.ft))}
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    mf = CC.MelFront().to(dev)
    for ti in range(a.max):
        en = S / "d53_en" / f"en_{ti:03d}"
        T = torch.from_numpy(np.fromfile(en / "enroll.bin", dtype="<f4").reshape(24, -1)).to(dev)
        p75 = torch.tensor(json.loads((en / "enroll.json").read_text())["per_p75"], device=dev)
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = torch.from_numpy(f0[:n].astype(np.float32)).to(dev)[None]
        with torch.no_grad():
            m = mf(torch.from_numpy(T1.prime(x))[None].to(dev))
            p = c1(m)[..., -n:].softmax(1)
            c0n = C3.causal_avg(f2(m)[:, None], 3)[:, :, -n:]
            lf = torch.where(f0 > 0, torch.log(f0.clamp(min=1.0) / 200.0), torch.zeros_like(f0))[:, None]
            v = (f0 > 0).float()[:, None]
            cond = torch.cat([c0n, (T @ p[0])[None] / 10, lf, v, p75.clamp(0, 1.5)[None, :, None] * v], 1)
            for name, g in gens.items():
                y = g(cond, TR.excitation(f0, n * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"p{ti:04d}_t{ti:03d}_{name}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / f"p{ti:04d}_t{ti:03d}_SRC.wav")
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / f"cen_{ti:03d}.wav")
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
