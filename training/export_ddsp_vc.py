"""DDSP-VC の重みを Rust(crates/lightvc-core/src/ddsp_vc.rs)へ書き出す+parity 用の固定入力と期待出力。

出力 <out>/:
  weights.bin      全テンソル f32 LE を名前順に連結(EMA 重み・state_dict のキー名そのまま)
  manifest.json    {name: [offset(要素), shape]}・定数(SR, HOP, N_MEL, K_HARM, ...)
  fixture/         x.f32(入力 2s)・f0.f32(因果 YIN フレーム)・ref.f32(参照 3s)・noise.f32([T,512])・y.f32(期待出力)・spk.f32

    CUDA_VISIBLE_DEVICES= uv run python export_ddsp_vc.py --ckpt ../results/ddsp_vc/last.pt --out ../results/ddsp_vc/export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as AD
import ddsp_vc as V

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/ddsp_vc/last.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/ddsp_vc/export"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m = V.DDSPVC(norm=bool(st.get("norm", False)))
    m.load_state_dict(st["ema"])
    m.eval()
    sd = m.state_dict()
    man = {"tensors": {}, "const": {"SR": V.SR, "HOP": V.HOP, "DELAY": V.DELAY, "N_MEL": V.N_MEL, "MEL_NFFT": V.MEL_NFFT,
                                    "K_HARM": V.K_HARM, "F_MAX": V.F_MAX, "NOISE_NFFT": V.NOISE_NFFT, "step": int(st["step"]),
                                    "NORM": bool(st.get("norm", False)), "NORM_PRIOR": V.NORM_PRIOR}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(sd):
            v = sd[k].detach().float().contiguous().numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import s0_artic as S
    it = S.held24()[0]
    x = it["x"].astype(np.float32)[:2 * V.SR]
    f0, _ = AD.causal_yin(x.astype(np.float64), voi_max=0.45)
    f0 = f0.astype(np.float32)[:len(x) // V.HOP]
    ref = S.held24()[1]["x"].astype(np.float32)[:3 * V.SR]
    T = len(x) // V.HOP
    noise = np.random.default_rng(0).standard_normal((T, V.NOISE_NFFT)).astype(np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x)[None]
        mel = m.front(xt)
        s = m.spk(m.front(torch.from_numpy(ref)[None]))
        y = m(mel, m.level(mel), torch.from_numpy(f0)[None], s, len(x), noise=torch.from_numpy(noise)[None])[0].numpy()
    for nm, arr in (("x", x), ("f0", f0), ("ref", ref), ("noise", noise), ("y", y), ("spk", s[0].numpy()), ("mel", mel[0].numpy())):
        arr.astype("<f4").tofile(out / "fixture" / f"{nm}.f32")
    print("exported", off, "floats ->", out, "| fixture y rms", float(np.sqrt((y ** 2).mean())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
