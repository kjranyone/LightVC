"""D327: 変換に残る男声の由来(D326b: 女声の入力からの変換と男声の入力からの変換を AUC 0.95〜1.0 で見分けられる)が、操作者のどの通り道から来るかを切り分ける。
変種ごとに両方の性別の入力へ同じ操作を施して描画し、d326_pair.py の 2 標本の判別(操作者の話者でまとめた分割)で AUC を測る。診断専用(発話全体の統計を使う変種あり)。
変種: base / c0flat(音量 c0n を有声のフレームの平均の定数に)/ f0const(有声の log f0 を平均の定数に)/ f0smooth(有声の log f0 を 0.2 s の移動平均に)/ hard(事後を argmax の one-hot に)/ all(c0flat + f0const + hard)。
    uv run python d327_leak.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d327
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

SETS = {"devvctk": P.H / "devvctk" / "devset.json", "devtts": None}
VARIANTS = ["base", "c0flat", "f0const", "f0smooth", "hard", "all"]
BASE_PPG = ["soft"]


def smooth(x: np.ndarray, m: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w)
    num = np.convolve(np.where(m, x, 0.0), k, "same")
    den = np.convolve(m.astype(np.float64), k, "same")
    return np.where(m, num / np.maximum(den, 1e-9), x)


def variant(c: np.ndarray, v: str) -> tuple[np.ndarray, str]:
    c = c.copy()
    m = c[:, 26] > 0.5
    if v in ("c0flat", "all") and m.any():
        c[:, 0] = c[m, 0].mean()
    if v in ("f0const", "all") and m.any():
        c[m, 25] = c[m, 25].mean()
    if v == "f0smooth" and m.any():
        c[:, 25] = smooth(c[:, 25].astype(np.float64), m, 40).astype(np.float32)
    return c, ("hard" if v in ("hard", "all") else BASE_PPG[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tts_spk", required=True, help="devtts の操作者の話者の割り当て(d326 の再構成)")
    ap.add_argument("--perm", type=int, default=50)
    ap.add_argument("--sets", nargs="+", default=["devvctk", "devtts"], help="devvctkm = 内容をそろえた組(D326c)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    P.CTX["adapt"] = bool(st.get("adapt", False))
    P.slow_from_ckpt(st)
    BASE_PPG[0] = st.get("ppg", "soft")
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    SETS["devtts"] = Path(a.tts_spk)
    SETS["devvctkm"] = P.H / "devvctkm" / "devset.json"
    pairs = []
    for s_ in a.sets:
        spkj = SETS[s_]
        ed = P.H / "e2e" / f"eval_{s_}"
        S = P.H / s_
        for v in a.variants:
            od = O / s_ / v
            od.mkdir(parents=True, exist_ok=True)
            for f in sorted(ed.glob("p0*.cond")):
                ti = int(f.stem.split("_t")[1][:3])
                c, lg = P.load_dump(ed, f.stem)
                cv, mode = variant(c, v)
                P.PPG = mode
                sf.write(od / f"{f.stem}_CV.wav", P.render(g, cv, lg, P.load_enroll(S / "d85_work" / f"en_{ti:03d}"), dev, T_op=P.op_tab(S, f.stem) if P.OPSLOW["on"] else None), N.SR)
            pairs.append(f"{s_}_{v}={od}/p00*_CV.wav|{od}/p01*_CV.wav|{spkj}")
            print("rendered", s_, v, flush=True)
    P.PPG = BASE_PPG[0]
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "d326_pair.py"), "--perm", str(a.perm), "--pairs", *pairs, "--out", str(O / "pair.json")], capture_output=True, text=True)
    print(r.stdout[-6000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
