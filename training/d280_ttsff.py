"""D280: diag_ppgvoc1(5k の EMA)の操作者の効果が男声に固有か TTS の入力に固有かを分ける。組 devtts(D265: 目標 = dev の 30 人・操作者 = 女声 TTS v3 p00・男声 TTS v3 p01)
を同じ系(Rust の書き出し・目標の登録の表)で鳴らし、変換 FT / MT と入力そのもの FT_SRC / MT_SRC を v1 / v1⊥ で測る。
    uv run python d280_ttsff.py --ckpt ../results/diag_ppgvoc1/last_ema.pt --out ../results/conv_p0/d280
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devtts")
    ap.add_argument("--reps", action="store_true", help="目標ごとに 2 本ある組(devvctk・D326)で、1 本目(番号が偶数)と 2 本目(奇数)を別の群として測る")
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
    print("ckpt step", st["step"], "ppg", P.PPG, flush=True)
    ed = P.H / "e2e" / f"eval_{a.set}"
    S = P.H / a.set
    od = O / "conv"
    od.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(ed.glob("p0*.cond")):
        ti = int(f.stem.split("_t")[1][:3])
        c, lg = P.load_dump(ed, f.stem)
        sf.write(od / f"{f.stem}_CV.wav", P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{ti:03d}"), dev), N.SR)
        n += 1
    print("rendered", n, flush=True)
    groups = [f"FT={od}/p00*_CV.wav", f"MT={od}/p01*_CV.wav", f"FT_SRC={S}/d56/p00*_SRC.wav", f"MT_SRC={S}/d56/p01*_SRC.wav"]
    if a.reps:
        groups = [f"{k}{n}_{r}={d}/p0{g}[0-5][{par}]_t*_{x}.wav" for k, g in (("F", "0"), ("M", "1")) for n, d, x in (("T", od, "CV"), ("T_SRC", S / "d56", "SRC")) for r, par in (("a", "02468"), ("b", "13579"))]
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--one_per_target", "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
