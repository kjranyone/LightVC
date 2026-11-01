"""D449: 推論の遅い音色の揺らぎの設定(σ・因果な FIR の乱数 か 非因果の乱数)が女声 → 女声の局所(0.5 s)の自然さを変えるか。
D448: 学習の中の評価の描画(σ 1.0・非因果)の 43 は 0.5 s で −1.18・製品の設定(σ 1.5・因果)の描画(d444・d447)は −0.53。
devff の部分集合 0(30 本)を候補で設定ごとに描画 → d286_window を写し合成 CP・入力 SRC と並べる。
    uv run python d449_slowlocal.py --ckpt ../results/diag_ppgvoc43/snap/ema_002500.pt --out ../results/conv_p0/d449
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

SETTINGS = {"NC10": (1.0, False), "NC15": (1.5, False), "C10": (1.0, True), "C15": (1.5, True), "OFF": (0.0, False)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sub", default="../results/conv_p0/d444_p43/devff/sub")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    keys = sorted(f.name[:10] for f in Path(a.sub, "CP_0").glob("p*.wav"))
    for ref in ("CP_0", "SRC_0"):
        d = O / "grp" / ref
        d.mkdir(parents=True, exist_ok=True)
        for f in Path(a.sub, ref).glob("p*.wav"):
            if not (d / f.name).exists():
                (d / f.name).symlink_to(f.resolve())
    for nm, (sg, causal) in SETTINGS.items():
        st["slow_sigma"], st["slow_causal"] = sg, causal
        P.slow_from_ckpt(st)
        if sg == 0:
            P.SLOW["on"] = False
        d = O / "grp" / nm
        d.mkdir(parents=True, exist_ok=True)
        for k in keys:
            ti = int(k[7:10])
            c, lg = P.load_dump(P.H / "e2e" / "eval_devff", k)
            sf.write(d / f"{k}_{nm}.wav", P.render(g, c, lg, P.load_enroll(P.H / "devff" / "d85_work" / f"en_{ti:03d}"), dev), N.SR)
        print("rendered", nm, P.SLOW.get("on"), P.SLOW.get("sigma"), P.SLOW.get("causal"), flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "d286_window.py"), "--eq_dir", str(O / "grp"), "--out", str(O / "window.json")], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-800:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
