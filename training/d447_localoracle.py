"""D447: 女声 → 女声の候補と写し合成 CP の差(D446: 0.5 s の局所で +0.69)を、どの条件が運ぶか(本物の値への差し替え = oracle・診断)。
devff の部分集合 0(目標ごとに 1 本・30 本)を候補で、条件の行を入力そのものの Rust の条件に差し替えて描画:
ORIG / PER(周期性 4 行 = 入力の実の値・表の定数の代わり)/ ENV(包絡 24 行 = 入力の実の包絡・表の引き + 文脈 + 揺らぎの代わり)/ BOTH。
→ d286_window(0.5 / 2 / 8 s の窓の v1)を CP・SRC の部分集合 0 と並べる。
    uv run python d447_localoracle.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d447
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

VARS = ("ORIG", "PER", "ENV", "BOTH")


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
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    keys = sorted(f.name[:10] for f in Path(a.sub, "CP_0").glob("p*.wav"))
    for v in VARS:
        (O / "grp" / v).mkdir(parents=True, exist_ok=True)
    for ref in ("CP_0", "SRC_0"):
        d = O / "grp" / ref
        d.mkdir(parents=True, exist_ok=True)
        for f in Path(a.sub, ref).glob("p*.wav"):
            if not (d / f.name).exists():
                (d / f.name).symlink_to(f.resolve())
    with torch.no_grad():
        for k in keys:
            ti = int(k[7:10])
            c, lg = P.load_dump(P.H / "e2e" / "eval_devff", k)
            spk = P.load_enroll(P.H / "devff" / "d85_work" / f"en_{ti:03d}")
            for v in VARS:
                cond, exc, stt = P.cond_exc(g, c, lg, spk, dev)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                if v in ("ENV", "BOTH"):
                    cond = torch.cat([cond[:, :1], ct[:, 1:25], cond[:, 25:]], 1)
                if v in ("PER", "BOTH"):
                    cond = torch.cat([cond[:, :27], ct[:, 27:31], cond[:, 31:]], 1)
                y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
                sf.write(O / "grp" / v / f"{k}_{v}.wav", (y / (abs(y).max() + 1e-9) * 0.9), N.SR)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "d286_window.py"), "--eq_dir", str(O / "grp"), "--out", str(O / "window.json")], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-800:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
