"""D421: オーナーの耳の比較用の組(B / D(ppgvoc36)/ ppgvoc34 / 製品 + 入力 + 目標の参照)。確認の組 dev2 の 6 本(目標が別々)・全部を同じ RMS(−23 dBFS)にそろえる(音量の手がかりを消す・blind-test-level-leak)。
    uv run python d421_listen.py --out ../results/conv_p0/d421_listen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
R = ROOT / "results/conv_p0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="dev2")
    a = ap.parse_args()
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    s = a.set
    pairs = sorted(f.name[:10] for f in (H / s / "d56").glob("p00*_SRC.wav"))
    pick = pairs[::10][:6]
    src = {
        "SRC": lambda k: H / s / "d56" / f"{k}_SRC.wav",
        "REF": lambda k: H / s / "d37" / "_work" / f"ref_{k[7:10]}.wav",
        "B": lambda k: next((R / "d374" / s / "PPG27").glob(f"{k}*.wav")),
        "D": lambda k: next((R / "d416_002500" / s / "PPG36").glob(f"{k}*.wav")),
        "D2": lambda k: next((R / "d428_002500" / s / "PPG41").glob(f"{k}*.wav")),
        "P34": lambda k: next((R / "d410_005000" / s / "PPG34").glob(f"{k}*.wav")),
        "P38": lambda k: next((R / "d426_005000" / s / "PPG38").glob(f"{k}*.wav")),
        "PROD": lambda k: H / s / "d114" / f"{k}_G03S10ZH.wav",
    }
    rows = []
    for i, k in enumerate(pick, 1):
        for name, fn in src.items():
            x, sr = sf.read(str(fn(k)), dtype="float32", always_2d=True)
            x = x.mean(1)
            if name == "REF":
                x = x[: 8 * sr]
            g = 10 ** (-23 / 20) / (np.sqrt((x ** 2).mean()) + 1e-9)
            y = np.clip(x * g, -1.0, 1.0)
            sf.write(O / f"{i:02d}_{name}.wav", y.astype(np.float32), sr)
        rows.append({"no": i, "pair": k, "target": k[7:10]})
    (O / "items.json").write_text(json.dumps(rows, indent=1))
    print("wrote", len(rows), "items x", len(src), "->", O, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
