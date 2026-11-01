"""D504: 男声の交差の標本の置き場(MalePool の書き出し・<dir>/wav/<key>.wav)に C2 の特徴を書く(<dir>/c2/<key>.npy・[d, T]・float16)。
C2 は因果 = ファイル全体を開始状態の規約(train_c1.prime)で流した値が製品の流し込みと同じ。フレーム数は Rust の書き出しと一致を確かめる。
    uv run python d504_c2_male.py --c2 ../results/c2_1/last.pt --dirs <H>/xmale <H>/condgan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    dev = "cuda"
    P.cvin_setup(a.c2, dev)
    for d in a.dirs:
        d = Path(d)
        (d / "c2").mkdir(exist_ok=True)
        n = 0
        for m in json.loads((d / "meta.json").read_text())["male"]:
            if m["key"].startswith("m0") or m["spk"].startswith("m0"):
                continue
            if not (d / "male" / f"{m['key']}.cond").exists():
                continue
            cc, _ = P.load_dump(d / "male", m["key"])
            x = load48(d / "wav" / f"{m['key']}.wav").astype(np.float32)
            with torch.no_grad():
                z = P.cvin_frames(torch.from_numpy(TC.prime(x))[None].to(dev), len(x) // 240)[0].cpu().numpy()
            assert z.shape[1] == len(cc), (m["key"], z.shape, len(cc))
            np.save(d / "c2" / f"{m['key']}.npy", z.astype(np.float16))
            n += 1
        print(d, "written", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
