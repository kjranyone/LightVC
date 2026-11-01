"""D523: 男声の交差の標本の置き場(MalePool・<dir>/wav/<key>.wav)に F4(因果な周期性の推定・results/f4_1・C1 と同じ因果 mel)を書く(<dir>/f4/<key>.npy・[4, T]・float16)。
ファイル全体を開始状態の規約(train_c1.prime)で流した値 = 製品の流し込みと同じ。フレーム数は Rust の書き出しと一致を確かめる(diag_ppgvoc53 の周期性の動き)。
    uv run python d523_f4_male.py --dirs <H>/xmale <H>/condgan
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import lvl as LV
import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    a = ap.parse_args()
    dev = "cuda"
    s4 = torch.load(ROOT / "results/f4_1/last.pt", map_location="cpu", weights_only=False)
    f4 = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev)
    f4.load_state_dict(s4["net"])
    f4.eval()
    mf = CC.MelFront().to(dev)
    for d in a.dirs:
        d = Path(d)
        (d / "f4").mkdir(exist_ok=True)
        n = 0
        for m in json.loads((d / "meta.json").read_text())["male"]:
            if m["key"].startswith("m0") or m["spk"].startswith("m0"):
                continue
            if not (d / "male" / f"{m['key']}.cond").exists():
                continue
            cc, _ = P.load_dump(d / "male", m["key"])
            x = load48(d / "wav" / f"{m['key']}.wav").astype(np.float32)
            T = len(x) // 240
            with torch.no_grad():
                z = f4(mf(torch.from_numpy(TC.prime(x))[None].to(dev)))[0, :, -T:].cpu().numpy()
            assert z.shape[1] == len(cc), (m["key"], z.shape, len(cc))
            np.save(d / "f4" / f"{m['key']}.npy", z.astype(np.float16))
            n += 1
        print(d, "written", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
