"""D362: 製品の向き(目標の声域が操作者より高い = 声域を上げる)の女声 → 女声だけの組(FF_up)で、候補が到達の線(AUC ≤ 0.95)の内側か。
devff(目標ごとに操作者 3 人)と devff2(1 人)の既存の書き出しから、目標ごとに声域の差(目標 − 操作者の登録の mu)が +min_st 以上で最大の操作者を 1 人選ぶ(目標は重複なし・devff2 の目標番号は +100 で名前の衝突を避ける)。
群ごとの C2ST(eval_v1perp・群の大きさは写し合成 CP 30 本とほぼ同じ)。
    uv run python d362_ffup.py --ckpts CAND=../results/diag_ppgvoc24/product_s15_007500.pt PPG18=../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d362
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def items(min_st: float) -> list:
    out = []
    for s_, off in (("devff", 0), ("devff2", 100)):
        by: dict = {}
        for r in json.loads((P.H / s_ / "devset.json").read_text())["pairs"]:
            mo = json.loads((P.H / s_ / "d85_work" / f"op_{r['pi']:04d}" / "enroll.json").read_text())["mu"]
            mt = json.loads((P.H / s_ / "d85_work" / f"en_{r['ti']:03d}" / "enroll.json").read_text())["mu"]
            by.setdefault(r["ti"], []).append(((mt - mo) * 12 / np.log(2), r["pi"]))
        for ti, v in sorted(by.items()):
            sh, pi = max(v)
            if sh < min_st:
                continue
            c, lg = P.load_dump(P.H / "e2e" / f"eval_{s_}", f"p{pi:04d}_t{ti:03d}")
            out.append((f"p{pi:04d}_t{ti + off:03d}_FU", c, lg, P.load_enroll(P.H / s_ / "d85_work" / f"en_{ti:03d}"), sh))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_st", type=float, default=1.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    its = items(a.min_st)
    print("FF_up items", len(its), "shift median %.1f st" % np.median([x[4] for x in its]), flush=True)
    groups = []
    for spec in a.ckpts:
        name, ck = spec.split("=", 1)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        P.PPG = st.get("ppg", "soft")
        P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
        P.slow_from_ckpt(st)
        g = P.init_gen(dev, st.get("spk", "table"))
        g.load_state_dict(st["ema"])
        g.eval()
        for sd in (a.seeds if P.SLOW["on"] else [0]):
            P.SLOW["seed"] = sd
            tag = f"{name}_r{sd}" if P.SLOW["on"] else name
            od = O / tag
            od.mkdir(parents=True, exist_ok=True)
            for nm, c, lg, spk, _ in its:
                sf.write(od / f"{nm}.wav", P.render(g, c, lg, spk, dev), N.SR)
            groups.append(f"{tag}={od}/*.wav")
            print("rendered", tag, flush=True)
    groups.append(f"CP={P.ROOT / 'results/diag_e2e1/eval'}/h*_CP.wav")
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
