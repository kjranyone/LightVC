"""D310: 内容の一致の損失(C1)が v1⊥ を押し戻す仕組みの仮説 = 出力の C1 の事後が本物より鋭い(はっきりしすぎた発音)。C1 の事後のエントロピーと最大確率(有声で音のあるフレーム)を、
本物の女声(REAL_POOL の真ん中のファイル 30 人)・男声 → 女声の各腕(dev・目標ごとに 1 本)・製品で比べる。
    uv run python d310_c1sharp.py --out ../results/conv_p0/d310.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def one_per_target(pat: str) -> list:
    seen, out = set(), []
    for f in sorted(glob.glob(pat)):
        t = Path(f).name.split("_t")[1][:3]
        if t not in seen:
            seen.add(t)
            out.append(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import train_c1 as T1
    from d6_unit_cov import load48
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mf = CC.MelFront().to(dev)
    real = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((ROOT / "female-dataset" / r["id"]).glob("*.wav"))
            real.append(str(ws[len(ws) // 2]))
        if len(real) == 30:
            break
    R_ = ROOT / "results"
    G = {"REAL_FEMALE": real,
         "PPG9": one_per_target(str(R_ / "diag_ppgvoc9/ev_002500/p00*_PV.wav")),
         "PPG11": one_per_target(str(R_ / "diag_ppgvoc11/ev_002500/p00*_PV.wav")),
         "PPG13": one_per_target(str(R_ / "diag_ppgvoc13/ev_002500/p00*_PV.wav")),
         "PROD": one_per_target(str(H / "dev/d114/p00*_G03S10ZH.wav")),
         "SRC_MALE": one_per_target(str(H / "dev/d56/p00*_SRC.wav"))}
    rep: dict = {}
    for g, files in G.items():
        ents, pmx = [], []
        for f in files:
            x = load48(f)[: 8 * 48000]
            with torch.no_grad():
                lg = c1(mf(torch.from_numpy(T1.prime(x))[None].to(dev)))[0]
                n = len(x) // 240
                p = lg[:, -n:].softmax(0)
                xm = torch.from_numpy(x[: n * 240].reshape(n, 240)).to(dev)
                rms = 20 * torch.log10(xm.pow(2).mean(1).sqrt() + 1e-9)
                m = rms > rms.max() - 35
                e = -(p * torch.log(p.clamp(min=1e-9))).sum(0)[m]
                ents.append(float(e.mean())); pmx.append(float(p.max(0).values[m].mean()))
        rep[g] = {"n": len(files), "entropy_mean": round(float(np.mean(ents)), 3), "entropy_sd": round(float(np.std(ents)), 3), "pmax_mean": round(float(np.mean(pmx)), 3)}
        print(g, rep[g], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
