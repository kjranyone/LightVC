"""D471(results/conv_p0/d471/prereg.yaml): 登録の定数 Δ で c0n を発声中だけずらす PP05 を、α の選択に使っていない実男声の組(devreal = 外郎売・devvctkm の男声)で一度だけ確かめる。
Δ = 目標の参照の有声の c0n の中央値 − 操作者の参照の有声の c0n の中央値(devreal: e2e/opref_devreal/op_uiro・devvctkm: 操作者の参照 d56/_work/ref_<spk>.wav を Rust で書き出す)。
版 = NONE / PP05 を候補(P43OP)で描画 → <out>/<set>/<版>/(対の評価)・<scratch>/<set>/d471/<版>/(d109)。
    uv run python d471_c0cal_real.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d471
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d470_c0cal import c0med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    od = P.H / "e2e" / "opref_devvctkm"
    od.mkdir(parents=True, exist_ok=True)
    vrows = [r for r in json.loads((P.H / "devvctkm" / "devset.json").read_text())["pairs"] if r["kind"].startswith("M")]
    jobs = []
    for spk in sorted({r["spk"] for r in vrows}):
        w = P.H / "devvctkm" / "d56" / "_work" / f"ref_{spk}.wav"
        if w.exists():
            jobs.append(([BIN, "artic", "--enroll", P.H / "devvctkm" / "d85_work" / "en_000"] + COM + ["-i", w], od / f"op_{spk}.cond", od / f"op_{spk}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(job, jobs))
    print("operator refs dumped", len(jobs), flush=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    rep = {}
    items = {"devreal": [(f"p{r['pi']:04d}_t{r['ti']:03d}", r["ti"], P.H / "e2e" / "opref_devreal" / "op_uiro.cond") for r in json.loads((P.H / "devreal" / "devset.json").read_text())["pairs"]],
             "devvctkm": [(f"p{r['pi']:04d}_t{r['ti']:03d}", r["ti"], od / f"op_{r['spk']}.cond") for r in vrows]}
    for s, its in items.items():
        ed = P.H / "e2e" / f"eval_{s}"
        deltas = []
        for k, ti, opc in its:
            if not (ed / f"{k}.cond").exists() or not opc.exists():
                continue
            c, lg = P.load_dump(ed, k)
            dl = c0med(ed / f"ref_{ti:03d}.cond") - c0med(opc)
            deltas.append(dl)
            w = np.clip(c[:, 0] - 1.0, 0.0, 1.0)
            for v, al in (("NONE", 0.0), ("PP05", 0.5)):
                cc = c.copy()
                cc[:, 0] = c[:, 0] + al * dl * w
                y = P.render(g, cc, lg, P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}"), dev, T_op=P.op_tab(P.H / s, k) if P.OPSLOW["on"] else None)
                for d in (O / s / v, P.H / s / "d471" / v):
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(d / f"{k}_{v}.wav", y, N.SR)
        rep[s] = {"n": len(deltas), "delta_median": round(float(np.median(deltas)), 3), "delta_p10_p90": [round(float(np.percentile(deltas, q)), 3) for q in (10, 90)]}
        print(s, rep[s], flush=True)
    (O / "deltas.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
