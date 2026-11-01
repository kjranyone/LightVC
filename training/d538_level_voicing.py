"""D538(D535 の続き・学習なし): 音量で閾値を変える F1 の有声の判定。偽の有声(息の端のピッチ)は小さい音(中央値 −31 dB)、V70 で増えた取りこぼしは大きい音にもかかる。
規則 = 有声 ⇔ p > 0.5(c0n_rel ≥ τ)/ p > θq(c0n_rel < τ)。c0n_rel = 書き出しの c0n − 操作者の参照(d56/_work/op_PPPP.wav の前置き 12 s 分の書き出し)の有声の c0n の中央値(製品では登録の定数)。
教師の有声に対し 偽の有声・取りこぼし(大きい音 = 教師が有声かつ c0n_rel ≥ −1.0・小さい音)を出す。devff5(選択の組)。出力 = results/conv_p0/d538.json。
    uv run python d538_level_voicing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f0est as FE
import nvoc as N
import train_ppgvoc as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    dev = "cuda"
    st = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    front = FE.Front().to(dev)
    net = FE.F0Est(ch=st["cfg"]["ch"], dils=tuple(st["cfg"]["dils"])).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    S = P.H / "devff5"
    ed = P.H / "e2e" / "eval_devff5_pre"
    tv = dict(np.load(ROOT / "results/conv_p0/d534/teacher_v_devff5.npz"))
    D = []
    seen = set()
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if k not in tv or r["pi"] in seen or not (ed / f"{k}.cond").exists():
            continue
        seen.add(r["pi"])
        c, _ = P.load_dump(ed, k)
        x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
        n = len(x) // N.HOP
        xx = torch.from_numpy(np.concatenate([np.zeros(FE.CTX48, np.float32), x[:n * N.HOP]]))[None].to(dev)
        with torch.no_grad():
            v, _ = net(front(xx, n))
        pv = torch.sigmoid(v[0]).cpu().numpy()
        pre = c[:2400]
        ref = float(np.median(pre[pre[:, 26] > 0.5, 0])) if (pre[:, 26] > 0.5).sum() > 50 else float(np.median(c[c[:, 26] > 0.5, 0]))
        sl = slice(2400, 4000)
        D.append((pv[sl], c[sl, 0] - ref, tv[k][sl]))
    rules = {"BASE": (0.5, -99.0, 0.5), "V70": (0.7, -99.0, 0.7)}
    for tau in (-1.0, -1.5, -2.0, -2.5, -3.0):
        for tq in (0.7, 0.8, 0.9):
            rules[f"L{-tau:.1f}_{int(tq * 100)}"] = (0.5, tau, tq)
    rep = {}
    for nm, (th, tau, tq) in rules.items():
        fv = ml = mq = nu = nvl = nvq = 0
        for pv, lv, t_ in D:
            vv = np.where(lv >= tau, pv > th, pv > tq)
            loud = lv >= -1.0
            fv += (vv & ~t_).sum(); nu += (~t_).sum()
            ml += (~vv & t_ & loud).sum(); nvl += (t_ & loud).sum()
            mq += (~vv & t_ & ~loud).sum(); nvq += (t_ & ~loud).sum()
        rep[nm] = {"false_voiced": round(fv / nu, 4), "miss_loud": round(ml / nvl, 4), "miss_quiet": round(mq / nvq, 4)}
        print(nm, rep[nm], flush=True)
    rep["n_ops"] = len(D)
    (ROOT / "results/conv_p0/d538.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
