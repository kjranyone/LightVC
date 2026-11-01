"""D535(D534 の Python 版・学習なし): F1(f0est3・Rust とフレーム一致)の有声のロジットと音高の確信度を一度だけ求め、因果な有声の判定の設定を多数振って
教師の有声(f0hi.teacher_f0・D534 のキャッシュ)との 偽の有声(F1 が有声・教師が無声)と取りこぼし(教師が有声・F1 が無声)の率・切り替わりの回数を出す。
設定 = 一つの閾値 θ(確率)・ヒステリシス(入る on・抜ける off)・確信度の下限 c。devff5 の元の 8 s のフレーム。出力 = results/conv_p0/d535.json。
    uv run python d535_f1_voicing_roc.py
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


def hyst(lg: np.ndarray, on: float, off: float) -> np.ndarray:
    a_on, a_off = np.log(on / (1 - on)), np.log(off / (1 - off))
    v = np.zeros(len(lg), bool)
    s = False
    for i, x in enumerate(lg):
        s = (x > a_off) if s else (x > a_on)
        v[i] = s
    return v


def main() -> int:
    dev = "cuda"
    st = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    front = FE.Front().to(dev)
    net = FE.F0Est(ch=st["cfg"]["ch"], dils=tuple(st["cfg"]["dils"])).to(dev)
    net.load_state_dict(st["net"])
    net.eval()
    S = P.H / "devff5"
    tv = dict(np.load(ROOT / "results/conv_p0/d534/teacher_v_devff5.npz"))
    L, C, TV = [], [], []
    seen = set()
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if k not in tv or r["pi"] in seen:
            continue
        seen.add(r["pi"])
        x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
        n = len(x) // N.HOP
        xx = torch.from_numpy(np.concatenate([np.zeros(FE.CTX48, np.float32), x[:n * N.HOP]]))[None].to(dev)
        with torch.no_grad():
            v, p = net(front(xx, n))
        L.append(v[0].cpu().numpy()[2400:4000])
        C.append(p[0].softmax(0).amax(0).cpu().numpy()[2400:4000])
        TV.append(tv[k][2400:4000])
    cfgs = {f"T{int(t * 100):02d}": ("thr", t) for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
    cfgs.update({f"H{int(a * 10)}{int(b * 10)}": ("hyst", a, b) for a, b in ((0.6, 0.4), (0.7, 0.3), (0.7, 0.4), (0.7, 0.5), (0.8, 0.3), (0.6, 0.3), (0.5, 0.3), (0.8, 0.5))})
    cfgs.update({f"C{int(c * 100):02d}": ("conf", c) for c in (0.1, 0.2, 0.3)})
    rep = {}
    for nm, cf in cfgs.items():
        fv, ms, sw = [], [], []
        for lg, cn, t_ in zip(L, C, TV):
            if cf[0] == "thr":
                vv = lg > np.log(cf[1] / (1 - cf[1]))
            elif cf[0] == "hyst":
                vv = hyst(lg, cf[1], cf[2])
            else:
                vv = (lg > 0) & (cn >= cf[1])
            n = min(len(vv), len(t_))
            vv, t_ = vv[:n], t_[:n]
            fv.append((vv & ~t_).sum() / max(1, (~t_).sum()))
            ms.append((~vv & t_).sum() / max(1, t_.sum()))
            sw.append(np.abs(np.diff(vv.astype(int))).sum() / (n / 200))
        rep[nm] = {"false_voiced": round(float(np.mean(fv)), 4), "miss": round(float(np.mean(ms)), 4), "sum": round(float(np.mean(fv) + np.mean(ms)), 4), "switch_per_s": round(float(np.mean(sw)), 2)}
        print(nm, rep[nm], flush=True)
    rep["TEACHER_switch_per_s"] = round(float(np.mean([np.abs(np.diff(t_.astype(int))).sum() / 8 for t_ in TV])), 2)
    rep["n_operators"] = len(L)
    print(rep["TEACHER_switch_per_s"], len(L), flush=True)
    (ROOT / "results/conv_p0/d535.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
