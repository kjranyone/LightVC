"""D540(f0est4 の主判定): F1 の版ごとに、組の入力(<set>/d467/<k>.wav・前置き 12 s + 元の 8 s)の元の 8 s のフレームで、教師の有声(f0hi.teacher_f0・ずれ 0)に対する
偽の有声(F1 有声・教師 無声)・取りこぼし(教師 有声・F1 無声)・大きい音の取りこぼし(c0n − 前置きの有声の c0n の中央値 ≥ −1.0・c0n は BASE の書き出し)・切り替わり / 秒 を出す。
版の対(B − A)の (偽の有声 + 取りこぼし) の差を操作者でまとめたブートストラップの 95% 区間で。教師の有声は <out>/teacher_v_<set>.npz に保存。
    uv run python d540_f1_voicing_eval.py --sets devff23x devff5 --ckpts F3=../results/f0est3/last.pt F4=../results/f0est4/last.pt --pair F4 F3 --out ../results/conv_p0/d540
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f0est as FE
import f0hi as FH
import nvoc as N
import train_ppgvoc as P
from d6_unit_cov import load48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--pair", nargs=2, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    nets = {}
    for c in a.ckpts:
        nm, p = c.split("=")
        st = torch.load(p, map_location="cpu", weights_only=False)
        net = FE.F0Est(ch=st["cfg"]["ch"], dils=tuple(st["cfg"]["dils"])).to(dev)
        net.load_state_dict(st["net"])
        nets[nm] = net.eval()
    front = FE.Front().to(dev)
    rep = {}
    for s in a.sets:
        S = P.H / s
        ed = P.H / "e2e" / f"eval_{s}_pre"
        cache = O / f"teacher_v_{s}.npz"
        tv = dict(np.load(cache)) if cache.exists() else {}
        pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"] if (S / "d467" / f"p{r['pi']:04d}_t{r['ti']:03d}.wav").exists() and (ed / f"p{r['pi']:04d}_t{r['ti']:03d}.cond").exists()]
        rows = {nm: {} for nm in nets}
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
            n = len(x) // N.HOP
            if k not in tv:
                ft, _ = FH.teacher_f0(x.astype(np.float64), n=n)
                tv[k] = ft[:n] > 0
            c, _ = P.load_dump(ed, k)
            pre = c[:2400]
            ref = float(np.median(pre[pre[:, 26] > 0.5, 0])) if (pre[:, 26] > 0.5).sum() > 50 else float(np.median(c[c[:, 26] > 0.5, 0]))
            sl = slice(2400, min(4000, len(c), n))
            t_ = tv[k][sl]
            loud = (c[sl, 0] - ref) >= -1.0
            xx = torch.from_numpy(np.concatenate([np.zeros(FE.CTX48, np.float32), x[:n * N.HOP]]))[None].to(dev)
            with torch.no_grad():
                ft_ = front(xx, n)
                for nm, net in nets.items():
                    v, _ = net(ft_)
                    vv = (v[0] > 0).cpu().numpy()[sl]
                    rows[nm][k] = (float((vv & ~t_).sum() / max(1, (~t_).sum())), float((~vv & t_).sum() / max(1, t_.sum())),
                                   float((~vv & t_ & loud).sum() / max(1, (t_ & loud).sum())), float(np.abs(np.diff(vv.astype(int))).sum() / (len(vv) / 200)), r.get("spk", k))
        np.savez(cache, **tv)
        rs = {}
        for nm, d in rows.items():
            rs[nm] = {m: round(float(np.mean([v[i] for v in d.values()])), 4) for i, m in enumerate(("false_voiced", "miss", "miss_loud", "switch_per_s"))}
            print(s, nm, rs[nm], flush=True)
        rs["TEACHER_switch_per_s"] = round(float(np.mean([np.abs(np.diff(tv[k][2400:4000].astype(int))).sum() / 8 for k in tv])), 2)
        if a.pair and all(p_ in rows for p_ in a.pair):
            b, c_ = a.pair
            ks = sorted(set(rows[b]) & set(rows[c_]))
            spk = [rows[b][k][4] for k in ks]
            us = sorted(set(spk))
            idx = {u: [i for i, s_ in enumerate(spk) if s_ == u] for u in us}
            rng = np.random.default_rng(540)
            for j, m in ((None, "sum"), (0, "false_voiced"), (1, "miss"), (2, "miss_loud")):
                d = np.array([(rows[b][k][0] + rows[b][k][1] - rows[c_][k][0] - rows[c_][k][1]) if j is None else rows[b][k][j] - rows[c_][k][j] for k in ks])
                bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
                rs[f"{b}-{c_}_{m}"] = [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
                print(s, f"{b}-{c_}_{m}", rs[f"{b}-{c_}_{m}"], flush=True)
        rep[s] = rs
    (O / "d540.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
