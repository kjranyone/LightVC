"""diag_ppgvoc31 の起動前の検査(煙テストの step 1 で入れ替えの log-mel が自己の 1.55 倍・利得の差 4.4 dB の 1 本 → 原因の切り分け)。
同じ区間で 事後 = F 自身(自然)/ 往復 / 源の男声 / 源の女声 を学習と同じ前向き(2.8 s・WARM を除く・RMS をそろえない)で描画し、log-mel L1 と RMS の差(出力 − 目標 dB)を並べる。
    uv run python d396b_poolcheck.py --pool <dev の親>/d396 --ckpt ../results/diag_ppgvoc27/snap/ema_002500.pt --n 48
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_ppgvoc as P
import train_rvoc as TR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = "cuda"
    TR.SEG = 33600 * 4
    TR.T = TR.SEG // 240
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=False)
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.train()
    mels = [P.build_mel(nf, hm, nm).to(dev) for nf, hm, nm in P.MEL_SPECS]
    sp = P.SwapPool(a.pool, TR.T, TR.SEG, 0.0, 0.5, 7)
    rows = []
    torch.manual_seed(0)
    for i in range(a.n):
        f, _, _, t0 = sp.pick()
        xm, xf = sp.rng.choice(sp.males), sp.rng.choice([q for q in sp.tf if q != f])
        c = torch.from_numpy(sp.c[f][t0:t0 + sp.T].T.copy()).float().to(dev)[None]
        s0 = t0 * 240 - sp.off
        tgt = torch.from_numpy(sp.wav[f][s0:s0 + sp.seg].copy()).to(dev)[None]
        T_, P_ = sp.tab[f]
        spk = (torch.from_numpy(T_).float().to(dev)[None], torch.from_numpy(P_).float().to(dev)[None])
        f0 = torch.where(c[:, 26] > 0.5, 200.0 * torch.exp(c[:, 25]), torch.zeros_like(c[:, 25]))
        exc = TR.excitation(f0, sp.seg, torch.Generator().manual_seed(i))
        z = P.slow_noise(1, sp.T, dev, torch.Generator(device="cpu").manual_seed(i))
        row = {"f": f}
        posts = {"NAT": P.softmax_np(sp.lg[f][t0:t0 + sp.T].astype(np.float32)), "RT": sp.one(f, "rt", xm, t0), "M": sp.one(f, "m", xm, t0), "F": sp.one(f, "f", xf, t0)}
        for k, q in posts.items():
            p = torch.from_numpy(q.T.copy()).float().to(dev)[None]
            with torch.no_grad():
                y = g(g.make(p, c[:, 0:1], c[:, 25:26], c[:, 26:27], spk, z), exc, spk)[:, TR.WARM:]
                tl = tgt[:, TR.WARM:]
                row[f"lm_{k}"] = float(P.logmel_l1(y[:, None], tl[:, None], mels))
                row[f"db_{k}"] = float(20 * torch.log10(y.pow(2).mean().sqrt() / tl.pow(2).mean().sqrt()))
                row[f"lmg_{k}"] = float(P.logmel_l1(y[:, None], (tl * y.pow(2).mean().sqrt() / tl.pow(2).mean().sqrt())[:, None], mels))
        rows.append(row)
    rep = {k: {"mean": round(float(np.mean([r[k] for r in rows])), 4), "sd": round(float(np.std([r[k] for r in rows])), 4)} for k in rows[0] if k != "f"}
    for k in ("RT", "M", "F"):
        d = np.array([r[f"lm_{k}"] / r["lm_NAT"] for r in rows])
        rep[f"ratio_{k}_NAT"] = round(float(d.mean()), 4)
        d = np.array([r[f"lmg_{k}"] / r["lmg_NAT"] for r in rows])
        rep[f"ratio_gain_{k}_NAT"] = round(float(d.mean()), 4)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    worst = sorted(rows, key=lambda r: abs(r["db_NAT"]))[-5:]
    print("利得の差の大きい区間(NAT)", [(r["f"], round(r["db_NAT"], 1)) for r in worst], flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": rep, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
