"""D354: 遅い音色の揺らぎ z を乱数の代わりに操作者自身の遅い揺らぎ(操作者の包絡 − 操作者の登録の表 Σp T_op / 10 の 0.5 s 平均 − 区間の平均 → 主成分へ射影・train_ppgvoc.slow_real と同じ定義)にすると
生成の床(SR / FF / FF2)と男声 → 女声(dev)は乱数より良いか(韻律と結びついた揺らぎを目標へ移す)。非因果の診断(0.5 s の中心の平均と区間の平均)= 効けば因果な版を設計する。
操作者の表: SR = 本人の登録・FF / FF2 / dev = <組>/d85_work/op_PPPP。
    uv run python d354_opdrift.py --ckpt ../results/diag_ppgvoc24/product_s15_007500.pt --out ../results/conv_p0/d354
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


def items(s_: str) -> list:
    """[(名前, cond, logit, 目標の登録, 操作者の表 [24, K])]"""
    if s_ == "sr":
        return [(nm, c, lg, spk, spk[0]) for nm, c, lg, spk in P.eval_items(P.e2e_held(), "table", "cuda")[0]]
    ed = P.H / "e2e" / f"eval_{s_}"
    out, seen = [], set()
    rows = json.loads((P.H / s_ / "devset.json").read_text())["pairs"] if s_ == "dev" else None
    keys = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in rows] if rows else [f.stem for f in sorted(ed.glob("p0*.cond"))]
    for k in keys:
        pi, ti = int(k[1:5]), int(k.split("_t")[1][:3])
        if s_ != "dev":
            if ti in seen:
                continue
            seen.add(ti)
        c, lg = P.load_dump(ed, k)
        out.append((f"{k}_X", c, lg, P.load_enroll(P.H / s_ / "d85_work" / f"en_{ti:03d}"), P.load_enroll(P.H / s_ / "d85_work" / f"op_{pi:04d}")[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["sr", "devff", "devff2", "dev"])
    ap.add_argument("--variants", nargs="+", default=["NOISE", "OPZ", "OPZ15"], help="+ OPMID / OPMID05 = 遅い乱数(ckpt の σ)に加えて操作者の中間の帯(0.15 s 平均 − 0.5 s 平均)の包絡の残差 × 1 / 0.5(D365)")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    assert P.SLOW["on"], "遅い揺らぎを学習した ckpt が要る"
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    for s_ in a.sets:
        its = items(s_)
        groups = []
        with torch.no_grad():
            for v in a.variants:
                od = O / s_ / v
                od.mkdir(parents=True, exist_ok=True)
                for nm, c, lg, spk, Top in its:
                    p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                    ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                    s2 = P.spk_tensor(spk, dev)
                    z = None
                    if v.startswith("OPZ"):
                        base_op = torch.einsum("dk,bkt->bdt", torch.from_numpy(Top).float().to(dev), p) / 10
                        z = P.slow_real(ct[:, 1:25], base_op) * (1.5 if v.endswith("15") else 1.0)
                    lf = ct[:, 25:26]
                    cond = g.make(p, ct[:, 0:1], lf, ct[:, 26:27], s2, z)
                    if v.startswith("OPMID"):
                        base_op = torch.einsum("dk,bkt->bdt", torch.from_numpy(Top).float().to(dev), p) / 10
                        r_ = ct[:, 1:25] - base_op
                        mid = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(r_, (15, 14), mode="replicate"), 30, 1) - torch.nn.functional.avg_pool1d(torch.nn.functional.pad(r_, (50, 49), mode="replicate"), 100, 1)
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + (0.5 if v.endswith("05") else 1.0) * mid, cond[:, 25:]], 1)
                    f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(lf[:, 0]), torch.zeros_like(lf[:, 0]))
                    exc = P.TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                    y = g(cond, exc, s2)[0].cpu().numpy()[N.DELAY:]
                    sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
                groups.append(f"{v}={od}/*.wav")
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / f"{s_}_v1perp.json")], capture_output=True, text=True)
        print(s_, len(its), r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
