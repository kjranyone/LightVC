"""D485(results/conv_p0/d485/prereg.yaml): 操作者の遅い揺らぎ(OP15)に、操作者の中間の帯(過去 0.15 s の平均 − 過去 0.5 s の平均・基底へ射影・因果)を g 倍で足す版を定常状態で。
D467 の前置きの書き出し(eval_<組>_pre)を使い、元の 8 s を <out>/<組>/<版>/ に。版 = OP15・OPM05・OPM10。
    uv run python d485_opmid.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d485
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

PRE = 12.0


def mid_causal(r: torch.Tensor) -> torch.Tensor:
    a = F.avg_pool1d(F.pad(r, (29, 0), mode="replicate"), 30, 1)
    b = F.avg_pool1d(F.pad(r, (99, 0), mode="replicate"), 100, 1)
    B = P.SLOW["B"].to(r.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, a - b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["devff2", "dev2", "dev3", "devreal"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    assert P.OPSLOW["on"]
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    n0 = int(PRE * N.SR)
    for s in a.sets:
        S = P.H / s
        ed = P.H / "e2e" / f"eval_{s}_pre"
        pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"] if r.get("kind", "FF") != "AA"]
        n = 0
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
            ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
            stt = P.spk_tensor(P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev)
            rr = ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(P.op_tab(S, k)).float().to(dev), p) / 10
            zs = P.OPSLOW["gain"] * P.opslow_causal(rr, P.OPSLOW["tau"])
            zm = mid_causal(rr)
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            exc = P.TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            with torch.no_grad():
                for v, gm in (("OP15", 0.0), ("OPM05", 0.5), ("OPM10", 1.0)):
                    y = g(g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, zs + gm * zm), exc, stt)[0].cpu().numpy()[N.DELAY:]
                    y = y[n0:n0 + 8 * N.SR]
                    (O / s / v).mkdir(parents=True, exist_ok=True)
                    sf.write(O / s / v / f"{k}_{v}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            n += 1
        print("rendered", s, n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
