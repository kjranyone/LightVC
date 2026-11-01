"""D404(results/conv_p0/d404_prereg.yaml): 文脈の符号器のずれ d を 0 にして(包絡の行 = 表の引きだけ)描画し、表の帯域の形への追従(D399 の傾き)が上がるかを見る。
    uv run python d404_ctxzero.py --ckpt ../results/diag_ppgvoc27/product_002500_m0.pt --out ../results/conv_p0/d404
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d399_tablefollow import BANDS, prof


class Zero(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 24, x.shape[-1], device=x.device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    ctx = g.ctx
    sets = {"dev": ("eval_dev", "p00*.cond", None), "devff": ("eval_devff", "p0*.cond", "one")}
    rep: dict = {}
    dstat = []
    for s, (ed_, pat, one) in sets.items():
        ed = P.H / "e2e" / ed_
        refs = {int(f.stem[4:7]): prof(f) for f in sorted((P.H / s / "d37" / "_work").glob("ref_*.wav"))}
        seen = set()
        prof_by = {"B": {}, "CTX0": {}}
        for f in sorted(ed.glob(pat)):
            ti = int(f.stem.split("_t")[1][:3])
            if one and ti in seen:
                continue
            seen.add(ti)
            c, lg = P.load_dump(ed, f.stem)
            spk = P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}")
            for v in ("B", "CTX0"):
                g.ctx = ctx if v == "B" else Zero()
                y = P.render(g, c, lg, spk, dev)
                od = O / s / v
                od.mkdir(parents=True, exist_ok=True)
                sf.write(od / f"{f.stem}_{v}.wav", y, N.SR)
                prof_by[v].setdefault(ti, []).append(prof(od / f"{f.stem}_{v}.wav"))
            g.ctx = ctx
            with torch.no_grad():
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                st_ = P.spk_tensor(spk, dev)
                base = P.cond_ppg(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], *st_)
                d = ctx(torch.cat([p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], base[:, 1:25]], 1))
                dstat.append(float(d.std() / base[:, 1:25].std(-1).mean()))
        for v, by in prof_by.items():
            X = np.array([refs[t] for t in by]); Y = np.array([np.mean(by[t], 0) for t in by])
            out = {}
            for k, (lo, hi) in enumerate(BANDS):
                x, y = X[:, k] - X[:, k].mean(), Y[:, k] - Y[:, k].mean()
                out[f"{lo}-{hi}"] = (round(float((x * y).sum() / (x ** 2).sum()), 2), round(float(np.corrcoef(x, y)[0, 1]), 2), round(float((Y[:, k] - X[:, k]).mean()), 1))
            rep[f"{s}_{v}"] = out
            print(s, v, "(slope, r, bias)", json.dumps(out), flush=True)
    rep["d_std_over_rowstd"] = round(float(np.mean(dstat)), 3)
    print("d の大きさ(包絡の行の時間の標準偏差に対する比)", rep["d_std_over_rowstd"], flush=True)
    (O / "summary.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
