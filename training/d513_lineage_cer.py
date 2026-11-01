"""D513(診断): 生成器に本人の実の包絡 + 実の周期性 + 事後を与えた描画(SELF_ALLP・D498)の明瞭さが、学習の系譜(ppgvoc1 → … → 43)のどこで写し合成から離れたか。
devff23x の定常状態の条件(eval_devff23x_pre)・文脈の符号器と遅い揺らぎを通さない(条件を直接)= 生成器そのものの明瞭さ。CER は d23_cer で別に測る。
    uv run python d513_lineage_cer.py --ckpts N1=../results/diag_ppgvoc1/last_ema.pt ... --out ../results/conv_p0/d513
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
import train_rvoc as TR

PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=126)
    a = ap.parse_args()
    dev = "cuda"
    S = P.H / "devff23x"
    ed = P.H / "e2e" / "eval_devff23x_pre"
    n0 = int(PRE * N.SR)
    pairs = json.loads((S / "devset.json").read_text())["pairs"][: a.n]
    for spec in a.ckpts:
        name, path = spec.split("=")
        st = torch.load(path, map_location="cpu", weights_only=False)
        P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192), adapt=bool(st.get("adapt", False)))
        P.PPG = st.get("ppg", "soft")
        g = P.init_gen(dev, st.get("spk", "table"))
        g.load_state_dict(st["ema"])
        g.eval()
        od = Path(a.out) / name
        od.mkdir(parents=True, exist_ok=True)
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, lg = P.load_dump(ed, k)
            with torch.no_grad():
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                cond = torch.cat([ct[:, :31], p], 1)
                f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                y = g(cond, exc, None)[0].cpu().numpy()[N.DELAY:][n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{name}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", name, len(pairs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
