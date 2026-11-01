"""D470(results/conv_p0/d470/prereg.yaml): 男声の操作者の c0n を発声中のフレームだけ登録の定数でずらす推論の工夫を、選択に使っていない dev2(選択)・dev3(確認)で測る。
c0n' = c0n + α·Δ·clip(c0n − 1, 0, 1)・Δ = 目標の参照の有声の c0n の中央値 − 操作者の参照の有声の c0n の中央値(d296_confirm の c0med と同じ)。版 = NONE / PP05 / PP10 / FIX04(一律 +0.4)。
描画 → <out>/<set>/<版>/pNNNN_tTTT_<版>.wav(対の評価)・<scratch>/<set>/d470/<版>/(d109)。評価は d470_run.sh。
    uv run python d470_c0cal.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --sets dev2 dev3 --out ../results/conv_p0/d470
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

VARS = {"NONE": None, "PP05": 0.5, "PP10": 1.0, "FIX04": "fix"}


def c0med(p: Path) -> float:
    c = np.fromfile(p, "<f4").reshape(-1, 31)
    v = c[:, 26] > 0.5
    return float(np.median(c[v, 0])) if v.any() else float(np.median(c[:, 0]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
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
    rep: dict = {}
    for s in a.sets:
        ed = P.H / "e2e" / f"eval_{s}"
        deltas = []
        for f in sorted(ed.glob("p00*.cond")):
            k = f.stem
            ti, pi = int(k.split("_t")[1][:3]), int(k[1:5])
            c, lg = P.load_dump(ed, k)
            dl = c0med(ed / f"ref_{ti:03d}.cond") - c0med(P.H / "e2e" / f"opref_{s}" / f"op_{pi:04d}.cond")
            deltas.append(dl)
            w = np.clip(c[:, 0] - 1.0, 0.0, 1.0)
            for v, al in VARS.items():
                cc = c.copy()
                if al == "fix":
                    cc[:, 0] = c[:, 0] + 0.4 * w
                elif al is not None:
                    cc[:, 0] = c[:, 0] + al * dl * w
                y = P.render(g, cc, lg, P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}"), dev, T_op=P.op_tab(P.H / s, k) if P.OPSLOW["on"] else None)
                for d in (O / s / v, P.H / s / "d470" / v):
                    d.mkdir(parents=True, exist_ok=True)
                    sf.write(d / f"{k}_{v}.wav", y, N.SR)
        rep[s] = {"n": len(deltas), "delta_median": round(float(np.median(deltas)), 3), "delta_p10_p90": [round(float(np.percentile(deltas, q)), 3) for q in (10, 90)]}
        print(s, rep[s], flush=True)
    (O / "deltas.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
