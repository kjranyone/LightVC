"""D340: D339 の遅い音色の揺らぎ(乱数・固定の統計)を男声 → 女声(PV・dev 60)と女声 → 女声(FF・devff 30)の変換に足して v1⊥ を測る(学習なし・ppgvoc18)。
--sets dev2 dev3 で確認の組(d296 と同じ組の作り方: <set>/devset.json・e2e/eval_<set>・<set>/d85_work/en_TTT)。
    uv run python d340_pvnoise.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d340
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
from d339_slownoise import fit_stats, make_noise


def items(s_: str) -> list:
    if s_ == "devff":
        return [(nm, c, lg, spk) for nm, c, lg, spk in P.ff_items(["devff"])]
    ed = P.H / "e2e" / f"eval_{s_}"
    out = []
    for r in json.loads((P.H / s_ / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, lg = P.load_dump(ed, k)
        out.append((f"{k}_CV", c, lg, P.load_enroll(P.H / s_ / "d85_work" / f"en_{r['ti']:03d}")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev", "devff"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 1.5, 2.0])
    ap.add_argument("--seed", type=int, default=340)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    B, amp, tgt, info = fit_stats(8, {h[0] for h in P.e2e_held()})
    print(json.dumps(info), flush=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"], P.CTX["ch"], P.CTX["adapt"] = bool(st.get("ctx", False)), int(st.get("ctx_ch", 192)), bool(st.get("adapt", False))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    for s_ in a.sets:
        its = items(s_)
        groups = []
        with torch.no_grad():
            for sg in a.sigmas:
                od = O / s_ / f"N{sg}"
                od.mkdir(parents=True, exist_ok=True)
                for i, (nm, c, lg, spk) in enumerate(its):
                    cond, exc, s2 = P.cond_exc(g, c, lg, spk, dev)
                    if sg > 0:
                        nz = torch.from_numpy(make_noise(B, amp, tgt, cond.shape[-1], a.seed + i)).to(dev)[None]
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + sg * nz, cond[:, 25:]], 1)
                    y = g(cond, exc, s2)[0].cpu().numpy()[N.DELAY:]
                    sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
                groups.append(f"N{sg}={od}/*.wav")
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / f"{s_}_v1perp.json")], capture_output=True, text=True)
        print(s_, len(its), r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
