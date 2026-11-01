"""D433(results/conv_p0/d433_prereg.yaml): D2 と 40 の 5,000 step の重みの線形補間の診断。
    uv run python d433_soup.py --a ../results/diag_ppgvoc41/product_002500_m0.pt --b ../results/diag_ppgvoc40/product_005000_m0.pt --out ../results/conv_p0/d433
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    a = ap.parse_args()
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    A = torch.load(a.a, map_location="cpu", weights_only=False)
    B = torch.load(a.b, map_location="cpu", weights_only=False)
    assert set(A["ema"]) == set(B["ema"]), "鍵が違う"
    specs = []
    for al in a.alphas:
        st = dict(A)
        st["ema"] = {k: ((1 - al) * A["ema"][k].float() + al * B["ema"][k].float()).to(A["ema"][k].dtype) if A["ema"][k].is_floating_point() else A["ema"][k] for k in A["ema"]}
        p = O / f"soup_{int(al * 100):03d}.pt"
        torch.save(st, p)
        specs.append(f"S{int(al * 100):03d}={p}")
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "d296_confirm.py"), "--ckpts", *specs, "--sets", "dev", "--out", str(O)], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-800:] if r.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
