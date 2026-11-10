"""p2_knn_convert の物差しの側の参考の鍵(diag): A0F の目標 24 声 × j 0〜2 の SPARC の EMA を書き出す(SPARC の依存は ../sparc_env)。
出力 = data/snv_e1_feat/sparc/a0/{声}_{j}.npz。
    cd .. && ../sparc_env/bin/python training/snv_diag/p2_sparc_a0.py
"""
import json
from pathlib import Path

import numpy as np
from sparc import load_model

ROOT = Path(__file__).resolve().parents[2]
meta = json.loads((ROOT / "results/snv_a1/convert_meta.json").read_text())
out = ROOT / "data/snv_e1_feat/sparc/a0"
out.mkdir(parents=True, exist_ok=True)
m = load_model("multi", device="cuda")
for v in meta["targets"]:
    for j in (0, 1, 2):
        q = out / f"{v}_{j}.npz"
        if q.exists():
            continue
        c = m.encode(str(ROOT / f"results/snv_a0/wav/{v}_{j}.wav"))
        np.savez(q, ema=np.asarray(c["ema"], dtype=np.float32))
print("done", len(list(out.glob("*.npz"))), flush=True)
