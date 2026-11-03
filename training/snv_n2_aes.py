"""N2(学習なし): 物差しの候補 Audiobox Aesthetics(CE・CU・PC・PQ)を snv_n2_make.py の素材の全ての鍵で読む。依存は ../aes_env(uv・audiobox_aesthetics 0.0.4)。
事前登録 = results/snv_n2/prereg.yaml。出力 = results/snv_n2/aes.json。
    cd .. && ../aes_env/bin/python training/snv_n2_aes.py <in>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BS = 16


def main() -> int:
    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    res = {}
    for f in sorted(Path(sys.argv[1]).glob("*_*.npz")):
        if f.stem == "wavlm":
            continue
        d = np.load(f)
        ks = d.files
        for i in range(0, len(ks), BS):
            rows = pred.forward([{"path": torch.from_numpy(d[k]).float()[None], "sample_rate": 16000} for k in ks[i: i + BS]])
            for k, r in zip(ks[i: i + BS], rows):
                res[k] = r
        print(f.stem, len(ks), flush=True)
    (ROOT / "results/snv_n2/aes.json").write_text(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
