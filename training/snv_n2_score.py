"""N2(学習なし): 物差しの候補のうち UTMOS22 strong と WavLM-large の埋め込み(12 層目の時間の平均 + 標準偏差・正規化)を、snv_n2_make.py の素材の全ての鍵で読む。
事前登録 = results/snv_n2/prereg.yaml。出力 = results/snv_n2/utmos.json・<in>/wavlm.npz。
    cd training && uv run --no-sync python snv_n2_score.py <in>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import WavLMModel

from snv_a0_measure import DEV, ROOT, SR

OUT = ROOT / "results/snv_n2"
LAYER = 12


def main() -> int:
    src = Path(sys.argv[1])
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    wl = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEV).eval()
    mos, emb = {}, {}
    for f in sorted(src.glob("*_*.npz")):
        if f.stem == "wavlm":
            continue
        d = np.load(f)
        for k in d.files:
            w = torch.from_numpy(d[k]).float()[None].to(DEV)
            with torch.no_grad():
                mos[k] = float(utmos(w, SR).item())
                x = (w - w.mean()) / (w.std() + 1e-7)
                h = wl(x, output_hidden_states=True).hidden_states[LAYER][0]
                e = torch.cat([h.mean(0), h.std(0)])
            emb[k] = torch.nn.functional.normalize(e, dim=0).cpu().numpy().astype(np.float32)
        print(f.stem, len(d.files), flush=True)
    (OUT / "utmos.json").write_text(json.dumps(mos))
    np.savez(src / "wavlm.npz", **emb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
