"""途中の信号 XFER の較正: A3 の n・p の最後で、検証 30 声の変換が源と読まれる率。"""
from __future__ import annotations
import sys
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train_snv_a3 import Model, load_corpus, xfer_val
from train_snv_a2 import ROOT, VAL_VOICES
voices = load_corpus()
val = sorted(voices)[-VAL_VOICES:]
for tag in ("diag_snv_a3_n", "diag_snv_a3_p"):
    m = Model().to("cuda")
    m.load_state_dict(torch.load(ROOT / f"results/{tag}/last.pt", map_location="cpu", weights_only=True)["model"])
    nz = np.load(ROOT / f"results/{tag}/norm.npz")
    print(tag, xfer_val(m, voices, val, nz["mu"], nz["sd"], "cuda"), flush=True)
