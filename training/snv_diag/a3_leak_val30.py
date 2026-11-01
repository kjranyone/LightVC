"""学習の途中の打ち切りの閾値の較正: 検証 30 声の LEAK(leak_val)を A3 の p・n の最後と未学習の E で。"""
from __future__ import annotations
import sys, time
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train_snv_a3 import Model, leak_val, load_corpus, envelope
from train_snv_a2 import ROOT, VAL_VOICES
t = time.time()
voices = load_corpus()
names = sorted(voices)
val = names[-VAL_VOICES:]
print("loaded", len(voices), time.time() - t, flush=True)
for tag in ("diag_snv_a3_p", "diag_snv_a3_n"):
    m = Model().to("cuda")
    m.load_state_dict(torch.load(ROOT / f"results/{tag}/last.pt", map_location="cpu", weights_only=True)["model"])
    print(tag, leak_val(m, voices, val, "cuda"), time.time() - t, flush=True)
torch.manual_seed(0)
r = Model().to("cuda")
r.load_state_dict({k: v for k, v in torch.load(ROOT / "results/diag_snv_a3_p/last.pt", map_location="cpu", weights_only=True)["model"].items() if k.startswith("env_")}, strict=False)
print("untrained", leak_val(r, voices, val, "cuda"), time.time() - t, flush=True)
