import sys, time
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train_snv_a3 import Model, load_corpus, xfer_ecapa, unit_tests
from train_snv_a2 import ROOT, VAL_VOICES
torch.manual_seed(0)
unit_tests()
voices = load_corpus()
val = sorted(voices)[-VAL_VOICES:]
for tag in ("diag_snv_a3c_w", "diag_snv_a3_n"):
    m = Model().to("cuda")
    m.load_state_dict(torch.load(ROOT / f"results/{tag}/last.pt", map_location="cpu", weights_only=True)["model"])
    nz = np.load(ROOT / f"results/{tag}/norm.npz")
    t = time.time()
    print(tag, xfer_ecapa(m, voices, val, nz["mu"], nz["sd"], "cuda"), f"{time.time() - t:.0f}s", flush=True)
