"""耳の数値条件の目盛り: 耳で不合格が確定した Y-S1 c32 codec 往復(2026-09-24)を held21 の同じ物差し(eval_nvoc.metrics)で測る。

    CUDA_VISIBLE_DEVICES=0 uv run python calib_nvoc_gate.py
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/kojirotanaka/kjranyone/LightVC/training")
import eval_nvoc as E
from causal_codec import CausalCodec
from scipy.signal import correlate
dev = "cuda"
ck = torch.load("../results/s1_3_c32/s1_3_c32_last.pt", map_location=dev, weights_only=False)
codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev); codec.load_state_dict(ck["ema"]); codec.eval()
rows, lags = [], []
for it in E.held_items():
    x = it["x"][:len(it["x"]) // 480 * 480]
    with torch.no_grad():
        y = codec.decode(codec.encode(torch.from_numpy(x).float()[None, None].to(dev)))[0, 0].cpu().numpy()
    c = correlate(y[:96000], x[:96000], mode="full")[96000 - 1:96000 - 1 + 2000]
    lag = int(np.argmax(np.abs(c)))
    lags.append(lag)
    rows.append(E.metrics(y, x, lag, dev))
print("lags", sorted(set(lags))[:5])
print("Y-S1 c32 codec round trip (ear FAIL 2026-09-24) held21:", {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in rows[0]})
