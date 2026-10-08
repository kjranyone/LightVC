"""倍音間の谷の深さ(500–3000Hz・有声フレーム・log|STFT| の p90 − p10・nfft 4096)を系ごとに測る。
耳の目盛り: 元音声 3.27・BigVGAN(合格)3.23・Y-S1 往復(不合格)3.06(2026-09-29・held21 の有声率上位 8 発話)。

    CUDA_VISIBLE_DEVICES=0 uv run python probe_contrast.py
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/kojirotanaka/kjranyone/LightVC/training")
import eval_nvoc as E, nvoc as N, make_nvoc_ear as M
from causal_codec import CausalCodec
import scipy.signal as ss
dev = "cuda"
items = sorted(E.held_items(), key=lambda it: -float((it["f0"] > 0).mean()))[:8]
def gen(path, nosrc=False):
    st = torch.load(path, map_location="cpu", weights_only=False)
    m = N.NVoc().to(dev); m.load_state_dict(st["ema"]); m.eval()
    return lambda it: E.run_nvoc(m, it["x"], (it["f0"] * 0) if nosrc else it["f0"], dev)[N.DELAY:]
ck = torch.load("../results/s1_3_c32/s1_3_c32_last.pt", map_location=dev, weights_only=False)
codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev); codec.load_state_dict(ck["ema"]); codec.eval()
def ys1(it):
    x = it["x"][:len(it["x"]) // 480 * 480]
    with torch.no_grad(): return codec.decode(codec.encode(torch.from_numpy(x).float()[None, None].to(dev)))[0, 0].cpu().numpy()
bv = M.bigvgan_fn(dev)
systems = {"original": lambda it: it["x"], "bigvgan(PASS)": lambda it: bv(it["x"]), "ys1(FAIL)": ys1,
           "nvoc_Rend40k": gen("../results/nvoc2/snap/ema_40k.pt"), "nvoc_r20nosrc": gen("../results/diag_nvoc_r20nosrc/last.pt", True),
           "nvoc4_80k_gan": gen("../results/nvoc4/last.pt")}
def feats(y, f0):
    f, t, Z = ss.stft(y, 48000, nperseg=4096, noverlap=4096 - 480)
    L = np.log(np.abs(Z) + 1e-6)
    band = (f >= 500) & (f <= 3000)
    T = min(L.shape[1], len(f0))
    v = f0[:T] > 0
    Lb = L[band][:, :T][:, v]
    contrast = np.percentile(Lb, 90, axis=0) - np.percentile(Lb, 10, axis=0)
    return float(np.mean(contrast))
res = {}
for nm, fn in systems.items():
    vals = []
    for it in items:
        y = np.asarray(fn(it), np.float64)
        vals.append(feats(y, it["f0"]))
    res[nm] = round(float(np.mean(vals)), 3)
    print(nm, "inter-harmonic contrast 500-3k (voiced, p90-p10 log|STFT|):", res[nm], flush=True)
