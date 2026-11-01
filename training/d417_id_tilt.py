import sys, json, glob
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
sys.path.insert(0, "/home/kojirotanaka/kjranyone/LightVC/training")
import train_rvoc as TR
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
dev = "cuda"
from transformers import WavLMForXVector
sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
import idloss as ID
ec = ID.Ecapa(dev)
@torch.no_grad()
def emb(x48):
    x = torch.from_numpy(resample_poly(x48, 16000, 48000).astype(np.float32))[None].to(dev)
    w = torch.nn.functional.normalize(sv(input_values=x).embeddings, dim=-1)[0].cpu().numpy()
    e = ec(torch.from_numpy(x48.astype(np.float32))[None].to(dev))[0].cpu().numpy()
    return w, e / np.linalg.norm(e)
meta = json.loads((H / "dev" / "cen_reg" / "meta.json").read_text())
ok = sorted(m["ti"] for m in meta if m["ok"])
cen = {ti: emb(sf.read(H / "dev" / "cen_reg" / f"cen_{ti:03d}.wav", dtype="float32")[0]) for ti in ok}
res = {}
for ti in ok:
    x, sr = sf.read(H / "dev" / "d37" / "_work" / f"ref_{ti:03d}.wav", dtype="float32")
    x = x[: 48000 * 8]
    for name, prm in (("orig", (0.0, 0.0)), ("tilt+6", (0.0, 6.0)), ("tilt-6", (0.0, -6.0)), ("lp8k", (8000.0, 0.0))):
        y = TR.channel_apply(x, prm) if name != "orig" else x
        w, e = emb(y)
        for m_, v in (("wsv", w), ("ecapa", e)):
            cs = {t: float(v @ cen[t][0 if m_ == "wsv" else 1]) for t in ok}
            margin = cs[ti] - np.mean([c for t, c in cs.items() if t != ti])
            res.setdefault((name, m_), []).append(margin)
base = {m_: np.array(res[("orig", m_)]) for m_ in ("wsv", "ecapa")}
for (name, m_), v in res.items():
    d = np.array(v) - base[m_]
    print(name, m_, "margin %.4f  diff vs orig %.4f (sd %.4f)" % (np.mean(v), d.mean(), d.std()))
