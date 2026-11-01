"""較正: 変換に成功した A2 の x(演奏 = SPARC の EMA + 声源)を、A3 系の途中信号 XFER_ECAPA と同じ手順(検証 30 声・次の声の 2 本の参照・ECAPA + LDA の 30 声中の 1 位)で読む。"""
import sys
import numpy as np, torch, librosa
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from speechbrain.inference.speaker import EncoderClassifier
from snv_a2_eval import vocoder
from train_snv_a2 import FEAT, REF_UTT, ROOT, VAL_VOICES, G, load_corpus as load_a2
from train_snv_a3 import SR, load_corpus as load_a3
dev = "cuda"
v3 = load_a3()
v2 = load_a2()
val = sorted(v3)[-VAL_VOICES:]
assert val == sorted(v2)[-VAL_VOICES:]
ck = torch.load(ROOT / "results/diag_snv_a2_x/last.pt", map_location="cpu", weights_only=True)
g = G(ck["mode"]).to(dev).eval()
g.load_state_dict(ck["g"])
nz = np.load(ROOT / "results/diag_snv_a2_x/norm.npz")
mu, sd = nz["mu"], nz["sd"]
st = {}
for n in val:
    f = np.concatenate([np.exp(p[p[:, 14] > 0.5, 12]) for _, p in v2[n]])
    st[n] = (float(f.mean()), float(f.std()))
T = np.load(FEAT / "ecapa_lda_T.npz")
ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": dev})


def emb(y):
    y16 = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=16000)
    e = F.normalize(ec.encode_batch(torch.from_numpy(y16).float()[None].to(dev)).reshape(-1), dim=0).cpu().numpy()
    z = (e - T["mu"]) @ T["scalings"]
    return z / np.linalg.norm(z)


voc = vocoder()
to_b, to_a = [], []
with torch.no_grad():
    anc = np.stack([emb(np.concatenate([np.asarray(r[2], dtype=np.float32) for r in v3[n][REF_UTT:REF_UTT + 3]])) for n in val])
    for i, a_ in enumerate(val):
        j = (i + 1) % len(val)
        b_ = val[j]
        rm = np.concatenate([r[0] for r in v2[b_][:REF_UTT]], 1)
        rp = (np.concatenate([r[1] for r in v2[b_][:REF_UTT]], 0) - mu) / sd
        s = g.structure(torch.from_numpy(rm)[None].to(dev), torch.from_numpy(rp)[None].to(dev), torch.ones(1, rm.shape[1], dtype=torch.bool, device=dev))
        for mel, p in v2[a_][REF_UTT + 3:]:
            q = p.copy()
            f0 = (np.exp(q[:, 12]) - st[a_][0]) / st[a_][1] * st[b_][1] + st[b_][0]
            q[:, 12] = np.log(np.maximum(f0, 1.0))
            out = g(torch.from_numpy((q - mu) / sd)[None].to(dev), s)
            k = int(np.argmax(anc @ emb(voc(out).squeeze().cpu().numpy())))
            to_b.append(k == j), to_a.append(k == i)
print("A2x XFER_ECAPA", {"to_target": float(np.mean(to_b)), "read_as_source": float(np.mean(to_a)), "n": len(to_b)})
