"""A3 の漏れの較正: 同じ探り(A0 の 36 声・フレーム単位のロジスティック回帰・発話で対数確率を足す)を、生の包絡・未学習の E・学習済みの E に当てる。"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch, librosa
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train_snv_a3 import HOP, SR, Model, envelope
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
ROOT = Path("/home/kojirotanaka/kjranyone/LightVC")
A0 = ROOT / "results/snv_a0"
its = json.loads((A0 / "wav/items.json").read_text())
pv = [it for it in its if it["role"] in ("enroll", "test")]
voices = sorted({it["voice"] for it in pv})
sex = {it["voice"]: it["sex"] for it in pv}
tr_its = [it for it in pv if it["role"] == "enroll"]
te_its = [it for it in pv if it["role"] == "test"]
a0w = {(it["voice"], it["j"]): librosa.load(str(A0 / f"wav/{it['voice']}_{it['j']}.wav"), sr=SR, mono=True)[0] for it in pv}
dev = "cuda"
def feats(fn):
    out = {}
    with torch.no_grad():
        for k, y in a0w.items():
            out[k] = fn(torch.from_numpy(y[: (len(y) // HOP) * HOP])[None].to(dev))
    return out
def probe(lat, label):
    Xtr = [lat[(it["voice"], it["j"])] for it in tr_its]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(np.concatenate(Xtr), np.concatenate([np.full(len(f), label(it)) for f, it in zip(Xtr, tr_its)]))
    pred = [clf.classes_[np.argmax(clf.predict_log_proba(lat[(it["voice"], it["j"])]).sum(0))] for it in te_its]
    return float(np.mean(np.array(pred) == np.array([label(it) for it in te_its])))
def report(name, lat):
    print(name, {"voice36": probe(lat, lambda it: voices.index(it["voice"])), "sex": probe(lat, lambda it: int(sex[it["voice"]] == "F"))}, flush=True)
print("n voices", len(voices), "F", sum(sex[v] == "F" for v in voices), "train utts", len(tr_its), "test utts", len(te_its), flush=True)
m = Model().to(dev).eval()
ck = torch.load(ROOT / "results/diag_snv_a3_p/last.pt", map_location="cpu", weights_only=True)
m.load_state_dict(ck["model"])
mu, sd = m.env_mu, m.env_sd
env = feats(lambda y: ((envelope(y) - mu) / sd)[0].T.cpu().numpy())
report("raw_env80", env)
pca = PCA(12).fit(np.concatenate([env[(it["voice"], it["j"])] for it in tr_its]))
report("raw_env_pca12", {k: pca.transform(v) for k, v in env.items()})
report("trained_p", feats(lambda y: m.e((envelope(y) - mu) / sd)[0].cpu().numpy()))
torch.manual_seed(0)
r = Model().to(dev).eval()
report("untrained_E", feats(lambda y: r.e((envelope(y) - mu) / sd)[0].cpu().numpy()))
