"""検証 30 声の LEAK の物差しで SPARC の EMA(A2 の x の演奏)がどこに出るか = 打ち切りの閾値の良い側の錨。"""
from __future__ import annotations
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train_snv_a2 import FEAT, ROOT, VAL_VOICES, playing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
names = sorted(v["voice"] for v in man["voices"])
val = names[-VAL_VOICES:]
files = {v["voice"]: [f["t"] for f in v["files"]] for v in man["voices"]}
Z = {}
for n in val:
    Z[n] = []
    for t in files[n]:
        mel = np.load(FEAT / f"mel/e1/{n}/{t}.npy", mmap_mode="r")
        Z[n].append(playing(FEAT / f"sparc/e1/{n}/{t}.npz", mel.shape[1])[:, :12])
h = min(len(v) for v in Z.values()) // 2
Xtr = np.concatenate([z[::2] for n in val for z in Z[n][:h]])
ytr = np.concatenate([np.full(len(z[::2]), i) for i, n in enumerate(val) for z in Z[n][:h]])
te = [(i, z) for i, n in enumerate(val) for z in Z[n][h:]]
for name, clf in (("qda", QuadraticDiscriminantAnalysis(reg_param=0.05)), ("linear", LogisticRegression(C=1.0, max_iter=1000))):
    c = make_pipeline(StandardScaler(), clf).fit(Xtr, ytr)
    print(name, float(np.mean([c.classes_[np.argmax(c.predict_log_proba(z).sum(0))] == i for i, z in te])), flush=True)
