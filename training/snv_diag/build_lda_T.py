"""ECAPA + LDA の変換 T(snv_a3_eval と同じ手順)を一度だけ作り FEAT/ecapa_lda_T.npz に保存する。学習中の途中信号 XFER_ECAPA が読む。"""
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from snv_a0_measure import Embedders, load, to16
from snv_a0b_measure import fit_lda, splits
from train_snv_a2 import FEAT
sp = splits()
emb = Embedders()
X, yy = [], []
for i, s in enumerate(sp["lda"]):
    for p in sp["pool"][s]:
        X.append(emb(to16(*load(p)))["ecapa"])
        yy.append(i)
mu, sc = fit_lda(np.stack(X), np.array(yy))
np.savez(FEAT / "ecapa_lda_T.npz", mu=mu, scalings=sc, n=len(X))
print("saved", FEAT / "ecapa_lda_T.npz", sc.shape, len(X))
