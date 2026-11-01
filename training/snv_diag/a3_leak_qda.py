"""線形の探りは平均の差しか見ない。共分散まで見る探り(フレームの話者ごとのガウス = QDA・発話で対数尤度を足す)で、走行の正規化・E・SPARC の EMA を比べる。次元は 12 に揃える。"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import a3_leak_base as b
from a3_leak_norm import running
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def qprobe(lat, label):
    Xtr = [lat[(it["voice"], it["j"])] for it in b.tr_its]
    clf = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05)).fit(np.concatenate(Xtr), np.concatenate([np.full(len(f), label(it)) for f, it in zip(Xtr, b.tr_its)]))
    pred = [clf.classes_[np.argmax(clf.predict_log_proba(lat[(it["voice"], it["j"])]).sum(0))] for it in b.te_its]
    return float(np.mean(np.array(pred) == np.array([label(it) for it in b.te_its])))
def pca12(lat):
    p = PCA(12).fit(np.concatenate([lat[(it["voice"], it["j"])] for it in b.tr_its]))
    return {k: p.transform(v) for k, v in lat.items()}
def rep(name, lat):
    print(name, "linear", {"voice36": b.probe(lat, lambda it: b.voices.index(it["voice"])), "sex": b.probe(lat, lambda it: int(b.sex[it["voice"]] == "F"))},
          "QDA", {"voice36": qprobe(lat, lambda it: b.voices.index(it["voice"])), "sex": qprobe(lat, lambda it: int(b.sex[it["voice"]] == "F"))}, flush=True)
ema = {(it["voice"], it["j"]): np.load(b.ROOT / f"results/snv_a1/feat/{it['voice']}_{it['j']}.npz")["ema"] for it in b.pv}
rep("sparc_ema12", ema)
rep("raw_env_pca12", pca12(b.env))
rep("run_tau1_pca12", pca12({k: running(v, 1.0, False) for k, v in b.env.items()}))
rep("run_tau3_pca12", pca12({k: running(v, 3.0, False) for k, v in b.env.items()}))
rep("trained_p_E12", b.feats(lambda y: b.m.e((b.envelope(y) - b.mu) / b.sd)[0].cpu().numpy()))
