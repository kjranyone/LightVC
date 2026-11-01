"""話者の統計の差し替え(白色化 → 別の話者の平均・共分散で着色)が、包絡の声をどれだけ別の話者へ移すか。
探りは差し替えに使った統計と別の型にする(同じガウスでは循環): 1 フレームの GMM(8 成分)と、前後 2 フレームを積んだ QDA(時間の共分散)。
登録 = A0 の enroll 3 本(統計と探りの学習)・試験 = test 3 本を別の声へ差し替えて、元の声 A と差し替え先 B のどちらと読まれるか。"""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import a3_leak_base as b
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
env = b.env
V = b.voices
def sqrtm(C, inv=False):
    w, U = np.linalg.eigh(C)
    w = np.maximum(w, 1e-6)
    return (U * (w ** (-0.5 if inv else 0.5))) @ U.T
stats = {}
for v in V:
    X = np.concatenate([env[(it["voice"], it["j"])] for it in b.tr_its if it["voice"] == v])
    mu = X.mean(0); C = np.cov(X.T); C = C + 0.05 * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
    stats[v] = (mu, sqrtm(C), sqrtm(C, True))
rng = np.random.default_rng(0)
perm = {v: V[(i + 1 + rng.integers(0, len(V) - 1)) % len(V)] for i, v in enumerate(V)}
assert all(perm[v] != v for v in V)
def wct(x, a, bb):
    ma, _, Ca_i = stats[a]; mb, Cb, _ = stats[bb]
    return mb + (x - ma) @ Ca_i @ Cb
def stack(x, k=2):
    p = np.pad(x, ((k, k), (0, 0)), mode="edge")
    return np.concatenate([p[i:i + len(x)] for i in range(2 * k + 1)], 1)
tests = [(it["voice"], env[(it["voice"], it["j"])]) for it in b.te_its]
def run(name, feat, fit_eval):
    tr = {v: np.concatenate([feat(env[(it["voice"], it["j"])]) for it in b.tr_its if it["voice"] == v]) for v in V}
    score = fit_eval(tr)
    raw = np.mean([score(feat(x)) == a for a, x in tests])
    sw = [score(feat(wct(x, a, perm[a]))) for a, x in tests]
    print(name, "raw→A", round(raw, 3), "swapped→A", round(np.mean([s == a for s, (a, _) in zip(sw, tests)]), 3), "swapped→B", round(np.mean([s == perm[a] for s, (a, _) in zip(sw, tests)]), 3), flush=True)
def gmm_fit(k, d):
    def f(tr):
        pca = make_pipeline(StandardScaler(), PCA(d)).fit(np.concatenate(list(tr.values())))
        gm = {v: GaussianMixture(k, covariance_type="full", reg_covar=1e-3, random_state=0).fit(pca.transform(x)) for v, x in tr.items()}
        return lambda x: max(V, key=lambda v: gm[v].score_samples(pca.transform(x)).sum())
    return f
def qda_fit(d):
    def f(tr):
        X = np.concatenate(list(tr.values())); y = np.concatenate([np.full(len(x), i) for i, x in enumerate(tr.values())])
        c = make_pipeline(StandardScaler(), PCA(d), QuadraticDiscriminantAnalysis(reg_param=0.05)).fit(X, y)
        return lambda x: V[int(c.classes_[np.argmax(c.predict_log_proba(x).sum(0))])]
    return f
run("qda80_frame(循環の確認)", lambda x: x, qda_fit(80))
run("gmm8_pca20_frame", lambda x: x, gmm_fit(8, 20))
run("qda_stack5_pca40", stack, qda_fit(40))
run("gmm8_stack5_pca40", stack, gmm_fit(8, 40))
