"""A3c の食い違いの切り分け: 検証 30 声の XFER(mel の QDA で源と読む率 0.00)と A0 の O(ECAPA で源と読む率 0.29)。
物差しを入れ替える: 検証 30 声の変換を ECAPA + LDA で読み、A0 の O と M を mel の QDA で読む。A3 の n を較正に並べる。"""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from snv_a0_measure import DEV, ROOT, Embedders, load, to16
from snv_a0b_measure import fit_lda, proj, splits
from snv_a2_eval import map_pitch, vocoder
from snv_a3_eval import A0, A1, male_segments, wav44
from train_snv_a2 import FEAT, REF_UTT, VAL_VOICES, playing
from train_snv_a3 import SR, Model, envelope, load_corpus

TAGS = ("diag_snv_a3_n", "diag_snv_a3c_w")
voices = load_corpus()
val = sorted(voices)[-VAL_VOICES:]
meta = json.loads((A1 / "convert_meta.json").read_text())
srcs, tgts = meta["sources"], meta["targets"]
drows = json.loads((A1 / "diag/meta.json").read_text())
males = male_segments(srcs)
tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
st = {}
for n in val:
    f = np.concatenate([np.exp(s_[s_[:, 2] > 0.5, 0]) for _, s_, _ in voices[n]])
    st[n] = (float(f.mean()), float(f.std()))


def qda(X: list, y: list):
    return make_pipeline(StandardScaler(), PCA(40), QuadraticDiscriminantAnalysis(reg_param=0.05)).fit(np.concatenate(X), np.concatenate(y))


def qpred(clf, m: np.ndarray) -> int:
    return int(clf.classes_[np.argmax(clf.predict_log_proba(m.T).sum(0))])


clf_val = qda([m.T for n in val for m, _, _ in voices[n][REF_UTT:REF_UTT + 3]],
              [np.full(m.shape[1], i) for i, n in enumerate(val) for m, _, _ in voices[n][REF_UTT:REF_UTT + 3]])
a0mel = {v: np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32) for v in tgts}
clf_a0 = qda([a0mel[v].T for v in tgts], [np.full(a0mel[v].shape[1], i) for i, v in enumerate(tgts)])
ti = {v: i for i, v in enumerate(tgts)}
d0real = [qpred(clf_a0, np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").astype(np.float32)) == ti[r["target"]] for r in drows]
print("melQDA A0 real_acc(D0 の本物の mel)", float(np.mean(d0real)), flush=True)

voc = vocoder()
mels: dict = {}
for tag in TAGS:
    d = ROOT / "results" / tag
    m = Model().to(DEV).eval()
    m.load_state_dict(torch.load(d / "last.pt", map_location="cpu", weights_only=True)["model"])
    nz = np.load(d / "norm.npz")
    mu, sd = nz["mu"], nz["sd"]
    with torch.no_grad():
        for i, a_ in enumerate(val):
            j = (i + 1) % len(val)
            refs = voices[val[j]][:REF_UTT]
            rm = np.concatenate([r[0] for r in refs], 1)
            rs = (np.concatenate([r[1] for r in refs], 0) - mu) / sd
            rw = np.concatenate([np.asarray(r[2], dtype=np.float32) for r in refs])
            s = m.structure(torch.from_numpy(rm)[None].to(DEV), envelope(torch.from_numpy(rw)[None].to(DEV)), torch.from_numpy(rs)[None].to(DEV),
                            torch.ones(1, rm.shape[1], dtype=torch.bool, device=DEV))
            for u, (mel, src, w) in enumerate(voices[a_][REF_UTT + 3:]):
                q = src.copy()
                f0 = (np.exp(q[:, 0]) - st[a_][0]) / st[a_][1] * st[val[j]][1] + st[val[j]][0]
                q[:, 0] = np.log(np.maximum(f0, 1.0))
                mels[("X", tag, i, j, u)] = m(envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(DEV)), torch.from_numpy((q - mu) / sd)[None].to(DEV), s)[0].cpu().numpy()
                if tag == TAGS[0]:
                    mels[("Xreal", "-", i, i, u)] = mel
        S = {}
        for v in tgts:
            n = a0mel[v].shape[1]
            rw = wav44(FEAT / f"wav44/a0enroll__{v}.wav", n)
            rs = (playing(FEAT / f"sparc/a1src/tref_{v}.npz", n)[:, 12:15] - mu) / sd
            S[v] = m.structure(torch.from_numpy(a0mel[v])[None].to(DEV), envelope(torch.from_numpy(rw)[None].to(DEV)),
                               torch.from_numpy(rs)[None].to(DEV), torch.ones(1, n, dtype=torch.bool, device=DEV))

        def run(y: np.ndarray, p_npz, v: str) -> np.ndarray:
            p = map_pitch(playing(p_npz, len(y) // 512), np.load(p_npz)["pitch_stats"].reshape(-1)[:2], tstats[v])
            return m(envelope(torch.from_numpy(y)[None].to(DEV)), torch.from_numpy((p[:, 12:15] - mu) / sd)[None].to(DEV), S[v])[0].cpu().numpy()

        for r in drows:
            v, gi, o = r["target"], r["g"], r["d2_src"]
            n = np.load(FEAT / f"mel/a1diag/D0_{v}_{gi}.npy").shape[1]
            mels[("O", tag, o, v, gi)] = run(wav44(FEAT / f"wav44/a1diag__D0_{o}_{gi}.wav", n), FEAT / f"sparc/a1diag/src_{o}_{gi}.npz", v)
        for s_ in srcs:
            for v in tgts:
                mels[("M", tag, s_, v, 0)] = run(males[s_], FEAT / f"sparc/a1src/{s_}.npz", v)
    del m
    torch.cuda.empty_cache()

res: dict = {}
for tag in TAGS:
    ko = [k for k in mels if k[0] == "O" and k[1] == tag]
    km = [k for k in mels if k[0] == "M" and k[1] == tag]
    po = {k: tgts[qpred(clf_a0, mels[k])] for k in ko}
    pm = {k: tgts[qpred(clf_a0, mels[k])] for k in km}
    res[f"melQDA_O_{tag}"] = {"to_target": float(np.mean([po[k] == k[3] for k in ko])), "read_as_source": float(np.mean([po[k] == k[2] for k in ko])), "n": len(ko)}
    res[f"melQDA_M_{tag}"] = {"to_target": float(np.mean([pm[k] == k[3] for k in km])), "n": len(km)}
    kx = [k for k in mels if k[0] == "X" and k[1] == tag]
    px = {k: qpred(clf_val, mels[k]) for k in kx}
    res[f"melQDA_X_{tag}"] = {"to_target": float(np.mean([px[k] == k[3] for k in kx])), "read_as_source": float(np.mean([px[k] == k[2] for k in kx])), "n": len(kx)}
    print(tag, {k: v for k, v in res.items() if tag in k}, flush=True)

with torch.no_grad():
    wavs = {k: voc(torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))[None].to(DEV)).squeeze().cpu().numpy() for k, v in mels.items() if k[0] in ("X", "Xreal")}
del voc
torch.cuda.empty_cache()
sp = splits()
emb = Embedders()
X, yy = [], []
for i, s in enumerate(sp["lda"]):
    for p in sp["pool"][s]:
        X.append(emb(to16(*load(p)))["ecapa"])
        yy.append(i)
T = fit_lda(np.stack(X), np.array(yy))
Ev = [proj(emb(to16(np.concatenate([np.asarray(r[2], dtype=np.float64) for r in voices[n][REF_UTT:REF_UTT + 3]]), SR))["ecapa"], T) for n in val]
E = {k: proj(emb(to16(w.astype(np.float64), SR))["ecapa"], T) for k, w in wavs.items()}
best = {k: int(np.argmax([float(E[k] @ e) for e in Ev])) for k in E}
kr = [k for k in E if k[0] == "Xreal"]
res["ecapa_X_real_acc_rendered"] = float(np.mean([best[k] == k[2] for k in kr]))
for tag in TAGS:
    kx = [k for k in E if k[0] == "X" and k[1] == tag]
    res[f"ecapa_X_{tag}"] = {"to_target": float(np.mean([best[k] == k[3] for k in kx])), "read_as_source": float(np.mean([best[k] == k[2] for k in kx])), "n": len(kx)}
res["melQDA_A0_real_acc"] = float(np.mean(d0real))
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
