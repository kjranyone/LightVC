"""N0(学習なし): 音素の無い発声の物差しの較正。事前登録 = results/snv_n0/prereg.yaml。
ID_nv = 普通の発話の登録 3 本(A0)に対する ECAPA + LDA の女 24 声の top-1 を発声の種類ごとに・CAT = 発声の種類を声で分けた交差検証で当てる・
NAT_nv = UTMOS が音素の無い発声の劣化を読むか・HF = 有音の区間の 3〜12 kHz の割合(実録音の符号器の往復と比べる)。前後の無音は除いて読む(top_db 40)。
    cd training && uv run --no-sync python snv_n0_measure.py [snv_n0r]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.stats import binom

from snv_a0_measure import DEV, ROOT, SR, Embedders, buzz, griffinlim, load, lowpass, noise, to16
from snv_a0b_measure import proj

N0 = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "snv_n0")
A0 = ROOT / "results/snv_a0"
CATS = ("moan", "scream", "sigh", "pant", "cry", "laugh", "mixed")


def trim(y: np.ndarray) -> np.ndarray:
    return librosa.effects.trim(y, top_db=40)[0]


def hf_share(y: np.ndarray, sr: int) -> float:
    y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=44100) if sr != 44100 else y
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    e = S.sum(0)
    P = S[:, e > 0.01 * e.max()].sum(1)
    f = librosa.fft_frequencies(sr=44100, n_fft=2048)
    return float(100 * P[(f >= 3000) & (f < 12000)].sum() / P.sum())


def cat_feat(y16: np.ndarray) -> np.ndarray:
    m = np.log(librosa.feature.melspectrogram(y=y16.astype(np.float32), sr=SR, n_fft=1024, hop_length=256, n_mels=80) + 1e-6)
    e = m.max(0)
    m = m[:, e > e.max() - 6.0] if (e > e.max() - 6.0).sum() >= 3 else m
    d = np.diff(m, axis=1) if m.shape[1] > 1 else np.zeros_like(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)])


def main() -> int:
    its = json.loads((N0 / "wav/items.json").read_text())
    wav = {}
    for it in its:
        y, sr = load(N0 / f"wav/{it['voice']}_{it['cat']}_{it['k']}.wav")
        wav[(it["voice"], it["cat"], it["k"])] = (trim(y), sr)
    fem = [f"pf{k:03d}" for k in range(24)]
    res: dict = {"n_items": len(wav)}

    hf = defaultdict(list)
    for (v, c, _), (y, sr) in wav.items():
        hf[c].append(hf_share(y, sr))
    res["HF"] = {c: round(float(np.median(hf[c])), 2) for c in CATS} | {"all": round(float(np.median([x for c in CATS for x in hf[c]])), 2)}
    res["HF"]["dark"] = res["HF"]["all"] < 2.9
    print("HF", res["HF"], flush=True)

    emb = Embedders()
    T = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])
    def E(y: np.ndarray, sr: int) -> np.ndarray:
        with torch.no_grad():
            e = emb.ec.encode_batch(torch.from_numpy(to16(y, sr)).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)
    enr = {}
    for v in fem:
        ys = [load(A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)]
        enr[v] = E(np.concatenate([y for y, _ in ys]), ys[0][1])
    M = np.stack([enr[v] for v in fem])
    top1 = defaultdict(list)
    for (v, c, k), (y, sr) in wav.items():
        if v in fem:
            top1[c].append(fem[int(np.argmax(M @ E(y, sr)))] == v)
    for v in fem:
        for j in (3, 4, 5):
            y, sr = load(A0 / f"wav/{v}_{j}.wav")
            top1["normal_test"].append(fem[int(np.argmax(M @ E(y, sr)))] == v)
    res["ID_nv"] = {c: {"top1": round(float(np.mean(b)), 3), "n": len(b), "pass": bool(np.mean(b) >= 0.50)} for c, b in top1.items()}
    print("ID_nv", res["ID_nv"], flush=True)
    del emb
    torch.cuda.empty_cache()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    keys = sorted(wav)
    X = np.stack([cat_feat(to16(*wav[k])) for k in keys])
    y = np.array([CATS.index(k[1]) for k in keys])
    g = np.array([k[0] for k in keys])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    pred = cross_val_predict(clf, X, y, groups=g, cv=GroupKFold(6))
    yp = np.random.default_rng(0).permutation(y)
    pperm = cross_val_predict(clf, X, yp, groups=g, cv=GroupKFold(6))
    upper = float(binom.ppf(0.975, len(y), 1 / len(CATS)) / len(y))
    acc, accp = float(np.mean(pred == y)), float(np.mean(pperm == yp))
    conf = {CATS[i]: {CATS[j]: int(np.sum((y == i) & (pred == j))) for j in range(len(CATS))} for i in range(len(CATS))}
    res["CAT"] = {"acc": round(acc, 3), "perm_acc": round(accp, 3), "chance_upper95": round(upper, 3), "pass": bool(acc >= 0.80 and accp <= upper),
                  "per_class": {CATS[i]: round(float(np.mean(pred[y == i] == i)), 3) for i in range(len(CATS))}, "confusion": conf}
    print("CAT", {k: v for k, v in res["CAT"].items() if k != "confusion"}, flush=True)

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()

    def mos(y16: np.ndarray) -> float:
        with torch.no_grad():
            return float(utmos(torch.from_numpy(y16).float()[None].to(DEV), SR).item())

    wins = defaultdict(list)
    base = []
    for i, ((v, c, k), (y, sr)) in enumerate(sorted(wav.items())):
        if v not in fem:
            continue
        y16 = to16(y, sr).astype(np.float64)
        m0 = mos(y16)
        base.append(m0)
        for name, f in (("snr10", lambda z: noise(z, 10, i)), ("lpf4k", lambda z: lowpass(z, SR, 4000)), ("griffinlim", griffinlim), ("buzz", lambda z: buzz(z, SR))):
            wins[name].append(m0 > mos(f(y16)))
    res["NAT_nv"] = {n: {"win": round(float(np.mean(w)), 3), "n": len(w), "pass": bool(np.mean(w) >= 0.90)} for n, w in wins.items()}
    res["NAT_nv"]["clean_mean"] = round(float(np.mean(base)), 3)
    print("NAT_nv", res["NAT_nv"], flush=True)
    (N0 / "n0_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
