"""A0b(A0 の不合格の分岐・学習なし): 事前登録 = results/snv_a0b/prereg.yaml。
ID = 特徴 3 種 × 本物の女声の LDA の有無を開発の組で選び A0 の組で一度だけ確かめる・CER = whisper small / large-v3-turbo・LEAK = フレーム単位の探り。
    cd training && uv run --no-sync python snv_a0b_measure.py
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

from snv_a0_measure import ALPHAS, DEV, ROOT, SR, Embedders, boot_ci, cer, eer, kana, load, lowpass, noise, to16, world_job

OUT = ROOT / "results/snv_a0b"
A0 = ROOT / "results/snv_a0"
PAIR = ROOT / "data/pair_tts_v4l"
FEATS = ("ecapa", "wavlm_sv", "wavlm_large")


def real_pool() -> dict:
    rows = list(csv.DictReader(open(ROOT / "data/kansei_vc/manifests/canonical_utterances.tsv"), delimiter="\t"))
    by = defaultdict(list)
    for r in rows:
        if r["source_type"] == "real_female" and 3.0 <= float(r["duration_sec"] or 0) <= 12.0:
            by[r["speaker_id"]].append(ROOT / "training" / r["wav_path"])
    return {s: sorted(v)[:6] for s, v in by.items() if len(v) >= 6}


def splits() -> dict:
    pool = real_pool()
    spk = sorted(pool)
    s3 = random.Random(0).sample(spk, 24)
    rest = [s for s in spk if s not in set(s3)]
    dev = random.Random(1).sample(rest, 24)
    rest2 = [s for s in rest if s not in set(dev)]
    tr = random.Random(2).sample(rest2, 400)
    return {"pool": pool, "S3": s3, "dev": dev, "lda": tr}


class Feat:
    def __init__(self) -> None:
        from transformers import AutoFeatureExtractor, WavLMModel
        self.base = Embedders()
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        self.wl = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEV).eval()

    @torch.no_grad()
    def __call__(self, y16: np.ndarray) -> dict:
        d = self.base(y16)
        inp = self.fe(y16, sampling_rate=SR, return_tensors="pt").to(DEV)
        hs = self.wl(**inp, output_hidden_states=True).hidden_states
        v = torch.stack([h.mean(1)[0] for h in hs]).mean(0)
        d["wavlm_large"] = torch.nn.functional.normalize(v, dim=0).cpu().numpy()
        return d


class Cache:
    def __init__(self, f: Feat) -> None:
        self.f, self.c = f, {}

    def file(self, p: Path) -> dict:
        k = str(p)
        if k not in self.c:
            y, sr = load(p)
            self.c[k] = self.f(to16(y, sr))
        return self.c[k]

    def concat(self, ps: list) -> dict:
        k = "+".join(map(str, ps))
        if k not in self.c:
            self.c[k] = self.f(np.concatenate([to16(*load(p)) for p in ps]))
        return self.c[k]


def fit_lda(X: np.ndarray, y: np.ndarray) -> tuple:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    mu = X.mean(0)
    k = min(150, len(set(y)) - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X - mu, y)
    return mu, lda.scalings_[:, :k]


def proj(e: np.ndarray, t: tuple | None) -> np.ndarray:
    if t is None:
        return e
    z = (e - t[0]) @ t[1]
    return z / np.linalg.norm(z)


def id_set(cache: Cache, sets: dict, feat: str, t: tuple | None) -> dict:
    E = {v: proj(cache.concat(enr)[feat], t) for v, (enr, _) in sets.items()}
    top, pos, neg = [], [], []
    for v, (_, tests) in sets.items():
        for p in tests:
            e = proj(cache.file(p)[feat], t)
            sc = {u: float(e @ E[u]) for u in E}
            top.append(max(sc, key=sc.get) == v)
            pos.append(sc[v])
            neg += [s for u, s in sc.items() if u != v]
    return {"top1": float(np.mean(top)), "eer": eer(np.array(pos), np.array(neg)), "n": len(top)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sp = splits()
    pool = sp["pool"]
    real_dev = {s: (pool[s][:3], pool[s][3:6]) for s in sp["dev"]}
    real_s3 = {s: (pool[s][:3], pool[s][3:6]) for s in sp["S3"]}
    iro_dev = {f"pf{k:03d}": ([PAIR / f"pf{k:03d}" / f"t{j:02d}.wav" for j in (1, 2, 3)], [PAIR / f"pf{k:03d}" / f"t{j:02d}.wav" for j in (4, 5, 6)]) for k in range(24, 40)}
    its = json.loads((A0 / "wav/items.json").read_text())
    fv = sorted({it["voice"] for it in its if it["sex"] == "F"})
    s1 = {v: ([A0 / f"wav/{v}_{j}.wav" for j in (0, 1, 2)], [A0 / f"wav/{v}_{j}.wav" for j in (3, 4, 5)]) for v in fv}
    res: dict = {"prereg": "results/snv_a0b/prereg.yaml", "n_real_pool": len(pool)}

    cache = Cache(Feat())
    Xs = {f: [] for f in FEATS}
    ys = []
    for i, s in enumerate(sp["lda"]):
        for p in pool[s]:
            d = cache.file(p)
            for f in FEATS:
                Xs[f].append(d[f])
            ys.append(i)
        if i % 100 == 0:
            print("lda feats", i, flush=True)
    T = {f: fit_lda(np.stack(Xs[f]), np.array(ys)) for f in FEATS}

    dev = {}
    for f in FEATS:
        for use in (False, True):
            t = T[f] if use else None
            key = f"{f}{'+lda' if use else ''}"
            a, b = id_set(cache, real_dev, f, t), id_set(cache, iro_dev, f, t)
            dev[key] = {"real_dev": a, "irodori_dev": b, "mean_top1": (a["top1"] + b["top1"]) / 2}
            print("ID dev", key, dev[key], flush=True)
    order = sorted(dev, key=lambda k: (-dev[k]["mean_top1"], "+lda" in k, FEATS.index(k.split("+")[0])))
    sel = order[0]
    sf, slda = sel.split("+")[0], T[sel.split("+")[0]] if "+lda" in sel else None
    res["ID_dev"] = dev
    res["ID_selected"] = sel
    conf = {"S1_irodori_v41": id_set(cache, s1, sf, slda), "S3_real_female": id_set(cache, real_s3, sf, slda)}
    ftest = [it for it in its if it["sex"] == "F" and it["role"] == "test"]
    with ProcessPoolExecutor(10) as ex:
        wf = list(ex.map(world_job, [(load(A0 / f"wav/{it['voice']}_{it['j']}.wav")[0], 48000, ALPHAS, ()) for it in ftest]))
    E = {v: proj(cache.concat(s1[v][0])[sf], slda) for v in fv}
    cos3 = {}
    for a in ALPHAS:
        cos3[a] = np.array([float(proj(cache.f(to16(w[("a", a)], 48000))[sf], slda) @ E[it["voice"]]) for it, w in zip(ftest, wf)])
    conf["C-ID3"] = {f"{a:.4f}": {"cos_own_mean": float(cos3[a].mean()), **({"diff_vs_1_ci": boot_ci(cos3[a] - cos3[1.0])} if a != 1.0 else {})} for a in ALPHAS}
    res["ID_confirm"] = conf
    print("ID selected", sel, json.dumps(conf, indent=1), flush=True)
    del cache
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel
    conds = {"clean": lambda y, sr: y, "snr20": lambda y, sr: noise(y, 20, 1), "snr10": lambda y, sr: noise(y, 10, 1), "snr0": lambda y, sr: noise(y, 0, 1),
             "lp4000": lambda y, sr: lowpass(y, sr, 4000), "lp2000": lambda y, sr: lowpass(y, sr, 2000)}
    fitems = [it for it in its if it["sex"] == "F" and it["role"] == "test"]
    mitems = [it for it in its if it["sex"] == "M" and it["role"] == "test"]
    audio = {(it["voice"], it["j"]): load(A0 / f"wav/{it['voice']}_{it['j']}.wav") for it in fitems + mitems}

    def run(model: WhisperModel, items: list, logprob: bool = False) -> dict:
        out = {}
        for c, f in conds.items():
            vals, lps = [], []
            for it in items:
                y, sr = audio[(it["voice"], it["j"])]
                segs, _ = model.transcribe(to16(f(y, sr), sr), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
                segs = list(segs)
                vals.append(cer(kana(it["text"]), kana("".join(s.text for s in segs))))
                lps.append(float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0)
            out[c] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)), "p90": float(np.percentile(vals, 90))}
            if logprob:
                out[c]["avg_logprob_mean"] = float(np.mean(lps))
        return out

    cr = {}
    for name, path in (("small", "small"), ("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")):
        m = WhisperModel(path, device=DEV, compute_type="float16")
        cr[name] = {"F": run(m, fitems)}
        print("CER", name, "F", cr[name]["F"], flush=True)
        del m
        torch.cuda.empty_cache()

    def passes(r: dict) -> bool:
        f = r["clean"]["median"]
        return f <= 0.10 and r["snr20"]["median"] < r["snr10"]["median"] < r["snr0"]["median"] and r["snr0"]["median"] >= f + 0.10

    ok = [n for n in cr if passes(cr[n]["F"])]
    csel = max(ok, key=lambda n: cr[n]["F"]["snr0"]["median"] - cr[n]["F"]["clean"]["median"]) if ok else None
    if csel:
        m = WhisperModel("small" if csel == "small" else "mobiuslabsgmbh/faster-whisper-large-v3-turbo", device=DEV, compute_type="float16")
        cr[csel]["M"] = run(m, mitems)
        cr[csel]["M_pass"] = passes(cr[csel]["M"])
        print("CER", csel, "M", cr[csel]["M"], flush=True)
        del m
    m = WhisperModel("large-v3", device=DEV, compute_type="float16")
    cr["large-v3_logprob_record"] = run(m, fitems, logprob=True)
    print("CER large-v3 logprob", cr["large-v3_logprob_record"], flush=True)
    del m
    torch.cuda.empty_cache()
    res["CER"] = cr
    res["CER_selected"] = csel

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import binom
    voices = sorted({it["voice"] for it in its})

    def frames(it: dict) -> np.ndarray:
        y, sr = load(A0 / f"wav/{it['voice']}_{it['j']}.wav")
        return np.log(librosa.feature.melspectrogram(y=to16(y, sr), sr=SR, n_fft=1024, hop_length=160, n_mels=80) + 1e-6).T

    tr = [it for it in its if it["role"] == "enroll"]
    te = [it for it in its if it["role"] == "test"]
    rng = np.random.default_rng(0)
    Ftr = []
    for it in tr:
        f = frames(it)
        Ftr.append(f[rng.choice(len(f), min(300, len(f)), replace=False)])
    Fte = [frames(it) for it in te]
    ytr_u = np.array([voices.index(it["voice"]) for it in tr])
    yte = np.array([voices.index(it["voice"]) for it in te])

    def probe(Ftr_: list, Fte_: list, yu: np.ndarray) -> float:
        X = np.concatenate(Ftr_)
        y = np.concatenate([np.full(len(f), c) for f, c in zip(Ftr_, yu)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X, y)
        pred = [clf.classes_[np.argmax(clf.predict_log_proba(f).sum(0))] for f in Fte_]
        return float(np.mean(np.array(pred) == yte))

    res["LEAK"] = {"logmel_frame": probe(Ftr, Fte, ytr_u), "permuted": probe(Ftr, Fte, rng.permutation(ytr_u)),
                   "random": probe([rng.standard_normal(f.shape) for f in Ftr], [rng.standard_normal(f.shape) for f in Fte], ytr_u),
                   "chance_upper95": float(binom.ppf(0.95, len(yte), 1 / len(voices)) / len(yte)), "n_test": int(len(yte))}
    print("LEAK", res["LEAK"], flush=True)
    (OUT / "a0b_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
