"""A1(synth_nam_vc §5・学習なし): SPARC の変換を A0〜A0c で較正した物差しで読む。事前登録 = results/snv_a1/prereg.yaml。
ID = ECAPA + LDA の動き m と 24 声中の top-1・INT = large-v3 の CER(入力の書き起こし 対 出力)・NAT = UTMOS の対の差・LEAK = SPARC の特徴のフレーム単位の探り。
    cd training && uv run --no-sync python snv_a1_measure.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, SR, Embedders, cer, kana, load, to16, world_grid_job
from snv_a0b_measure import fit_lda, proj, splits

OUT = ROOT / "results/snv_a1"
A0 = ROOT / "results/snv_a0"


def main() -> int:
    meta = json.loads((OUT / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    rd = lambda p: sf.read(str(p))[0].astype(np.float32)
    x = {s: rd(OUT / f"src/{s}.wav") for s in srcs}
    with ProcessPoolExecutor(10) as ex:
        dsp = dict(zip(srcs, [d[(12.0, 1.2)].astype(np.float32) for d in ex.map(world_grid_job, [(x[s].astype(np.float64), SR, (12.0,), (1.2,)) for s in srcs])]))
    outs = {}
    for s in srcs:
        outs[("resynth", s, None)] = rd(OUT / f"wav/{s}__resynth.wav")
        outs[("dsp", s, None)] = dsp[s]
        for v in tgts:
            for mode in ("std", "k12"):
                outs[(mode, s, v)] = rd(OUT / f"wav/{s}__{v}__{mode}.wav")
    print("loaded", len(outs), flush=True)

    sp = splits()
    pool = sp["pool"]
    emb = Embedders()
    X, y = [], []
    for i, s in enumerate(sp["lda"]):
        for p in pool[s]:
            X.append(emb(to16(*load(p)))["ecapa"])
            y.append(i)
    T = fit_lda(np.stack(X), np.array(y))
    E_t = {v: emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)]))["ecapa"] for v in tgts}
    E_s = {s: emb(rd(OUT / f"src/{s.rsplit('_', 1)[0]}_enroll.wav"))["ecapa"] for s in srcs}
    eo = {k: emb(w)["ecapa"] for k, w in outs.items()}
    eo_in = {s: emb(x[s])["ecapa"] for s in srcs}

    def id_stats(keys: list, lda: tuple | None) -> dict:
        Pt = {v: proj(E_t[v], lda) for v in tgts}
        Ps = {s: proj(E_s[s], lda) for s in E_s}
        ms, top = [], []
        for mode, s, v in keys:
            e = proj(eo[(mode, s, v if mode in ("std", "k12") else None)], lda)
            ms.append(float(e @ Pt[v] - e @ Ps[s]))
            sc = {u: float(e @ Pt[u]) for u in tgts}
            top.append(max(sc, key=sc.get) == v)
        ms = np.array(ms)
        return {"m_mean": float(ms.mean()), "m_pos_frac": float(np.mean(ms > 0)), "top1": float(np.mean(top)), "n": len(ms)}

    res: dict = {"prereg": "results/snv_a1/prereg.yaml"}
    idr = {}
    for name, lda in (("ecapa+lda", T), ("ecapa", None)):
        idr[name] = {mode: id_stats([(mode, s, v) for s in srcs for v in tgts], lda) for mode in ("std", "k12")}
        idr[name]["dsp_anchor"] = id_stats([("dsp", s, v) for s in srcs for v in tgts], lda)
        idr[name]["resynth_anchor"] = id_stats([("resynth", s, v) for s in srcs for v in tgts], lda)
        Pt = {v: proj(E_t[v], lda) for v in tgts}
        Ps = {s: proj(E_s[s], lda) for s in E_s}
        idr[name]["input_m_mean"] = float(np.mean([proj(eo_in[s], lda) @ Pt[v] - proj(eo_in[s], lda) @ Ps[s] for s in srcs for v in tgts]))
    for name in idr:
        for src_name in ("uirou", "namikawa"):
            idr[name][f"std_{src_name}"] = id_stats([("std", s, v) for s in srcs if s.startswith(src_name) for v in tgts], T if name == "ecapa+lda" else None)
    res["ID"] = idr
    print("ID", json.dumps(idr, indent=1), flush=True)
    del emb
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(s.text for s in segs))

    ref = {s: tr(x[s]) for s in srcs}
    cr = {}
    for mode in ("resynth", "dsp", "std", "k12"):
        keys = [k for k in outs if k[0] == mode]
        vals = [cer(ref[k[1]], tr(outs[k])) for k in keys]
        cr[mode] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)), "n": len(vals)}
        print("INT", mode, cr[mode], flush=True)
    cr["input_ref_examples"] = {s: ref[s][:60] for s in srcs[:2] + srcs[6:8]}
    res["INT"] = cr
    del asr
    torch.cuda.empty_cache()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    with torch.no_grad():
        mos = {k: float(utmos(torch.from_numpy(w).float()[None].to(DEV), SR).item()) for k, w in outs.items()}
        mos_in = {s: float(utmos(torch.from_numpy(x[s]).float()[None].to(DEV), SR).item()) for s in srcs}
    nr = {"input_mean": float(np.mean(list(mos_in.values())))}
    for mode in ("resynth", "dsp", "std", "k12"):
        nr[f"{mode}_mean"] = float(np.mean([v for k, v in mos.items() if k[0] == mode]))
    for mode in ("std", "k12"):
        pairs = [(mos[(mode, s, v)], mos[("dsp", s, None)], mos[("resynth", s, None)]) for s in srcs for v in tgts]
        nr[f"{mode}_win_vs_dsp"] = float(np.mean([a > b for a, b, _ in pairs]))
        nr[f"{mode}_win_vs_resynth"] = float(np.mean([a > c for a, _, c in pairs]))
    res["NAT"] = nr
    print("NAT", nr, flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    its = json.loads((A0 / "wav/items.json").read_text())
    voices = sorted({it["voice"] for it in its})
    sex = {it["voice"]: it["sex"] for it in its}
    tr_its = [it for it in its if it["role"] == "enroll"]
    te_its = [it for it in its if it["role"] == "test"]

    def feats(it: dict, kind: str) -> np.ndarray:
        d = np.load(OUT / f"feat/{it['voice']}_{it['j']}.npz")
        parts = {"a": [d["ema"]], "b": [d["ema"], d["loudness"], d["periodicity"]],
                 "c": [d["ema"], np.log(np.maximum(d["pitch"], 1.0)), d["loudness"], d["periodicity"]]}[kind]
        return np.concatenate(parts, axis=1)

    def probe(kind: str, label) -> float:
        Xtr = [feats(it, kind) for it in tr_its]
        ytr = [label(it) for it in tr_its]
        Xa = np.concatenate(Xtr)
        ya = np.concatenate([np.full(len(f), c) for f, c in zip(Xtr, ytr)])
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(Xa, ya)
        pred = [clf.classes_[np.argmax(clf.predict_log_proba(feats(it, kind)).sum(0))] for it in te_its]
        return float(np.mean(np.array(pred) == np.array([label(it) for it in te_its])))

    res["LEAK"] = {f"{k}_voice36": probe(k, lambda it: voices.index(it["voice"])) for k in ("a", "b", "c")}
    res["LEAK"].update({f"{k}_sex": probe(k, lambda it: int(sex[it["voice"]] == "F")) for k in ("a", "b", "c")})
    res["LEAK"]["ref"] = {"logmel_frame_voice36": 0.917, "chance_voice36": 1 / 36, "chance_sex": float(max(np.mean([sex[it["voice"]] == "F" for it in te_its]), 1 - np.mean([sex[it["voice"]] == "F" for it in te_its])))}
    print("LEAK", res["LEAK"], flush=True)
    (OUT / "a1_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
