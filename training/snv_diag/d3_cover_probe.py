"""段 D3(paper_snv rev27 §6・学習なし): 操作者の母集団への族の覆い。開発の組の本物の男声(JVNV の M1・M2・VCTK の男声 47 人)ごとに、最も近い声の ECAPA の余弦(spkrec-ecapa-voxceleb・LDA なし・系 2l の物差し)を、
E1 の学習の男声(results/snv_i8_m13/split.json・各声 t03〜t05)・E1 の男声から数を合わせて引いた組(20 回の平均)・段 D3 の (v) dvNNN・(r) drNNN(各声 t00〜t02)で比べる。
副 = 声ごとの高さの中央値(前段 v3)・長時間の平均スペクトルの帯域の割合(0〜100 Hz・16〜24 kHz)。試験の声(外郎売・namikawa の登録)は条件のパラメタとして記録だけ。
mode=pair = 同じ属性と種の (v) と (r) の対の ECAPA の余弦(対 別の (v) どうし)と、録音の文ごとの帯域の差(dB)= 録音の文が音を変えたか(out 既定 results/snv_d3/pair.json)。
    cd training && uv run --no-sync python snv_diag/d3_cover_probe.py [mode=pair] [out=results/snv_d3/cover.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p0_front as P

ROOT = Path(__file__).resolve().parents[2]
JV = ROOT / "data/jvnv/jvnv_v1"
VCTK = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
D3 = ROOT / "data/snv_d3"
E1 = ROOT / "data/snv_e1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_DRAW = 20


def band(x: np.ndarray, lo: float, hi: float) -> float:
    S = np.abs(librosa.stft(x, n_fft=2048, hop_length=480, window="hann", center=False)) ** 2
    fr = np.fft.rfftfreq(2048, 1 / P.SR)
    p = S.mean(1)
    return float(10 * np.log10(p[(fr >= lo) & (fr < hi)].sum() / (p.sum() + 1e-20) + 1e-12))


def f0_med(xs: list[np.ndarray]) -> float:
    f = np.concatenate([P.analyze(x[: P.n_frames(len(x)) * P.H])["f0"] for x in xs])
    return float(np.median(f[f > 0])) if (f > 0).any() else float("nan")


BANDS = ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 16000), (16000, 24000))


def pair(emb, load48) -> dict:
    man = json.loads((D3 / "manifest.json").read_text())
    rec = {v["voice"]: v["caption"].split("。")[-2] for v in man["voices"]}
    fr = np.fft.rfftfreq(2048, 1 / P.SR)

    def ltas(xs: list[np.ndarray]) -> np.ndarray:
        S = np.abs(librosa.stft(np.concatenate(xs), n_fft=2048, hop_length=480, window="hann", center=False)) ** 2
        q = S.mean(1)
        return 10 * np.log10(q / q.sum() + 1e-12)

    E, L = {}, {}
    for v in man["voices"]:
        xs = [load48(D3 / v["voice"] / f"t0{j}.wav") for j in range(3)]
        E[v["voice"]], L[v["voice"]] = emb(xs), ltas(xs)
    n = sum(1 for v in man["voices"] if v["arm"] == "dv")
    pc = [float(E[f"dv{i:03d}"] @ E[f"dr{i:03d}"]) for i in range(n)]
    oc = [float(np.mean([E[f"dv{i:03d}"] @ E[f"dv{j:03d}"] for j in range(n) if j != i])) for i in range(n)]
    dl: dict = {}
    for i in range(n):
        d = L[f"dr{i:03d}"] - L[f"dv{i:03d}"]
        dl.setdefault(rec[f"dr{i:03d}"], []).append([float(np.mean(d[(fr >= lo) & (fr < hi)])) for lo, hi in BANDS])
    return {"ecapa_pair_median": round(float(np.median(pc)), 4), "ecapa_other_dv_median": round(float(np.median(oc)), 4), "bands_hz": [list(b) for b in BANDS],
            "ltas_dr_minus_dv_db": {r: np.round(np.median(np.array(x), 0), 2).tolist() for r, x in dl.items()}}


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    load48 = lambda p: librosa.load(str(p), sr=P.SR, mono=True)[0].astype(np.float32)

    def emb(xs: list[np.ndarray]) -> np.ndarray:
        w = np.concatenate([librosa.resample(x, orig_sr=P.SR, target_sr=16000) for x in xs]).astype(np.float32)
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w)[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    if opt.get("mode") == "pair":
        r = pair(emb, load48)
        print(json.dumps(r, ensure_ascii=False, indent=1), flush=True)
        (ROOT / opt.get("out", "results/snv_d3/pair.json")).write_text(json.dumps(r, ensure_ascii=False, indent=1))
        return 0

    def voice(paths: list[Path], side: bool) -> dict:
        xs = [load48(p) for p in paths]
        r = {"e": emb(xs)}
        if side:
            x = np.concatenate(xs)
            r |= {"f0": f0_med(xs), "lf100": band(x, 0, 100), "hf16": band(x, 16000, 24000)}
        return r

    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    fam = {"e1_train": {v: voice([E1 / v / f"t0{j}.wav" for j in (3, 4, 5)], True) for v in split["train"] if v.startswith("em")},
           "e1_val": {v: voice([E1 / v / f"t0{j}.wav" for j in (3, 4, 5)], True) for v in split["val"] if v.startswith("em")}}
    man = json.loads((D3 / "manifest.json").read_text())
    for arm in ("dv", "dr"):
        fam[arm] = {v["voice"]: voice([D3 / v["voice"] / f"t0{j}.wav" for j in range(3)], True) for v in man["voices"] if v["arm"] == arm}
    q: dict = {}
    for s in ("M1", "M2"):
        q[f"jvnv/{s}"] = voice(sorted(JV.glob(f"{s}/*/regular/*_03.wav"))[:4], True)
    males = [ln.split()[0] for ln in (VCTK / "speaker-info.txt").read_text().splitlines()[1:] if len(ln.split()) > 2 and ln.split()[2] == "M"]
    for s in males:
        fs = sorted((VCTK / "wav48" / f"p{s}").glob("*.wav"))
        if len(fs) >= 34:
            q[f"vctk/p{s}"] = voice(fs[30:34], True)
    test = {f"real/{s}_enroll": voice([ROOT / f"data/snv_p_feat/real/{s}_enroll.wav"], True) for s in ("uirou", "namikawa")}
    M = {k: np.stack([r["e"] for r in d.values()]) for k, d in fam.items()}
    near = lambda Ms, e: float((Ms @ e).max())
    rng = random.Random(0)
    names = list(fam["e1_train"])
    ctrl = {n: [np.stack([fam["e1_train"][x]["e"] for x in rng.sample(names, n)]) for _ in range(N_DRAW)] for n in (48, 96)}
    rows = {}
    for k, r in {**q, **test}.items():
        e = r["e"]
        rows[k] = {"e1_train": near(M["e1_train"], e), "ctrl48": float(np.mean([near(Ms, e) for Ms in ctrl[48]])), "ctrl96": float(np.mean([near(Ms, e) for Ms in ctrl[96]])),
                   "dv": near(M["dv"], e), "dr": near(M["dr"], e), "dv+dr": near(np.concatenate([M["dv"], M["dr"]]), e),
                   "e1+dv+dr": near(np.concatenate([M["e1_train"], M["dv"], M["dr"]]), e), "f0": r["f0"], "lf100": r["lf100"], "hf16": r["hf16"]}
    dev = [k for k in q]
    med = lambda key: round(float(np.median([rows[k][key] for k in dev])), 4)
    summ = {"n_dev": len(dev), "nearest_median": {key: med(key) for key in ("e1_train", "ctrl48", "ctrl96", "dv", "dr", "dv+dr", "e1+dv+dr")},
            "gain_median": {"dv-ctrl48": round(float(np.median([rows[k]["dv"] - rows[k]["ctrl48"] for k in dev])), 4),
                            "dr-ctrl48": round(float(np.median([rows[k]["dr"] - rows[k]["ctrl48"] for k in dev])), 4),
                            "dr-dv": round(float(np.median([rows[k]["dr"] - rows[k]["dv"] for k in dev])), 4),
                            "dv+dr-ctrl96": round(float(np.median([rows[k]["dv+dr"] - rows[k]["ctrl96"] for k in dev])), 4),
                            "e1+dv+dr-e1_train": round(float(np.median([rows[k]["e1+dv+dr"] - rows[k]["e1_train"] for k in dev])), 4)},
            "in_family_val_male_nearest": round(float(np.median([near(M["e1_train"], r["e"]) for r in fam["e1_val"].values()])), 4),
            "jvnv": {k: {kk: round(v, 4) for kk, v in rows[k].items()} for k in dev if k.startswith("jvnv")},
            "test_record_only": {k: {kk: round(v, 4) for kk, v in rows[k].items()} for k in test}}
    side = {}
    for k, d in fam.items():
        f0s = np.array([r["f0"] for r in d.values()])
        side[k] = {"n": len(d), "f0_p5": round(float(np.nanpercentile(f0s, 5)), 1), "f0_med": round(float(np.nanmedian(f0s)), 1), "f0_min": round(float(np.nanmin(f0s)), 1),
                   "lf100_med": round(float(np.median([r["lf100"] for r in d.values()])), 2), "hf16_med": round(float(np.median([r["hf16"] for r in d.values()])), 2)}
    side["dev_real"] = {"n": len(dev), "f0_p5": round(float(np.nanpercentile([rows[k]["f0"] for k in dev], 5)), 1), "f0_med": round(float(np.nanmedian([rows[k]["f0"] for k in dev])), 1),
                        "lf100_med": round(float(np.median([rows[k]["lf100"] for k in dev])), 2), "hf16_med": round(float(np.median([rows[k]["hf16"] for k in dev])), 2)}
    summ["side"] = side
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
    out = ROOT / opt.get("out", "results/snv_d3/cover.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summ, "rows": {k: {kk: round(v, 4) for kk, v in r.items()} for k, r in rows.items()},
                               "captions": {v["voice"]: v["caption"] for v in man["voices"]}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
