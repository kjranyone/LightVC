"""性別を跨ぐ状態の対応の崩れのうち、声道長(周波数軸の一様な伸縮)で説明できる分(学習なし)。
声ごとに長期の平均包絡(声の全発話)を、学習の声の平均の長期包絡へ最もよく重なる伸縮 α で読み直し(envelope の f / α・α は 0.75〜1.33 の対数 41 点)、
(1) 同じ文の組の DTW の揃いの質(揃えた相手の SPARC の EMA との相関・EMA は物差しだけ)と、
(2) A0 の Irodori の男声 → 目標の女声の参照の柔らかい最近傍(温度 0.1・cosine)で読んだ参照の EMA と今の EMA の相関(IM・E の空間の対応と同じ手順)を、
声ごとの平均を引いた包絡(cmn)と、伸縮で正規化してから平均を引いた包絡(vtl)で比べる。α の推定は声の全発話の統計 = 推論では操作者の登録か因果な走行推定に置き換える前提。
    cd training && uv run --no-sync python snv_diag/a3f_vtl_check.py
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_snv_a2 import FEAT, ROOT, VAL_VOICES, playing
from train_snv_a3 import HOP, envelope, load_corpus

DEV = "cuda"
ALPHAS = np.exp(np.linspace(np.log(0.75), np.log(1.33), 41))
A0 = ROOT / "results/snv_a0"
A1 = ROOT / "results/snv_a1"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(a.shape[1])]))


def env_of(w: np.ndarray, alpha: float = 1.0) -> torch.Tensor:
    y = torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(DEV)
    return envelope(y, torch.tensor([alpha], device=DEV) if alpha != 1.0 else None)[0].T


def fit_alpha(ws: list[np.ndarray], ref: torch.Tensor) -> float:
    best, ba = None, 1.0
    with torch.no_grad():
        for a in ALPHAS:
            lt = torch.cat([env_of(w, float(a)) for w in ws]).mean(0)
            d = float(((lt - lt.mean()) - (ref - ref.mean())).pow(2).mean())
            if best is None or d < best:
                best, ba = d, float(a)
    return ba


def main() -> int:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    voices = load_corpus()
    val = set(sorted(voices)[-VAL_VOICES:])
    sex = {v["voice"]: v["sex"] for v in man["voices"]}
    by = defaultdict(list)
    for v in man["voices"]:
        if v["voice"] not in voices:
            continue
        k = 0
        for f in v["files"]:
            if all((FEAT / f"{d}/e1/{v['voice']}/{f['t']}.{e}").exists() for d, e in (("mel", "npy"), ("sparc", "npz"), ("wav44", "npy"))):
                by[f["text"]].append((v["voice"], k, f["t"]))
                k += 1
    pairs = [(a, b) for us in by.values() for a, b in itertools.combinations(us, 2) if a[0] != b[0] and sex[a[0]] != sex[b[0]]]
    rng = np.random.default_rng(0)
    pairs = [pairs[i] for i in rng.choice(len(pairs), min(200, len(pairs)), replace=False)]
    tr = [n for n in sorted(voices) if n not in val]
    with torch.no_grad():
        ref = torch.stack([torch.cat([env_of(w) for _, _, w in voices[n][:3]]).mean(0) for n in rng.choice(tr, 120, replace=False)]).mean(0)

    alpha = {}
    need = sorted({x[0] for p in pairs for x in p})
    for n in need:
        alpha[n] = fit_alpha([w for _, _, w in voices[n]], ref)
    am = [alpha[n] for n in need if sex[n] == "M"]
    af = [alpha[n] for n in need if sex[n] == "F"]
    print("alpha E1 male median", round(float(np.median(am)), 3), "female", round(float(np.median(af)), 3), "n", len(am), len(af), flush=True)

    def feats(n: str, us: list[int], a: float) -> dict:
        with torch.no_grad():
            es = {u: env_of(voices[n][u][2], a) for u in range(len(voices[n]))}
            mu = torch.cat(list(es.values())).mean(0)
        return {u: (es[u] - mu).cpu().numpy().astype(np.float64) for u in us}

    res = defaultdict(list)
    for a_, b_ in pairs:
        ea = playing(FEAT / f"sparc/e1/{a_[0]}/{a_[2]}.npz", voices[a_[0]][a_[1]][0].shape[1])[:, :12]
        eb = playing(FEAT / f"sparc/e1/{b_[0]}/{b_[2]}.npz", voices[b_[0]][b_[1]][0].shape[1])[:, :12]
        for kind in ("cmn", "vtl"):
            fa = feats(a_[0], [a_[1]], alpha[a_[0]] if kind == "vtl" else 1.0)[a_[1]]
            fb = feats(b_[0], [b_[1]], alpha[b_[0]] if kind == "vtl" else 1.0)[b_[1]]
            _, wp = librosa.sequence.dtw(X=fa.T, Y=fb.T, metric="cosine")
            wp = wp[::-1]
            res[f"dtw_cross_{kind}"].append(corr(ea[wp[:, 0]], eb[wp[:, 1]]))
    print({k: round(float(np.mean(v)), 3) for k, v in res.items()}, "n", len(pairs), flush=True)

    its = [it for it in json.loads((A0 / "wav/items.json").read_text()) if it["role"] in ("enroll", "test")]
    meta = json.loads((A1 / "convert_meta.json").read_text())
    tgts = meta["targets"]
    a0w = {(it["voice"], it["j"]): librosa.load(str(A0 / f"wav/{it['voice']}_{it['j']}.wav"), sr=44100, mono=True)[0] for it in its}
    a0w = {k: w[: (len(w) // HOP) * HOP] for k, w in a0w.items()}
    a0v = sorted({it["voice"] for it in its})
    a0alpha = {v: fit_alpha([a0w[k] for k in a0w if k[0] == v], ref) for v in a0v}
    a0sex = {it["voice"]: it["sex"] for it in its}
    print("alpha A0 male median", round(float(np.median([a0alpha[v] for v in a0v if a0sex[v] == "M"])), 3), "female", round(float(np.median([a0alpha[v] for v in a0v if a0sex[v] == "F"])), 3), flush=True)

    def a0feat(v: str, kind: str) -> dict:
        a = a0alpha[v] if kind == "vtl" else 1.0
        with torch.no_grad():
            es = {k: env_of(w, a) for k, w in a0w.items() if k[0] == v}
            mu = torch.cat(list(es.values())).mean(0)
            return {k: F.normalize(e - mu, dim=-1) for k, e in es.items()}

    def ema_a0(k: tuple, n: int) -> np.ndarray:
        d = np.load(A1 / f"feat/{k[0]}_{k[1]}.npz")
        raw = d["ema"]
        ts, td = np.arange(len(raw)) / 50.0, np.arange(n) / (44100 / HOP)
        return np.stack([np.interp(td, ts, raw[:, j]) for j in range(12)], 1)

    tref = {}
    for v in tgts:
        n = np.load(FEAT / f"mel/a0enroll/{v}.npy").shape[1]
        tref[v] = (librosa.load(str(FEAT / f"wav44/a0enroll__{v}.wav"), sr=44100, mono=True)[0][: n * HOP], playing(FEAT / f"sparc/a1src/tref_{v}.npz", n)[:, :12])
    test_keys = {(it["voice"], it["j"]) for it in its if it["role"] == "test"}
    out = {}
    for kind in ("cmn", "vtl"):
        cs = []
        for v in tgts[::6]:
            a = a0alpha[v] if kind == "vtl" else 1.0
            with torch.no_grad():
                zr = env_of(tref[v][0], a)
                mus = torch.cat([env_of(w, a) for k, w in a0w.items() if k[0] == v]).mean(0)
                zr = F.normalize(zr - mus, dim=-1)
            for m_ in [x for x in a0v if a0sex[x] == "M"]:
                fq = a0feat(m_, kind)
                for k in [k for k in fq if k in test_keys]:
                    att = ((fq[k] @ zr.T) / 0.1).softmax(-1).double().cpu().numpy()
                    T = min(att.shape[0], len(ema_a0(k, att.shape[0])))
                    pa = att @ tref[v][1][: att.shape[1]]
                    cs.append(corr(pa[:T], ema_a0(k, att.shape[0])[:T]))
        out[f"IM_nn_{kind}"] = (round(float(np.mean(cs)), 3), len(cs))
    print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
