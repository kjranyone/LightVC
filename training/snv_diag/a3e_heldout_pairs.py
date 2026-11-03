"""A3e の事前検査の切り分け: 対照の損失で学んだ E の空間は、学習に使っていない組(検証 30 声を含む同じ文の組)でも調音の状態の対応になっているか。
E の出力(差し替えなし)で DTW を取り、揃えた相手の SPARC の EMA(物差しとしてだけ)と自分の EMA の相関を、学習の組と検証の組・同性と性別を跨ぐ組に分けて測る。
基準 = 声ごとの平均を引いた包絡の DTW(学習の組で 同性 0.733 / 跨ぐ 0.708)。学習の組 ≫ 検証の組なら組の覚え込み(組を増やす)、両方とも基準以下なら損失が状態以外で当てている。
    cd training && uv run --no-sync python snv_diag/a3e_heldout_pairs.py diag_snv_a3e_pre_b diag_snv_a3e_pre_l01 diag_snv_a3e_pre_l1
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
from train_snv_a3 import Model, envelope, load_corpus

DEV = "cuda"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmean([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(a.shape[1])]))


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
    pairs = [(a, b) for us in by.values() for a, b in itertools.combinations(us, 2) if a[0] != b[0]]
    held = [p for p in pairs if p[0][0] in val or p[1][0] in val]
    rng = np.random.default_rng(0)
    trn = [pairs[i] for i in rng.choice(len(pairs), 400, replace=False) if pairs[i][0][0] not in val and pairs[i][1][0] not in val]
    print("held-out pairs", len(held), "cross", sum(sex[a[0]] != sex[b[0]] for a, b in held), "train sample", len(trn), flush=True)
    keys = {x[:2] for p in held + trn for x in p}
    with torch.no_grad():
        env = {k: envelope(torch.from_numpy(np.asarray(voices[k[0]][k[1]][2], dtype=np.float32))[None].to(DEV)) for k in keys}
    ema = {x[:2]: playing(FEAT / f"sparc/e1/{x[0]}/{x[2]}.npz", voices[x[0]][x[1]][0].shape[1])[:, :12] for p in held + trn for x in p}
    vmu = {}
    with torch.no_grad():
        for n in {k[0] for k in keys}:
            vmu[n] = torch.cat([envelope(torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(DEV))[0] for _, _, w in voices[n]], 1).mean(1)
    gsd = torch.cat([env[k][0] for k in keys], 1).std(1)

    def feats(tag: str | None) -> dict:
        if tag is None:
            return {k: ((env[k][0].T - vmu[k[0]]) / gsd).cpu().numpy().astype(np.float64) for k in keys}
        m = Model().to(DEV).eval()
        m.load_state_dict(torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=True)["model"])
        with torch.no_grad():
            return {k: F.normalize(m.encode(env[k])[0], dim=-1).cpu().numpy().astype(np.float64) for k in keys}

    out = {}
    for tag in [None] + sys.argv[1:]:
        fz = feats(tag)
        r = defaultdict(list)
        for name, ps in (("train", trn), ("held", held)):
            for a, b in ps:
                _, wp = librosa.sequence.dtw(X=fz[a[:2]].T, Y=fz[b[:2]].T, metric="cosine")
                wp = wp[::-1]
                r[f"{name}_{'cross' if sex[a[0]] != sex[b[0]] else 'same'}"].append(corr(ema[a[:2]][wp[:, 0]], ema[b[:2]][wp[:, 1]]))
        out[tag or "envelope_cmn"] = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in sorted(r.items())}
        print(tag or "envelope_cmn", out[tag or "envelope_cmn"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
