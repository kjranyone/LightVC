"""A6 の起動前の検査 I の探り(学習なし・記録だけ): 抜く種類(叫び・喘ぎ・混ざり)の演奏(SPARC の 15 次元・学習の正規化)が、学習に残る分(570 声の普通の発話 + 見た 4 種)から
どれだけ外れるか。学習に無い 30 声(E1V)の音素の無い発声のフレームから学習の分の最近傍までの距離を、抜く種類と見た種類で比べる(比 > 1 = 抜く種類が分布の外)。
出力 = results/diag_snv_a6/ood_probe.json。
    cd training && uv run --no-sync python snv_diag/a6_ood_probe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT
from train_snv_a2 import FEAT, VAL_VOICES
from train_snv_a4 import load_corpus

HELD = frozenset({"scream", "moan", "mixed"})
N_REF = 300000


def main() -> int:
    rng = np.random.default_rng(0)
    nz = np.load(ROOT / "results/diag_snv_a4_c/norm.npz")
    mu, sd = nz["mu"], nz["sd"]
    voices = load_corpus(HELD)
    names = sorted(voices)
    val, tr = names[-VAL_VOICES:], names[:-VAL_VOICES]
    ref_n = np.concatenate([p for n in tr for _, p in voices[n]["n"]], 0)
    ref_v = np.concatenate([p for n in tr for _, p in voices[n]["v"]], 0)
    ref = np.concatenate([ref_n[rng.choice(len(ref_n), N_REF // 2, replace=False)], ref_v[rng.choice(len(ref_v), min(N_REF // 2, len(ref_v)), replace=False)]], 0)
    R = torch.from_numpy((ref - mu) / sd).float().to(DEV)
    from train_snv_a2 import playing
    nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in val]
    by_cat: dict = {}
    for it in nvman:
        n = f"{it['voice']}/{it['cat']}_0"
        p = playing(FEAT / f"sparc/e1nv/{n}.npz", np.load(FEAT / f"mel/e1nv/{n}.npy").shape[1])
        Q = torch.from_numpy((p - mu) / sd).float().to(DEV)
        d = torch.cat([torch.cdist(Q[i:i + 512], R).min(1).values for i in range(0, len(Q), 512)]).cpu().numpy()
        by_cat.setdefault(it["cat"], []).append(d)
    res = {"note": "記録だけ・A6 の起動前の検査 I の探り", "n_ref_frames": int(len(R)), "nn_dist_median_by_cat": {}, "nn_dist_p90_by_cat": {}}
    for c, ds in sorted(by_cat.items()):
        a = np.concatenate(ds)
        res["nn_dist_median_by_cat"][c] = round(float(np.median(a)), 4)
        res["nn_dist_p90_by_cat"][c] = round(float(np.percentile(a, 90)), 4)
    h = np.concatenate([np.concatenate(by_cat[c]) for c in by_cat if c in HELD])
    s = np.concatenate([np.concatenate(by_cat[c]) for c in by_cat if c not in HELD])
    res["held_over_seen_median"] = round(float(np.median(h) / np.median(s)), 3)
    res["held_over_seen_p90"] = round(float(np.percentile(h, 90) / np.percentile(s, 90)), 3)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    (ROOT / "results/diag_snv_a6/ood_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
