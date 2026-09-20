"""V2-0a 補助: F-F 同一テキスト DTW cost baseline（クロスジェンダーでない参照分布）。

    HF_HUB_OFFLINE=1 uv run python v20a_ff_baseline.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from v20a_dtw import dtw_path
from extract_content import load_contentvec, content_of


def main() -> int:
    import librosa
    model = load_contentvec()
    root = Path("/tmp/opencode/vctk_probe/VCTK-Corpus/VCTK-Corpus/wav48")
    genders = {}
    for line in open(root.parent / "speaker-info.txt"):
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            genders["p" + p[0]] = p[2]
    by_uid: dict[str, list[str]] = {}
    for f in sorted(root.rglob("*.wav")):
        sid, uid = f.stem.split("_")
        by_uid.setdefault(uid, []).append(sid)
    costs, ratios = [], []
    for uid, spks in by_uid.items():
        F = [s for s in spks if genders.get(s) == "F"]
        if len(F) < 2:
            continue
        feats = {}
        for s in F:
            w, _ = librosa.load(root / s / f"{s}_{uid}.wav", sr=16000, mono=True)
            feats[s] = content_of(model, w).float().numpy()
        for a, b in itertools.combinations(F, 2):
            ca, cb = feats[a], feats[b]
            ca = ca / (np.linalg.norm(ca, axis=-1, keepdims=True) + 1e-6)
            cb = cb / (np.linalg.norm(cb, axis=-1, keepdims=True) + 1e-6)
            path, c = dtw_path(1.0 - ca @ cb.T)
            costs.append(c)
            ratios.append(len(path) / max(ca.shape[0], cb.shape[0]))
    co = np.array(costs)
    print(f"F-F pairs: {len(co)}")
    print(f"cost p50/p90/p99: {np.median(co):.4f} / {np.quantile(co,0.9):.4f} / {np.quantile(co,0.99):.4f}")
    print(f"len_ratio p50/p99: {np.median(ratios):.3f} / {np.quantile(ratios,0.99):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
