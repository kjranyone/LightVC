"""N0 の事後の切り分け(合否の線ではない): ID_nv の落ちは、Irodori が音素の無い発声で声を保っていないのか、ECAPA が普通の発話の登録と音素の無い発声を結べないのか。
女 24 声の top-1 を 3 通りの登録で比べる: normal = A0 の登録 3 本(ID_nv と同じ)・same_cat = 同じ種類の別の 1 本・other_nv = 他の 6 種の音素の無い発声 12 本の連結。
same_cat / other_nv が高く normal が低ければ、声は保たれていて物差しが跨げない(登録を音素の無い発声で持てば読める)。全部低ければ生成が声を保っていない。
    cd training && uv run --no-sync python snv_diag/n0_id_within.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, load, to16
from snv_a0b_measure import proj
from snv_n0_measure import CATS, trim

N0 = ROOT / "results/snv_n0"
A0 = ROOT / "results/snv_a0"


def main() -> int:
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def E(y: np.ndarray, sr: int) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(to16(y, sr)).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    fem = [f"pf{k:03d}" for k in range(24)]
    wav = {}
    for v in fem:
        for c in CATS:
            for k in (0, 1):
                y, sr = load(N0 / f"wav/{v}_{c}_{k}.wav")
                wav[(v, c, k)] = (trim(y), sr)
    emb = {key: E(*w) for key, w in wav.items()}
    normal = {}
    for v in fem:
        ys = [load(A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)]
        normal[v] = E(np.concatenate([y for y, _ in ys]), ys[0][1])
    other = {}
    for v in fem:
        for c in CATS:
            ys = [wav[(v, c2, k)] for c2 in CATS if c2 != c for k in (0, 1)]
            other[(v, c)] = E(np.concatenate([y for y, _ in ys]), ys[0][1])
    res = defaultdict(lambda: defaultdict(list))
    for (v, c, k), e in emb.items():
        Mn = np.stack([normal[u] for u in fem])
        Ms = np.stack([emb[(u, c, 1 - k)] for u in fem])
        Mo = np.stack([other[(u, c)] for u in fem])
        for name, M in (("normal", Mn), ("same_cat", Ms), ("other_nv", Mo)):
            res[name][c].append(fem[int(np.argmax(M @ e))] == v)
    out = {name: {c: round(float(np.mean(b)), 3) for c, b in d.items()} | {"all": round(float(np.mean([x for b in d.values() for x in b])), 3)} for name, d in res.items()}
    for name, d in out.items():
        print(name, d, flush=True)
    (N0 / "n0_id_within.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
