"""N0 の物差しの切り分け(合否ではない・実録音は物差しの較正だけに使う): ECAPA + LDA は、本物の女声の普通の発話の登録から、同じ人の間投詞(はぁ・あっ・んっ・ふぅ・ひっく・きゃ で始まる 16 字以下)を当てられるか。
female-dataset で間投詞を持つ話者は 21 人(63 本)だけ。普通の発話 4 本 + 間投詞 1 本以上の話者を検査に、普通の発話 3 本以上の話者 200 人を混ぜ物に、本人 + 混ぜ物 23 人の 24 人で 20 回引いて top-1(偶然 0.042)。
登録 = 普通の発話 3 本の連結・検査 = 間投詞 / 普通の発話の 4 本目。本物で間投詞が当たるなら、N0 の落ちは Irodori が音素の無い発声で声を保っていない側。
    cd training && uv run --no-sync python snv_diag/n0_real_nv_id.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, to16
from snv_a0b_measure import proj
from snv_n0_measure import trim

PAT = re.compile(r"^[（(]?(はぁ|はあ|あっ|んっ|ふぅ|ひっく|きゃ)")


def main() -> int:
    by = defaultdict(lambda: {"normal": [], "nv": []})
    for lab in sorted((ROOT / "female-dataset").glob("*/*.lab")):
        w = lab.with_suffix(".wav")
        if not w.exists():
            continue
        t = lab.read_text(encoding="utf-8").strip()
        if len(t) <= 16 and PAT.search(t):
            by[lab.parent.name]["nv"].append(w)
        elif len(t) >= 20:
            by[lab.parent.name]["normal"].append(w)
    rng = random.Random(0)
    tst = sorted(s for s, d in by.items() if len(d["normal"]) >= 4 and len(d["nv"]) >= 1)
    pool = rng.sample(sorted(s for s, d in by.items() if len(d["normal"]) >= 3 and s not in tst), 200)
    print("test speakers", len(tst), "distractors", len(pool), flush=True)
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def rd(p: Path) -> tuple:
        y, sr = sf.read(str(p))
        return (y.mean(1) if y.ndim > 1 else y).astype(np.float32), sr

    def E(y: np.ndarray, sr: int) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(to16(y, sr)).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    enr, nv, nt, dur = {}, {}, {}, defaultdict(list)
    for s in tst + pool:
        ns = sorted(by[s]["normal"])[:4]
        ys = [rd(p) for p in ns[:3]]
        enr[s] = E(np.concatenate([y for y, _ in ys]), ys[0][1])
        if s not in tst:
            continue
        nt[s] = E(*rd(ns[3]))
        nv[s] = []
        for p in sorted(by[s]["nv"])[:4]:
            y, sr = rd(p)
            y = trim(y)
            if len(y) / sr >= 0.25:
                nv[s].append(E(y, sr))
                dur["nv"].append(len(y) / sr)
    res = defaultdict(list)
    for _ in range(20):
        for s in tst:
            pick = [s] + rng.sample(pool, 23)
            M = np.stack([enr[u] for u in pick])
            res["normal_test"].append(pick[int(np.argmax(M @ nt[s]))] == s)
            for e in nv[s]:
                res["interjection"].append(pick[int(np.argmax(M @ e))] == s)
    out = {k: (round(float(np.mean(v)), 3), len(v)) for k, v in res.items()} | {"nv_dur_median": round(float(np.median(dur["nv"])), 2), "test_speakers": len(tst)}
    print(out, flush=True)
    (ROOT / "results/snv_n0/n0_real_nv_id.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
