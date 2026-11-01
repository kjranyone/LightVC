"""D163(学習なし): TTS パラレルの組(m{i}_{A}_t{k}_{cond}.wav)の同一性を、D118(WavLM LDA)と独立の照合器(ECAPA・WavLM-SV)で測り直す。
中心 = 目標 A の評価に使っていない文の neutral(最大 8 本)の埋め込みの平均(D118 と同じ文)・中心の平均を引いて cos で 24 目標中の順位。
本物の較正 REAL_A = 目標の評価の文の neutral の実音声。複数のディレクトリの条件を 1 回の中心で並べ、項目ごとの順位を保存する。
    uv run python d163_tts_triangulate.py --pairs <d157>:CONV,XCD <d162>:XC0S --tag d163
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="<dir>:<cond>,<cond>...")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    import idloss as ID
    from d6_unit_cov import load48
    dev = "cuda"
    jobs = [(Path(p.split(":")[0]), c) for p in a.pairs for c in p.split(":")[1].split(",")]
    meta: dict = {}
    for d, c in jobs:
        for f in d.glob(f"m*_*_t*_{c}.wav"):
            m = re.match(r"m(\d+)_([0-9a-f]+)_t(\d+)_", f.name)
            meta.setdefault(m.group(2), set()).add(int(m.group(3)))
    tg = sorted(meta)
    vers = {"ECAPA": ID.Ecapa(dev), "WAVLM_SV": ID.WavlmSV(dev)}
    nz = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    rep: dict = {"n_targets": len(tg), "chance": round(1 / len(tg), 3)}
    with torch.no_grad():
        for vn, ver in vers.items():
            emb = lambda p: ver(torch.from_numpy(load48(p)[: 8 * 48000].astype(np.float32))[None].to(dev))[0].cpu().numpy()
            cen = []
            for s in tg:
                ws = [w for w in sorted((FEM / s).glob("t*_neutral.wav")) if int(re.match(r"t(\d+)", w.name).group(1)) not in meta[s]][:8]
                cen.append(nz(np.stack([emb(w) for w in ws])).mean(0))
            cen = np.stack(cen)
            mc = cen.mean(0, keepdims=True)
            C = nz(cen - mc)

            def rank(p: Path, ti: int) -> int:
                se = C @ nz(emb(p)[None] - mc)[0]
                return int((se > se[ti]).sum()) + 1

            r = {}
            rk = [rank(FEM / s / f"t{k:02d}_neutral.wav", ti) for ti, s in enumerate(tg) for k in sorted(meta[s])]
            r["REAL_A"] = {"top1": round(float(np.mean(np.array(rk) == 1)), 3), "mean_rank": round(float(np.mean(rk)), 2), "n": len(rk)}
            for d, c in jobs:
                items, rk = [], []
                for f in sorted(d.glob(f"m*_*_t*_{c}.wav")):
                    m = re.match(r"m(\d+)_([0-9a-f]+)_t(\d+)_", f.name)
                    rk.append(rank(f, tg.index(m.group(2))))
                    items.append(f.name[: -len(f"_{c}.wav")])
                rk = np.array(rk)
                r[c] = {"top1": round(float((rk == 1).mean()), 3), "mean_rank": round(float(rk.mean()), 2), "n": int(len(rk)), "items": items, "ranks": [int(z) for z in rk]}
            rep[vn] = r
            print(vn, {c: (v["top1"], v["mean_rank"]) for c, v in r.items()}, flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
