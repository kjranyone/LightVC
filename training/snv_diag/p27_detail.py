"""段 P27 補足 4(results/snv_p27/prereg.yaml の supplement_4・学習なし・描画なし): C の出力の細部の動きは目標から来るか演奏から来るか。
E1 の検証の女声 × t05 を源に、A0F の先頭 4 声の各々へ k 0 で変換して C の出力 yo を得る(同じ源なので 4 つの出力はフレームが揃う)。帯 lo = c1〜c12・hi = c13〜c24・有声のフレーム。
(1) 出力の平均からのずれを、4 目標の平均(共有)と残り(目標に固有)に分けた分散の割合。(2) 共有の成分と源の本当の包絡のずれの相関。(3) 出力の平均の、源の登録の平均 → 目標の参照の平均への射影(0 = 源・1 = 目標)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p27_detail.py snv_p24_c snv_p8_wst [out=results/snv_p27/detail.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p27_land as P
import probe_rig as G
import snv_p1_eval as EV

ROOT = G.ROOT
BANDS = {"lo": (1, 13), "hi": (13, 25)}


def vmean(us: list[dict], lo: int, hi: int) -> np.ndarray:
    return np.concatenate([u["y"][lo:hi, u["v"] > 0] for u in us], 1).mean(1)


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a]
    out = ROOT / opt.get("out", "results/snv_p27/detail.json")
    items = [it for it in P.build(["valf"], 64) if it["src"].endswith("t05")]
    _, tg4 = G.targets()
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    gen = EV.load_gen()
    res = {"prereg": "results/snv_p27/prereg.yaml(supplement_4)", "n_src": len(items), "systems": {}}
    for tag in tags:
        s = EV.System(tag, gen)
        acc = {b: {"spec": [], "corr": [], "proj": [], "std_ratio": []} for b in BANDS}
        for it in items:
            u, enr = it["u"], it["enr"]
            ys = [s.convert(u, enr, s.structure(f"t|{v}", trefs[v]), 0, with_y=True)[1][0].cpu().numpy() for v in tg4]
            n = ys[0].shape[1]
            vm = u["v"][:n] > 0
            for b, (lo, hi) in BANDS.items():
                d = np.stack([y[lo:hi, vm] - y[lo:hi, vm].mean(1, keepdims=True) for y in ys])
                sh = d.mean(0)
                tot = float((d ** 2).mean())
                acc[b]["spec"].append(float(((d - sh[None]) ** 2).mean()) / max(tot, 1e-12))
                ds = u["y"][lo:hi, :n][:, vm] - u["y"][lo:hi, :n][:, vm].mean(1, keepdims=True)
                acc[b]["corr"].append(float(np.mean([np.corrcoef(sh[i], ds[i])[0, 1] for i in range(hi - lo)])))
                acc[b]["std_ratio"].append(float(np.mean(d.std(2).mean(0) / np.maximum(ds.std(1), 1e-6))))
                ms = vmean(enr, lo, hi)
                for y, v in zip(ys, tg4):
                    mt = vmean(trefs[v], lo, hi)
                    acc[b]["proj"].append(float((y[lo:hi, vm].mean(1) - ms) @ (mt - ms) / max(float((mt - ms) @ (mt - ms)), 1e-12)))
        res["systems"][tag] = {b: {"目標に固有の割合": round(float(np.mean(a["spec"])), 3), "共有の成分と源のずれの相関": round(float(np.mean(a["corr"])), 3), "平均の着地(0 = 源・1 = 目標)": round(float(np.mean(a["proj"])), 3),
                                   "標準偏差の比(出力 / 源の本当)": round(float(np.mean(a["std_ratio"])), 3)} for b, a in acc.items()}
        print(tag, json.dumps(res["systems"][tag], ensure_ascii=False), flush=True)
        del s
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
