"""段 P90(results/snv_p90/prereg.yaml): 学習の発話の「周期性を崩した版」の特徴を作る(合成の声は倍音の間のエネルギーが本物の男声より約 10 dB 少ない = 周期性がきれいすぎる)。
元の音を WORLD で分け(f0 = 前段 v3 の値・包絡 = CheapTrick・非周期性 = D4C)、発話ごとに無作為に
(1) 有声のフレームの率 p ~ U(0, PSUB) で f0 を半分に(周期の倍化・半々でフレームごとに独立 / 平均 60 ms の区間)・(2) 有声のフレームの非周期性に a ~ U(0, AP) を足す・(3) 描き直した音に白色雑音を SNR ~ U(SNR_LO, SNR_HI) dB で足す。
前段 v3 で読み直して data/snv_p_feat_xs/{e1,e1nv}/<声>/<名>_k{0,1}.npz に置く(2 版は別の乱数)。符号の学習の入力側の視点にだけ使う(当てる先にしない)。
    cd training && uv run --no-sync python snv_p90_impure.py [workers=14] [limit=0] [stats=1]
"""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p_feat_xs"
PSUB, AP, SNR_LO, SNR_HI = 0.4, 0.35, 15.0, 45.0


def impure(x: np.ndarray, f0: np.ndarray, g: np.random.Generator) -> np.ndarray:
    import pyworld as pw
    n = min(len(f0), len(x) // P.H)
    f0 = f0[:n].astype(np.float64)
    t = np.arange(n) * (P.H / P.SR)
    sp = pw.cheaptrick(x, f0, t, P.SR)
    ap = pw.d4c(x, f0, t, P.SR)
    v = f0 > 0
    p = g.uniform(0, PSUB)
    if g.random() < 0.5:
        st = g.random(n) < p
    else:
        st, on = np.zeros(n, bool), False
        for i in range(n):
            on = (g.random() > 1 / 12) if on else (g.random() < p / (12 * max(1 - p, 1e-3)))
            st[i] = on
    f = f0.copy()
    f[v & st] /= 2
    a = ap.copy()
    a[v] = np.clip(a[v] + g.uniform(0, AP), 0, 1)
    import pyworld as pw2
    y = pw2.synthesize(f, sp, a, P.SR, 1000 * P.H / P.SR)[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    return y + g.standard_normal(len(y)) * np.sqrt((y ** 2).mean() / 10 ** (g.uniform(SNR_LO, SNR_HI) / 10))


def work(rel: str) -> str:
    import torch
    torch.set_num_threads(1)
    r = Path(rel)
    wav = ROOT / ("data/snv_e1" if r.parts[0] == "e1" else "data/snv_e1nv") / r.parts[1] / (r.stem + ".wav")
    x = librosa.load(str(wav), sr=P.SR, mono=True)[0].astype(np.float64)
    f0 = np.load(T.FEAT / r)["f0"]
    for k in (0, 1):
        dst = OUT / r.parent / f"{r.stem}_k{k}.npz"
        if dst.exists():
            continue
        g = np.random.default_rng(zlib.crc32(f"{rel}|{k}".encode()))
        f = P.analyze(impure(x, f0, g).astype(np.float32))
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **{q: np.asarray(z, np.float32) for q, z in f.items()})
    return rel


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    nw, limit = int(opt.get("workers", 14)), int(opt.get("limit", 0))
    tr = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["train"]
    rels = [str(p.relative_to(T.FEAT)) for v in tr for p in sorted((T.FEAT / "e1" / v).glob("t*.npz")) + sorted((T.FEAT / "e1nv" / v).glob("*.npz"))]
    rels = rels[:: max(1, len(rels) // limit)][:limit] if limit else rels
    print("発話", len(rels), flush=True)
    with ProcessPoolExecutor(nw) as ex:
        for i, _ in enumerate(ex.map(work, rels, chunksize=4)):
            if i % 1000 == 0:
                print(i, flush=True)
    n = sum(1 for _ in OUT.rglob("*.npz"))
    (OUT / "manifest.json").write_text(json.dumps({"stage": "P90", "what": "周期性を崩した版(WORLD の描き直し: 周期の倍化・非周期性・白色雑音)の特徴・k0 / k1 は別の乱数", "params": {"PSUB": PSUB, "AP": AP, "SNR": [SNR_LO, SNR_HI]}, "voices": len(tr), "files": n}, ensure_ascii=False, indent=1))
    print("files", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
