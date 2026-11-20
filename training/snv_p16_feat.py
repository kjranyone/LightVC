"""段 P16(paper_snv rev44 §5 M29): 擬似の操作者の版の特徴を作る。E1 と E1-NV の全ての発話を、声ごとに決めた (r, α, 部屋) で
WORLD の分析合成(高さ r 倍・声道 α 倍)→ 録音の鎖 v2(snv_d5_chain)に通し、前段 v3 で分析して data/snv_p_feat_x16/{e1,e1nv}/<声>/<発話>.npz に置く。
版の高さの目標 = 対数一様 [88, 170] Hz(開発の組の台帳の 5〜95 % = 95.5〜162 Hz を覆う)・r = 目標 / その声の高さの中央値([0.4, 1] に切る)・
α = min(1.3, 1 + 0.25 u log2(1/r))(u は一様 [0.4, 1.2])・部屋 = snv_d5_chain.room("p16|<声>", 0)。雑音の実現は発話ごと。
加工は入力側だけに使う(当てる先は data/snv_p_feat の加工しない特徴)。台帳 = data/snv_p_feat_x16/manifest.json。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_p16_feat.py [selftest]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

import snv_d5_chain as CH
import snv_d5_world as W
import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_p_feat"
OUT = ROOT / "data/snv_p_feat_x16"
SRC = {"e1": ROOT / "data/snv_e1", "e1nv": ROOT / "data/snv_e1nv"}


def voice_f0(voice: str) -> float:
    f = np.concatenate([np.load(q)["f0"] for q in sorted((FEAT / "e1" / voice).glob("t*.npz"))])
    lf = np.log(f[f > 0])
    return float(np.exp(np.median(lf[np.abs(lf - np.median(lf)) < 0.7])))


def params(voice: str, f0: float) -> dict:
    g = np.random.default_rng(CH.seed_of(f"p16|{voice}"))
    r = float(np.clip(np.exp(g.uniform(np.log(88.0), np.log(170.0))) / f0, 0.4, 1.0))
    u = float(g.uniform(0.4, 1.2))
    return {"f0": f0, "r": r, "alpha": float(min(1.3, 1.0 + 0.25 * u * np.log2(1.0 / r))), "room": CH.room(f"p16|{voice}", 0)}


def work(job: tuple) -> str:
    import librosa
    kind, voice, name, p = job
    torch.set_num_threads(1)
    dst = OUT / kind / voice / f"{name}.npz"
    if dst.exists():
        return ""
    n0 = len(np.load(FEAT / kind / voice / f"{name}.npz")["f0"])
    x = librosa.load(str(SRC[kind] / voice / f"{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
    x = W.shift(x, P.SR, p["r"], p["alpha"])
    x = CH.apply(x, p["room"], np.random.default_rng(CH.seed_of(f"p16|{voice}|{kind}|{name}")))
    f = P.analyze(x[: P.n_frames(len(x)) * P.H])
    if len(f["f0"]) != n0:
        return f"{kind}/{voice}/{name}: フレーム数 {len(f['f0'])} != {n0}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: v.astype(np.float32) for k, v in f.items()})
    tmp.rename(dst)
    return ""


def selftest() -> None:
    ps = [params(f"v{i}", 250.0 if i % 5 else 160.0) for i in range(2000)]
    r, a, t = np.array([p["r"] for p in ps]), np.array([p["alpha"] for p in ps]), np.array([p["r"] * p["f0"] for p in ps])
    assert 0.4 <= r.min() and r.max() <= 1.0 and 1.0 <= a.min() and a.max() <= 1.3
    assert 88 <= t.min() and t.max() <= 170.5 and abs(np.median(t) - (88 * 170) ** 0.5) < 4, (t.min(), t.max(), np.median(t))
    assert params("v0", 200.0) == params("v0", 200.0) and params("v0", 200.0)["room"] != params("v1", 200.0)["room"]
    print("selftest ok", "r", np.round(np.percentile(r, [5, 50, 95]), 3), "alpha", np.round(np.percentile(a, [5, 50, 95]), 3), "版の高さ", np.round(np.percentile(t, [5, 50, 95]), 1))


def main() -> int:
    if "selftest" in sys.argv[1:]:
        selftest()
        return 0
    files = [(kind, q.parent.name, q.stem) for kind in SRC for q in sorted((FEAT / kind).glob("*/*.npz"))]
    voices = sorted({v for _, v, _ in files})
    ps = {v: params(v, voice_f0(v)) for v in voices}
    jobs = [(kind, v, name, ps[v]) for kind, v, name in files]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps({"stage": "paper_snv rev44 §5 M29(段 P16)", "use": "入力側だけ(擬似の操作者の版)", "chain": CH.VER, "voices": ps}, ensure_ascii=False, indent=1))
    t = np.array([p["r"] * p["f0"] for p in ps.values()])
    print("発話", len(jobs), "声", len(voices), "版の高さの 5 / 50 / 95 %", np.round(np.percentile(t, [5, 50, 95]), 1), "r", np.round(np.percentile([p["r"] for p in ps.values()], [5, 50, 95]), 2), flush=True)
    bad = []
    with ProcessPoolExecutor(11) as ex:
        for i, e in enumerate(ex.map(work, jobs, chunksize=8)):
            if e:
                bad.append(e)
            if (i + 1) % 1000 == 0:
                print(i + 1, "/", len(jobs), "不一致", len(bad), flush=True)
    print("done", len(jobs), "不一致", len(bad), *bad[:10], sep="\n" if bad else " ", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
