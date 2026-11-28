"""段 P102(results/snv_p102/prereg.yaml): 源の側だけ半フレーム(120 標本 = 2.5 ms)遅らせた特徴を、前段 v3.1(SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500)で作る(入力側の加工・当てる先には使わない)。
出力 = data/snv_p_feat_v31h/{e1,e1nv}・data/snv_p_feat_x16_v31h/{e1,e1nv}(音は data/snv_p_wav_x16)と、固定の窓の包絡の版 data/snv_p_feat_st_v31h・data/snv_p_feat_st_x16_v31h(env = 遅らせた音の固定の窓の包絡)。
    cd training && SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 uv run --no-sync python snv_p102_feat.py [workers=16] [only=e1,e1nv,x16] [limit=0]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent
DLY = 120
WIN_MS = 40.0
SRC = {"": ROOT / "data/snv_p_feat", "x16": ROOT / "data/snv_p_feat_x16"}
OUT = {"": ROOT / "data/snv_p_feat_v31h", "x16": ROOT / "data/snv_p_feat_x16_v31h"}
OUT_ST = {"": ROOT / "data/snv_p_feat_st_v31h", "x16": ROOT / "data/snv_p_feat_st_x16_v31h"}
CONST = {"WIN_CAP_MS": 7.5, "W_YIN": 600, "TAU_UV_HZ": 500.0, "WIN_K": 3.0, "CT_MODE": ""}


def save(dst: Path, d: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v, np.float32) for k, v in d.items()})
    tmp.rename(dst)


def work(job: tuple) -> str:
    src, wav, dst, dst_st = (Path(q) for q in job)
    torch.set_num_threads(1)
    if dst.exists() and dst_st.exists():
        return ""
    n = len(np.load(src)["f0"])
    x = librosa.load(str(wav), sr=P.SR, mono=True)[0].astype(np.float32)
    x = np.concatenate([np.zeros(DLY, np.float32), x])[: len(x)]
    f = P.analyze(x[: P.n_frames(len(x)) * P.H])
    if f["env"].shape[1] != n:
        return f"{src}: フレーム数 {f['env'].shape[1]} != {n}"
    save(dst, f)
    save(dst_st, dict(f) | {"env": P.fixed_env(x, n, WIN_MS)})
    return ""


def jobs_of(kind: str) -> list[tuple]:
    out = []
    g = "x16" if kind == "x16" else ""
    ps = sorted(SRC[g].glob("e1*/*/*.npz")) if g else sorted((SRC[g] / kind).rglob("*.npz"))
    for p in ps:
        rel = p.relative_to(SRC[g])
        w = ROOT / "data/snv_p_wav_x16" / rel.with_suffix(".wav") if g else EV.wav_path(str(rel.with_suffix("")))
        assert w is not None and Path(w).exists(), p
        out.append((str(p), str(w), str(OUT[g] / rel), str(OUT_ST[g] / rel)))
    return out


def main() -> int:
    for k, v in CONST.items():
        assert getattr(P, k) == v, (k, getattr(P, k), v)
    assert EV.FEAT == SRC[""], EV.FEAT
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    nw, limit = int(opt.get("workers", 16)), int(opt.get("limit", 0))
    bad: list[str] = []
    for k in opt.get("only", "e1,e1nv,x16").split(","):
        js = jobs_of(k)
        js = js[:: max(1, len(js) // limit)][:limit] if limit else js
        print(k, "作る", len(js), flush=True)
        with ProcessPoolExecutor(nw) as ex:
            bad += [r for r in ex.map(work, js, chunksize=8) if r]
        print(k, "done", "フレーム数の不一致", len(bad), flush=True)
    for q in bad[:20]:
        print(q)
    man = {"stage": "P102", "what": "源の側だけ半フレーム遅らせ、前段 v3.1 で読んだ特徴(training/snv_p102_feat.py)", "delay_samples": DLY, "const": CONST}
    for g in ("", "x16"):
        for root in (OUT[g], OUT_ST[g]):
            if root.exists():
                (root / "front_v31h.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    xm = SRC["x16"] / "manifest.json"
    if OUT["x16"].exists() and xm.exists():
        (OUT["x16"] / "manifest.json").write_text(xm.read_text())
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
