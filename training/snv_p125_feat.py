from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch

import snv_f0v4 as V4
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent
SRC = {"": ROOT / "data/snv_p_feat", "x16": ROOT / "data/snv_p_feat_x16"}
ST = {"": ROOT / "data/snv_p_feat_st", "x16": ROOT / "data/snv_p_feat_st_x16"}
OUT = {"": ROOT / "data/snv_p_feat_v4", "x16": ROOT / "data/snv_p_feat_x16_v4"}
OUT_ST = {"": ROOT / "data/snv_p_feat_st_v4", "x16": ROOT / "data/snv_p_feat_st_x16_v4"}
CONST = {"WIN_CAP_MS": 7.5, "W_YIN": 600, "TAU_UV_HZ": 500.0, "WIN_K": 3.0, "CT_MODE": ""}


def save(dst: Path, d: dict) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: np.asarray(v, np.float32) for k, v in d.items()})
    tmp.rename(dst)


def analyze_v4(x: np.ndarray) -> dict:
    xt = torch.from_numpy(x)
    fn, fv, bn = V4.f0_dual_track(xt)
    f_ep = torch.where((fv <= 0) & (fn > 0), fn, fv)
    env = P.envelope(x, f_ep.numpy().astype(np.float64))
    per = P.periodicity(xt, f_ep).numpy()
    return {"f0": fn.numpy().astype(np.float32), "ap": bn.numpy().astype(np.float32), "env": env.astype(np.float32), "per": per.astype(np.float32)}


def work(job: tuple) -> str:
    src, wav, dst, st, dst_st = (Path(q) if q else None for q in job)
    torch.set_num_threads(1)
    if dst.exists() and (st is None or dst_st.exists()):
        return ""
    n = len(np.load(src)["f0"])
    x = librosa.load(str(wav), sr=P.SR, mono=True)[0].astype(np.float32)
    f = analyze_v4(x[: P.n_frames(len(x)) * P.H])
    if f["env"].shape[1] != n:
        return f"{src}: フレーム数 {f['env'].shape[1]} != {n}"
    save(dst, f)
    if st is not None:
        save(dst_st, dict(np.load(st)) | {k: f[k] for k in ("f0", "per", "ap")})
    return ""


def jobs_of(kind: str) -> list[tuple]:
    out = []
    g = "x16" if kind == "x16" else ""
    ps = sorted(SRC[g].glob("e1*/*/*.npz")) if g else sorted((SRC[g] / kind).rglob("*.npz"))
    for p in ps:
        rel = p.relative_to(SRC[g])
        w = ROOT / "data/snv_p_wav_x16" / rel.with_suffix(".wav") if g else EV.wav_path(str(rel.with_suffix("")))
        if w is None or not Path(w).exists():
            out.append((str(p), "", "", "", ""))
            continue
        st = ST[g] / rel
        out.append((str(p), str(w), str(OUT[g] / rel), str(st) if st.exists() else "", str(OUT_ST[g] / rel) if st.exists() else ""))
        if not g and Path(w).parent == p.parent:
            q = OUT[g] / rel.with_suffix(".wav")
            q.parent.mkdir(parents=True, exist_ok=True)
            if not q.exists() and not q.is_symlink():
                os.symlink(Path(w).resolve(), q)
    return out


def main() -> int:
    for k, v in CONST.items():
        assert getattr(P, k) == v, (k, getattr(P, k), v)
    assert EV.FEAT == SRC[""], EV.FEAT
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    nw, limit = int(opt.get("workers", 16)), int(opt.get("limit", 0))
    bad: list[str] = []
    for k in opt.get("only", "e1,e1nv,d,a0,cv,real,x16").split(","):
        js = jobs_of(k)
        miss = [j for j in js if not j[1]]
        js = [j for j in js if j[1]]
        js = js[:: max(1, len(js) // limit)][:limit] if limit else js
        print(k, "作る", len(js), "音が無い", len(miss), "固定の窓の版", sum(1 for j in js if j[3]), flush=True)
        with ProcessPoolExecutor(nw) as ex:
            bad += [r for r in ex.map(work, js, chunksize=8) if r]
        print(k, "done", "フレーム数の不一致", len(bad), flush=True)
    for q in bad[:20]:
        print(q)
    man = {"stage": "P125", "what": "前段 v4(f0_net + f_ep)で読み直した特徴(training/snv_p125_feat.py)", "const": CONST | {"F0_MODE": "v4_dual"}, "front_default": "data/snv_p_feat/front.json"}
    for g in ("", "x16"):
        for root in (OUT[g], OUT_ST[g]):
            if root.exists():
                (root / "front_v4.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    xm = SRC["x16"] / "manifest.json"
    if OUT["x16"].exists() and xm.exists():
        (OUT["x16"] / "manifest.json").write_text(xm.read_text())
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
