"""段 P108(results/snv_p108/prereg.yaml・学習なし): 高さの追跡の物差し。組 = 合成の男声・女声(E1 の検証の声)・音素の無い発声(E1-NV の悲鳴と喘ぎと息切れ)・本物の男声(probe の 4 人)。参照 = オフラインの pYIN(results/snv_p108/ref に保存)。
    cd training && uv run --no-sync python snv_diag/p108_lab.py <名前> [track=モジュール:関数]
追跡の関数 = f(x: torch.Tensor) -> (f0, ap)。既定は snv_p0_front.f0_track(今の追跡)。
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent.parent
REF = ROOT / "results/snv_p108/ref"
RFIX = True
EAR = {"real1": [("silence_hi", 6.30, 6.62, "uv"), ("ka_burst", 3.69, 3.74, "le", 150.0), ("cons_1144", 3.90, 3.94, "uv")], "fam_em594": [("breath", 1.12, 1.72, "uv"), ("breath2", 4.38, 4.82, "uv"), ("breath3", 7.96, 8.37, "uv")]}


def sets() -> dict:
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    m, f = [v for v in vs if v.startswith("em")][:6], [v for v in vs if v.startswith("ef")][:6]
    out = {"synM": [ROOT / f"data/snv_e1/{v}/{t}.wav" for v in m for t in ("t03", "t06", "t09")], "synF": [ROOT / f"data/snv_e1/{v}/{t}.wav" for v in f for t in ("t03", "t06", "t09")]}
    out["nvHi"] = [ROOT / f"data/snv_e1nv/{v}/scream_0.wav" for v in m[:3] + f[:6]]
    out["nvBreathy"] = [ROOT / f"data/snv_e1nv/{v}/{c}_0.wav" for v in m[:3] + f[:3] for c in ("moan", "pant")]
    out["real"] = [ROOT / "results/snv_p96/probe_in" / sp / "in.f32" for sp in ("ca91ecce", "cb87e9f4", "cc5576e7", "cd094b9e")]
    return {k: [p for p in v if p.exists()] for k, v in out.items()}


def load(p: Path) -> np.ndarray:
    x = np.fromfile(p, np.float32) if p.suffix == ".f32" else librosa.load(p, sr=P.SR, mono=True)[0].astype(np.float32)
    return x[: P.n_frames(len(x)) * P.H]


def ref(p: Path, x: np.ndarray, fmax: int) -> np.ndarray:
    q = REF / (("_".join(p.parts[-3:])).replace(".", "_") + f"_fmax{fmax}.npy")
    if not q.exists():
        r, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=P.SR, target_sr=16000), fmin=60, fmax=fmax, sr=16000, frame_length=1024, hop_length=80, center=True)
        np.save(q, np.where(vf, np.nan_to_num(r), 0.0).astype(np.float32))
    return np.load(q)


def score(f: np.ndarray, r: np.ndarray) -> dict:
    n = min(len(f), len(r))
    f, r = f[:n], r[:n]
    known = r >= 0
    a, b = (f > 0) & known, r > 0
    both = a & b
    d = np.log2(f[both] / r[both]) if both.any() else np.zeros(0)
    dj = np.diff(np.log2(np.maximum(f, 1.0)))
    hi = b & (r > 500)
    return {"n": n, "both": int(both.sum()), "gross": int((np.abs(d) > 0.2).sum()), "up": int((d > 0.5).sum()), "down": int((d < -0.5).sum()), "own_only": int((a & ~b).sum()), "ref_only": int((~a & b).sum()), "v_own": int(a.sum()), "v_ref": int(b.sum()),
            "jumps": int((a[1:] & a[:-1] & (np.abs(dj) > 0.4)).sum()), "hi_ref": int(hi.sum()), "hi_ok": int((hi & a & (np.abs(np.log2(np.maximum(f, 1.0) / np.maximum(r, 1.0))) < 0.2)).sum())}


def main() -> None:
    name = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    mod, fn = opt.get("track", "snv_p0_front:f0_track").split(":")
    track = getattr(importlib.import_module(mod), fn)
    REF.mkdir(parents=True, exist_ok=True)
    res: dict = {"name": name, "track": f"{mod}:{fn}", "sets": {}, "ear": {}}
    for sn, ps in sets().items():
        acc: dict = {}
        for p in ps:
            x = load(p)
            f = track(torch.from_numpy(x))[0].numpy().astype(np.float32)
            rr = ref(p, x, 1600 if sn == "nvHi" else 700)
            if sn == "nvHi" and RFIX:
                import pyworld
                hq = REF / (("_".join(p.parts[-3:])).replace(".", "_") + "_harvest1600.npy")
                if not hq.exists():
                    np.save(hq, pyworld.harvest(x.astype(np.float64), P.SR, frame_period=5.0, f0_floor=60.0, f0_ceil=1600.0)[0].astype(np.float32))
                hh = np.load(hq)
                m = min(len(rr), len(hh))
                rr = rr[:m].copy()
                rr[(rr > 0) & ~((hh[:m] > 0) & (np.abs(np.log2(np.maximum(rr, 1.0) / np.maximum(hh[:m], 1.0))) < 0.2))] = -1.0
            if sn in ("synM", "synF") and RFIX:
                rr = np.where(rr > 500, -1.0, rr)
            for k, v in score(f, rr).items():
                acc[k] = acc.get(k, 0) + v
            key = "real1" if "ca91ecce" in str(p) else "fam_em594" if "em594/t05" in str(p) else None
            if key:
                for e in EAR[key]:
                    seg = f[int(e[1] * 200) : int(e[2] * 200)]
                    res["ear"][f"{key}:{e[0]}"] = round(float((seg > 0).mean()), 2) if e[3] == "uv" else round(float(((seg > 0) & (seg > e[4])).mean()), 2)
        b = max(1, acc["both"])
        res["sets"][sn] = {"files": len(ps), "gross": round(acc["gross"] / b, 3), "up": round(acc["up"] / b, 3), "down": round(acc["down"] / b, 3), "own_only": round(acc["own_only"] / acc["n"], 3), "ref_only": round(acc["ref_only"] / acc["n"], 3), "voiced_own": round(acc["v_own"] / acc["n"], 3), "voiced_ref": round(acc["v_ref"] / acc["n"], 3),
                           "jumps_per_s": round(acc["jumps"] / max(1, acc["v_own"]) * 200, 2), "hi_recall": round(acc["hi_ok"] / acc["hi_ref"], 3) if acc["hi_ref"] else None, "hi_frames": acc["hi_ref"]}
        print(sn, json.dumps(res["sets"][sn]), flush=True)
    if "fam_em594:breath" not in res["ear"]:
        p = ROOT / "data/snv_e1/em594/t05.wav"
        f = track(torch.from_numpy(load(p)))[0].numpy()
        for e in EAR["fam_em594"]:
            res["ear"][f"fam_em594:{e[0]}"] = round(float((f[int(e[1] * 200) : int(e[2] * 200)] > 0).mean()), 2)
    print("ear(悪い読みが残る率・0 が良い)", json.dumps(res["ear"]), flush=True)
    (ROOT / f"results/snv_p108/lab_{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
