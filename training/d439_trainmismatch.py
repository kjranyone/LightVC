"""D439: ガビの線(フレーム周期 200 Hz)は学習と推論の条件の作り方の違いから来るか。
学習の出力では fline ≈ −0.2 dB(diag_ppgvoc43_smoke)・Rust の書き出しから描画した評価では 2.4 dB(D2 の SR)。
学習の条件 = harvest + stonemask の f0(非因果)・CheapTrick の中心の窓の c0(非因果)・推論 = Rust の因果な F1 の f0・--causal-ct の c0。
devff(実女声 → 女声・目標ごとに 1 本)の Rust の書き出しの条件のうち、f0 / 有声・c0n を学習の作り方(入力の音声から Python で)に替えて描画し、fline / amline を比べる。
    uv run python d439_trainmismatch.py --ckpt ../results/diag_ppgvoc41/product_002500_m0.pt --out ../results/conv_p0/d439
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import pae as PA
import train_ppgvoc as P
from d436_buzzcarrier import items_of
from eval_nvoc import amline
from fsync import fline

VARS = ("ORIG", "HF0", "PYC0", "BOTH")


def harvest48(x: np.ndarray) -> np.ndarray:
    import librosa
    import pyworld
    x16 = librosa.resample(x.astype(np.float64), orig_sr=N.SR, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=N.HOP / N.SR * 1000)
    return pyworld.stonemask(x16, f0, t, 16000).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    for v in VARS:
        (O / v).mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    ed = P.H / "e2e" / "eval_devff"
    res = {v: {"fline": [], "amline": []} for v in VARS}
    diag = {"c0n_jitter_rust": [], "c0n_jitter_py": [], "c0n_mean_diff_py_minus_rust": [], "lf_jitter_rust": [], "lf_jitter_harvest": [], "vv_agree": []}
    for k, ti in items_of("devff")[: a.n]:
        c, lg = P.load_dump(ed, k)
        n = len(c)
        x, sr = sf.read(P.H / "devff" / "d56" / f"{k}_SRC.wav", dtype="float32")
        hr = harvest48(x)
        hr = np.pad(hr, (0, max(0, n + 1 - len(hr))))[: n + 1]
        xa = np.pad(x, (PA.A, PA.A + max(0, (n + 1) * N.HOP - len(x))))
        env = PA.envelope(xa, hr)
        c0p = ((env[0] - PA.C0_SIL) / 10)[1:n + 1]
        hv = hr[1:n + 1]
        v_r = c[:, 26] > 0.5
        v_h = hv > 0
        both = v_r[1:] & v_r[:-1]
        diag["c0n_jitter_rust"].append(float(np.abs(np.diff(c[:, 0])).mean()))
        diag["c0n_jitter_py"].append(float(np.abs(np.diff(c0p)).mean()))
        diag["c0n_mean_diff_py_minus_rust"].append(float((c0p - c[:, 0])[v_r].mean()) if v_r.any() else 0.0)
        diag["lf_jitter_rust"].append(float(np.abs(np.diff(c[:, 25]))[both].mean()) if both.any() else 0.0)
        hb = v_h[1:] & v_h[:-1]
        lfh = np.where(v_h, np.log(np.maximum(hv, 1.0) / 200.0), 0.0)
        diag["lf_jitter_harvest"].append(float(np.abs(np.diff(lfh))[hb].mean()) if hb.any() else 0.0)
        diag["vv_agree"].append(float((v_r == v_h).mean()))
        spk = P.load_enroll(P.H / "devff" / "d85_work" / f"en_{ti:03d}")
        for v in VARS:
            cc = c.copy()
            if v in ("HF0", "BOTH"):
                cc[:, 25] = lfh
                cc[:, 26] = v_h.astype(np.float32)
            if v in ("PYC0", "BOTH"):
                cc[:, 0] = c0p
            y = P.render(g, cc, lg, spk, dev)
            sf.write(O / v / f"{k}_{v}.wav", y, N.SR)
            yy = y[: 8 * N.SR]
            res[v]["fline"].append(float(fline(torch.from_numpy(yy)[None])[0]))
            res[v]["amline"].append(amline(yy))
    rep = {"n": len(res["ORIG"]["fline"]), "median": {v: {m: round(float(np.median(x)), 3) for m, x in d.items()} for v, d in res.items()},
           "diag_median": {k: round(float(np.median(v)), 4) for k, v in diag.items()}}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (O / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
