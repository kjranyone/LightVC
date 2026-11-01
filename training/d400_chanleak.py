"""D400(results/conv_p0/d400_prereg.yaml): 入力の帯域(録音の経路)が事後を通って出力に漏れるか。devff(実女声 → 女声・目標ごとに 1 本 = 30 本)の入力を 8 kHz で低域通過して Rust で書き出し、
条件(音量・高さ・有声)と事後を 元 / 低域通過 から組み合わせて候補 B で描画し、出力の帯域の相対 dB(D399 と同じ)を ORIG と対で比べる。
    uv run python d400_chanleak.py --ckpt ../results/diag_ppgvoc27/product_002500_m0.pt --out ../results/conv_p0/d400
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d399_tablefollow import BANDS, prof

CONDS = ("ORIG", "LP_ALL", "LP_POST", "LP_COND", "LP_C0", "LP_FV")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff")
    ap.add_argument("--fc", type=float, default=8000.0)
    a = ap.parse_args()
    dev = "cuda"
    S, O = P.H / a.set, Path(a.out)
    W = P.H / "d400" / a.set
    W.mkdir(parents=True, exist_ok=True)
    for c in CONDS:
        (O / c).mkdir(parents=True, exist_ok=True)
    ed = P.H / "e2e" / f"eval_{a.set}"
    seen, items = set(), []
    for f in sorted(ed.glob("p0*.cond")):
        ti = int(f.stem.split("_t")[1][:3])
        if ti not in seen:
            seen.add(ti)
            items.append((f.stem, int(f.stem[1:5]), ti))
    sos = butter(10, a.fc, "low", fs=48000, output="sos")
    jobs = []
    for k, pi, ti in items:
        x, sr = sf.read(S / "d56" / f"{k}_SRC.wav", dtype="float32")
        lp = W / f"{k}_LP.wav"
        if not lp.exists():
            sf.write(lp, sosfiltfilt(sos, x).astype(np.float32), sr)
        jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{ti:03d}", "--operator-enroll", S / "d85_work" / f"op_{pi:04d}", "--med-from-operator"] + COM + ["-i", lp], W / f"{k}.cond", W / f"{k}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(job, jobs))
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    rows = []
    for k, pi, ti in items:
        c0, l0 = P.load_dump(ed, k)
        c1, l1 = P.load_dump(W, k)
        n = min(len(c0), len(c1))
        spk = P.load_enroll(S / "d85_work" / f"en_{ti:03d}")
        row = {"k": k, "ti": ti}
        cc0 = c0[:n].copy()
        cc0[:, 0] = c1[:n, 0]
        cfv = c0[:n].copy()
        cfv[:, 25:27] = c1[:n, 25:27]
        vv0, vv1 = c0[:n, 26] > 0.5, c1[:n, 26] > 0.5
        row["voiced_frac"] = [float(vv0.mean()), float(vv1.mean())]
        row["c0n_diff_unvoiced"] = float((c1[:n, 0] - c0[:n, 0])[~vv0].mean()) if (~vv0).any() else 0.0
        row["c0n_diff_voiced"] = float((c1[:n, 0] - c0[:n, 0])[vv0].mean()) if vv0.any() else 0.0
        for name, (c, l) in zip(CONDS, ((c0, l0), (c1, l1), (c0, l1), (c1, l0), (cc0, l0), (cfv, l0))):
            y = P.render(g, c[:n], l[:n], spk, dev)
            sf.write(O / name / f"{k}_{name}.wav", y, N.SR)
            row[name] = prof(O / name / f"{k}_{name}.wav").tolist()
        row["SRC"] = prof(S / "d56" / f"{k}_SRC.wav").tolist()
        row["SRC_LP"] = prof(W / f"{k}_LP.wav").tolist()
        rows.append(row)
    rng = np.random.default_rng(400)
    rep: dict = {"n": len(rows), "bands": [f"{lo}-{hi}" for lo, hi in BANDS]}
    for name in CONDS[1:] + ("SRC_LP",):
        base = "SRC" if name == "SRC_LP" else "ORIG"
        d = np.array([np.array(r[name]) - np.array(r[base]) for r in rows])
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean(0) for _ in range(2000)])
        rep[f"{name}-{base}"] = {b: {"diff_db": round(float(d[:, i].mean()), 2), "ci95": [round(float(np.percentile(bs[:, i], 2.5)), 2), round(float(np.percentile(bs[:, i], 97.5)), 2)]} for i, b in enumerate(rep["bands"])}
        print(name, "−", base, json.dumps(rep[f"{name}-{base}"]), flush=True)
    rep["voiced_frac_orig_lp"] = [round(float(np.mean([r["voiced_frac"][i] for r in rows])), 4) for i in (0, 1)]
    rep["c0n_diff_lp_minus_orig"] = {"unvoiced": round(float(np.mean([r["c0n_diff_unvoiced"] for r in rows])), 4), "voiced": round(float(np.mean([r["c0n_diff_voiced"] for r in rows])), 4)}
    print("有声の割合 元 / 低域通過", rep["voiced_frac_orig_lp"], "c0n の差(低域通過 − 元・c0n の単位 = 10 dB 相当の 1/10 ではなく c0/10)", rep["c0n_diff_lp_minus_orig"], flush=True)
    (O / "summary.json").write_text(json.dumps({"rep": rep, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
