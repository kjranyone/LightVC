"""D487(診断・gate_mf_v1 の物差しを詰める): 外郎売(実男声・朗唱・mp3)の変換の遠さは話し方(長い窓)か声の質(短い窓)か。
gate_mf_run.build と同じ組(参照 22 s・8 s の区間・目標 = dev2 の 30 人を巡回)を作り、製品の設定(定常状態・操作者の揺らぎ)で描画 →
<out>/grp/{MALE_UIRO, FEMALE_FF2(D467 の devff2 OP15)・CP_FF2(D467 の devff2 CP)・SRC_UIRO(入力そのもの)}を d286_window(0.5 / 2 / 8 s)で比べる。
    uv run python d487_uiro_windows.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d487
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d6_unit_cov import load48
from gate_mf_run import build

PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    src = O / "in" / "uiro"
    src.mkdir(parents=True, exist_ok=True)
    lk = src / "uiro_full.wav"
    if not lk.exists():
        lk.symlink_to(P.H / "devreal" / "d56" / "_work" / "uiro_full.wav")
    S = P.H / "d487_uiro"
    log = build(O / "in", S, 30, "dev2")
    ed = P.H / "e2e" / "eval_d487_uiro_pre"
    ed.mkdir(parents=True, exist_ok=True)
    n0 = int(PRE * N.SR)
    jobs = []
    for r in log:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        w = S / "d56/_work" / f"pre_{k}.wav"
        if not w.exists():
            pre = load48(S / "d56/_work" / f"op_{r['pi']:04d}.wav")[-n0:]
            sf.write(w, np.concatenate([np.pad(pre, (n0 - len(pre), 0)), load48(S / "d56" / f"{k}_SRC.wav")]).astype(np.float32), N.SR)
        jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["-i", w], ed / f"{k}.cond", ed / f"{k}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(job, jobs))
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 1.5
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    G = O / "grp"
    for d in ("MALE_UIRO", "SRC_UIRO", "FEMALE_FF2", "CP_FF2"):
        (G / d).mkdir(parents=True, exist_ok=True)
    for r in log:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, lg = P.load_dump(ed, k)
        y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
        sf.write(G / "MALE_UIRO" / f"{k}_MALE.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        dst = G / "SRC_UIRO" / f"{k}_SRC.wav"
        if not dst.exists():
            dst.symlink_to(S / "d56" / f"{k}_SRC.wav")
    for nm, d in (("FEMALE_FF2", P.ROOT / "results/conv_p0/d467/devff2/OP15"), ("CP_FF2", P.ROOT / "results/conv_p0/d467/devff2/CP")):
        for f in sorted(d.glob("p*.wav")):
            dst = G / nm / f.name
            if not dst.exists():
                dst.symlink_to(f.resolve())
    r_ = subprocess.run([sys.executable, str(Path(__file__).parent / "d286_window.py"), "--eq_dir", str(G), "--perp", "--out", str(O / "window.json")], capture_output=True, text=True)
    print(r_.stdout[-1500:], r_.stderr[-500:] if r_.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
