"""D552(推論だけの診断・男声 → 女声の差 MT − FT = v1⊥ +1.5 の帰属): 男声 TTS の入力(dev の d467/<k>.wav = 前置き 12 s + 8 s)と操作者の参照(d56/_work/op_PPPP.wav)を
WORLD(harvest・CheapTrick・D4C・48 kHz)で f0 × 2.0・包絡の周波数 × 1.15(声道を短く)に変えて(時刻は保つ)、登録(artic-enroll)からやり直して Rust で書き出す(eval_dev_pre_W)。
P43OP を製品の設定(OP15・定常状態)で CPU で描画: W_ALL = 変換後の入力の書き出しを全部・W_POST = 元の書き出しに事後だけ変換後の入力のもの(時刻は同じ)・W_COND = 事後は元・条件(c0n・f0・有声・包絡)は変換後。
比べるのは同じ文・同じ目標の TTS 女声の入力の変換 FT(d550)= d467_paired_v1 で別に。DSP の変換は診断だけ(製品の経路にしない)。
    uv run python d552_mf_world_norm.py --out ../results/conv_p0/d552
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

ROOT = Path(__file__).resolve().parent.parent
PRE = 12.0
FS = 48000


def wnorm(x: np.ndarray, fr: float = 2.0, alpha: float = 1.15) -> np.ndarray:
    import pyworld as pw
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, FS, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, FS)
    ap = pw.d4c(x, f0, t, FS)
    nb = sp.shape[1]
    src = np.arange(nb) / alpha
    sp2 = np.array([np.interp(src, np.arange(nb), s) for s in sp])
    ap2 = np.array([np.interp(src, np.arange(nb), s) for s in ap])
    y = pw.synthesize(f0 * fr, sp2, np.clip(ap2, 0, 1), FS, 5.0)[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    y = y / (np.sqrt((y ** 2).mean()) + 1e-9) * np.sqrt((x ** 2).mean())
    pk = np.abs(y).max()
    return (y * (0.95 / pk) if pk > 0.95 else y).astype(np.float32)


def prep(j: tuple) -> None:
    src, dst = j
    if not dst.exists():
        sf.write(dst, wnorm(load48(src)), FS)


def enroll(j: tuple) -> None:
    w, o = j
    if not (o / "enroll.json").exists():
        subprocess.run([str(BIN), "artic-enroll", "-r", str(w), "-o", str(o)], cwd=ROOT, capture_output=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="dev")
    a = ap.parse_args()
    S = P.H / a.set
    W = S / "d552"
    (W / "in").mkdir(parents=True, exist_ok=True)
    (W / "en").mkdir(parents=True, exist_ok=True)
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    preps, ens, jobs = [], [], []
    ed = P.H / "e2e" / f"eval_{a.set}_pre_W"
    ed.mkdir(parents=True, exist_ok=True)
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        opw = S / "d56/_work" / f"op_{r['pi']:04d}.wav"
        if not opw.exists():
            opw = S / "d56/_work" / f"ref_{r['spk']}.wav"
        preps += [(S / "d467" / f"{k}.wav", W / "in" / f"{k}.wav"), (opw, W / f"op_{r['pi']:04d}.wav")]
        ens.append((W / f"op_{r['pi']:04d}.wav", W / "en" / f"op_{r['pi']:04d}"))
        jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", W / "en" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["-i", W / "in" / f"{k}.wav"], ed / f"{k}.cond", ed / f"{k}.front"))
    with Pool(4) as pl:
        list(pl.imap_unordered(prep, list(dict.fromkeys(preps))))
        list(pl.imap_unordered(enroll, ens))
        list(pl.imap_unordered(job, jobs))
    print("prepped + enrolled + dumped", sum((ed / f"p{r['pi']:04d}_t{r['ti']:03d}.cond").exists() for r in pairs), flush=True)
    dev = "cpu"
    torch.set_num_threads(6)
    st = torch.load(ROOT / "results/diag_ppgvoc43/product_002500_m0.pt", map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192))
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    e0 = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists() or not (e0 / f"{k}.cond").exists():
            continue
        c0, l0 = P.load_dump(e0, k)
        c1, l1 = P.load_dump(ed, k)
        n = min(len(c0), len(c1))
        spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        T0 = P.load_enroll(S / "d85_work" / f"op_{r['pi']:04d}")[0]
        T1 = P.load_enroll(W / "en" / f"op_{r['pi']:04d}")[0]
        for nm, c, lg, To in (("W_ALL", c1[:n], l1[:n], T1), ("W_POST", c0[:n], l1[:n], T0), ("W_COND", c1[:n], l0[:n], T1)):
            od = Path(a.out) / nm
            od.mkdir(parents=True, exist_ok=True)
            fp = od / f"{k}_{nm}.wav"
            if fp.exists():
                continue
            y = P.render(g, c, lg, spk, dev, T_op=To)[n0:n0 + 8 * N.SR]
            sf.write(fp, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", k, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
