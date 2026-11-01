"""D544(推論だけ・D543 の明るさの漏れの帰属): devff5 の操作者の音声(登録の参照 d56/_work/op_PPPP.wav と入力 d467/<k>.wav)に高域の棚(RBJ high shelf 3 kHz +6 dB・S = 1)をかけ、
登録(artic-enroll)からやり直して Rust で書き出す(eval_devff5_pre_EQ)。P43OP(OP15)を 4 通りで描画: BASE(元の条件・元の事後)・ALL(棚の条件・棚の事後・棚の登録の表)・
POST(元の条件・棚の事後)・COND(棚の条件・元の事後・棚の登録の表)。出力の hf(4〜10 kHz / 0.1〜4 kHz)の差(版 − BASE)を、入力に足した差(棚の入力 − 元の入力)で割った値 = 漏れの割合。
写し合成 CP(rvoc3hi・棚の条件)を対照に(≈ 1 のはず)。出力 = results/conv_p0/d544/。
    uv run python d544_eq_leak.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
import train_rvoc as TR
from d273_data import BIN, job
from d274_evaldump import COM
from d543_bright_leak import spec
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
PRE = 12.0


def shelf(x: np.ndarray, f0: float = 3000.0, gdb: float = 6.0) -> np.ndarray:
    A = 10 ** (gdb / 40)
    w0 = 2 * np.pi * f0 / N.SR
    al = np.sin(w0) / 2 * np.sqrt(2)
    c = np.cos(w0)
    b = np.array([A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)])
    a = np.array([(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al])
    y = lfilter(b / a[0], a / a[0], x.astype(np.float64))
    pk = np.abs(y).max()
    return (y * (0.95 / pk) if pk > 0.95 else y).astype(np.float32)


def enroll(j: tuple) -> None:
    w, o = j
    if (o / "enroll.json").exists():
        return
    subprocess.run([str(BIN), "artic-enroll", "-r", str(w), "-o", str(o)], cwd=ROOT, capture_output=True)


def main() -> int:
    S = P.H / "devff5"
    W = S / "d544"
    O = ROOT / "results/conv_p0/d544"
    (W / "in").mkdir(parents=True, exist_ok=True)
    (W / "en").mkdir(parents=True, exist_ok=True)
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    ens, jobs = [], []
    ed = P.H / "e2e" / "eval_devff5_pre_EQ"
    ed.mkdir(parents=True, exist_ok=True)
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        ow = W / f"op_{r['pi']:04d}.wav"
        if not ow.exists():
            sf.write(ow, shelf(load48(S / "d56/_work" / f"op_{r['pi']:04d}.wav")), N.SR)
        iw = W / "in" / f"{k}.wav"
        if not iw.exists():
            sf.write(iw, shelf(load48(S / "d467" / f"{k}.wav")), N.SR)
        ens.append((ow, W / "en" / f"op_{r['pi']:04d}"))
        jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", W / "en" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["-i", iw], ed / f"{k}.cond", ed / f"{k}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(enroll, ens))
        list(pl.imap_unordered(job, jobs))
    print("enrolled + dumped", sum((ed / f"p{r['pi']:04d}_t{r['ti']:03d}.cond").exists() for r in pairs), flush=True)
    dev = "cuda"
    st = torch.load(ROOT / "results/diag_ppgvoc43/product_002500_m0.pt", map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192))
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    s_ = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    import rvoc as R
    gcp = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
    gcp.load_state_dict(s_["ema"])
    gcp.eval()
    e0 = P.H / "e2e" / "eval_devff5_pre"
    n0 = int(PRE * N.SR)
    rows = {}
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c0, l0 = P.load_dump(e0, k)
        c1, l1 = P.load_dump(ed, k)
        n = min(len(c0), len(c1))
        c0, l0, c1, l1 = c0[:n], l0[:n], c1[:n], l1[:n]
        spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        T0 = P.load_enroll(S / "d85_work" / f"op_{r['pi']:04d}")[0]
        T1 = P.load_enroll(W / "en" / f"op_{r['pi']:04d}")[0]
        out = {}
        for nm, c, lg, To in (("BASE", c0, l0, T0), ("ALL", c1, l1, T1), ("POST", c0, l1, T0), ("COND", c1, l0, T1)):
            y = P.render(g, c, lg, spk, dev, T_op=To)[n0:n0 + 8 * N.SR]
            out[nm] = spec(y)[0]
        for nm, c in (("CP0", c0), ("CP1", c1)):
            with torch.no_grad():
                cond = torch.from_numpy(c.T.astype(np.float32))[None].to(dev)
                f0 = torch.where(cond[:, 26] > 0.5, 200.0 * torch.exp(cond[:, 25]), torch.zeros_like(cond[:, 25]))
                y = gcp(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:][n0:n0 + 8 * N.SR]
            out[nm] = spec(y)[0]
        out["SRC0"] = spec(load48(S / "d467" / f"{k}.wav")[n0:n0 + 8 * N.SR])[0]
        out["SRC1"] = spec(load48(W / "in" / f"{k}.wav")[n0:n0 + 8 * N.SR])[0]
        rows[k] = out
    inj = np.array([v["SRC1"] - v["SRC0"] for v in rows.values()])
    rep = {"injected_dhf_median": round(float(np.median(inj)), 2), "n": len(rows)}
    for nm, base in (("ALL", "BASE"), ("POST", "BASE"), ("COND", "BASE"), ("CP1", "CP0")):
        d = np.array([v[nm] - v[base] for v in rows.values()])
        rep[nm] = {"delta_median": round(float(np.median(d)), 2), "leak": round(float(np.median(d / inj)), 3)}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    O.mkdir(parents=True, exist_ok=True)
    (O / "d544.json").write_text(json.dumps({"summary": rep, "per_utt": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
