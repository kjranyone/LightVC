"""D603(推論だけ・介入): 男声の入力(<組>/d467/<key>.wav = 前置き 12 s + SRC)の 150〜450 Hz を ±6 dB(零位相の帯域通過を足し引き)にして、
P58 の重みで d570 と同じ経路(Rust の条件の書き出し → 生成器・k = 5)で描画し、出力の低い帯が入力の変化を追う割合(操作者の漏れの利得)を測る。
版 = OURS5Y58 (元・d570 の既存) / OURS5Y58Qm6 / OURS5Y58Qp6。自己検査 = 元の書き出しで描き直した音が既存の OURS5Y58 と一致。
    uv run python d603_input_eq.py --sets dev2 dev3 --out ../results/conv_p0/d603/eq.json
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf
import torch

import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d592_refkc import boot
from d602_lowband import CF, LOW, ltas, ltas_x
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results/diag_ppgvoc58/product_002500_m0.pt"
PRE = 12.0
K = 5
GAINS = {"m6": -6.0, "p6": 6.0}


def eq_one(args: tuple) -> int:
    src, dst, gdb = args
    if Path(dst).exists():
        return 0
    x = load48(src).astype(np.float64)
    sos = ss.butter(2, [150.0, 450.0], btype="bandpass", fs=N.SR, output="sos")
    y = x + (10 ** (gdb / 20) - 1) * ss.sosfiltfilt(sos, x)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y.astype(np.float32), N.SR, subtype="FLOAT")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs_by: dict = {}
    for s in a.sets:
        S = P.H / s
        ED0 = P.H / "e2e" / f"eval_{s}_pre_nof0"
        pairs_by[s] = [f"p{r['pi']:04d}_t{r['ti']:03d}" for r in json.loads((S / "devset.json").read_text())["pairs"]
                       if (S / "d467" / f"p{r['pi']:04d}_t{r['ti']:03d}.wav").exists() and (ED0 / f"p{r['pi']:04d}_t{r['ti']:03d}.cond").exists()
                       and (S / "d570" / "OURS5Y58" / f"p{r['pi']:04d}_t{r['ti']:03d}_OURS5Y58.wav").exists()]
        eqj, dj = [], []
        for tag, gdb in GAINS.items():
            for k in pairs_by[s]:
                pi, ti = int(k[1:5]), int(k[7:10])
                fin = S / f"d603_in_{tag}" / f"{k}.wav"
                eqj.append((S / "d467" / f"{k}.wav", fin, gdb))
                ED = P.H / "e2e" / f"eval_{s}_d603_{tag}"
                ED.mkdir(parents=True, exist_ok=True)
                dj.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{ti:03d}", "--operator-enroll", S / "d85_work" / f"op_{pi:04d}", "--med-from-operator"] + COM + ["--no-f0-shift", "-i", fin], ED / f"{k}.cond", ED / f"{k}.front"))
        with Pool(12) as pl:
            print(s, "eq written", sum(pl.imap_unordered(eq_one, eqj)), flush=True)
        with Pool(3) as pl:
            list(pl.imap_unordered(job, dj))
        print(s, "dumps", len(dj), flush=True)
    dev = "cuda"
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 0.0
    P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    import train_c1 as TC
    n0 = int(PRE * N.SR)

    def rend(S: Path, ED: Path, fin: Path, k: str) -> np.ndarray:
        c, lg = P.load_dump(ED, k)
        pre = c[:2400, 0]
        mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
        c = c.copy()
        c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
        vv = c[:, 26] > 0.5
        en = P.load_enroll(S / "d85_work" / f"en_{int(k[7:10]):03d}")
        cvf = P.cvin_frames(torch.from_numpy(TC.prime(load48(fin).astype(np.float32)))[None].to(dev), len(lg))
        c[:, 25] = np.where(vv, c[:, 25] + K * np.log(2) / 12, c[:, 25])
        y = P.render(g, c, lg, en, dev, T_op=P.op_tab(S, k), cvf=cvf)[n0:n0 + 8 * N.SR]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    chk = []
    for s in a.sets:
        S = P.H / s
        for k in pairs_by[s][:3]:
            y = rend(S, P.H / "e2e" / f"eval_{s}_pre_nof0", S / "d467" / f"{k}.wav", k)
            y0 = load48(S / "d570" / "OURS5Y58" / f"{k}_OURS5Y58.wav")
            n = min(len(y), len(y0))
            chk.append(float(np.abs(y[:n] - y0[:n]).max()))
    print("self-check max_abs_diff vs existing OURS5Y58", [round(v, 5) for v in chk], flush=True)
    if max(chk) > 2e-3:
        print("self-check FAIL", flush=True)
        return 2
    for s in a.sets:
        S = P.H / s
        for tag in GAINS:
            od = S / "d570" / f"OURS5Y58Q{tag}"
            od.mkdir(parents=True, exist_ok=True)
            for k in pairs_by[s]:
                o = od / f"{k}_OURS5Y58Q{tag}.wav"
                if not o.exists():
                    sf.write(o, rend(S, P.H / "e2e" / f"eval_{s}_d603_{tag}", S / f"d603_in_{tag}" / f"{k}.wav", k), N.SR)
            print(s, tag, "rendered", len(pairs_by[s]), flush=True)
    rows: dict = {}
    for s in a.sets:
        S = P.H / s
        for k in pairs_by[s]:
            r = {"t": int(k[7:10]), "out0": float(ltas(S / "d570" / "OURS5Y58" / f"{k}_OURS5Y58.wav")[LOW].mean()), "in0": lo_in(S / "d467" / f"{k}.wav", n0)}
            for tag in GAINS:
                r[f"out_{tag}"] = float(ltas(S / "d570" / f"OURS5Y58Q{tag}" / f"{k}_OURS5Y58Q{tag}.wav")[LOW].mean())
                r[f"in_{tag}"] = lo_in(S / f"d603_in_{tag}" / f"{k}.wav", n0)
            rows[(s, k)] = r
    keys = sorted(rows)
    tg = np.array([f"{k[0]}|{rows[k]['t']}" for k in keys])
    rng = np.random.default_rng(603)
    rep: dict = {"n": len(keys), "self_check": chk, "low_bands_hz": [round(float(CF[i]), 1) for i in LOW], "by_gain": {}}
    for tag in GAINS:
        din = np.array([rows[k][f"in_{tag}"] - rows[k]["in0"] for k in keys])
        dout = np.array([rows[k][f"out_{tag}"] - rows[k]["out0"] for k in keys])
        ci = boot(dout, tg, rng)
        rep["by_gain"][tag] = {"d_in": [round(float(din.mean()), 3)] + boot(din, tg, rng), "d_out": [round(float(dout.mean()), 3)] + ci,
                               "ratio": [round(float(dout.mean() / din.mean()), 3)] + sorted(round(float(v / din.mean()), 3) for v in ci)}
        print(tag, rep["by_gain"][tag], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


def lo_in(p: Path, n0: int) -> float:
    return float(ltas_x(load48(p)[n0:n0 + 8 * N.SR])[LOW].mean())


if __name__ == "__main__":
    raise SystemExit(main())
