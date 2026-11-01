"""D483(results/conv_p0/d483/prereg.yaml): 操作者の遅い揺らぎ(OpSlow)の長い無音への頑健さ。入力 = 操作者の参照の末尾 12 s + 無音 15 s(−70 dBFS の白色雑音)+ 評価の 8 s を Rust で書き出し、
OP15(現行)・OP15G(c0n < 1 のフレームは平均の輪と移動平均を更新せず z = 0)・RND15C(製品の乱数)で描画して元の 8 s を <out>/<組>/<版>/ に。
    uv run python d483_silence.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d483
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

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d6_unit_cov import load48

PRE, SIL = 12.0, 15.0


def opslow_gated(r: np.ndarray, c0: np.ndarray, tau_s: float, B: np.ndarray) -> np.ndarray:
    """r [24, T]・c0 [T] → z [24, T]: c0n < 1 のフレームは輪と移動平均を更新せず z = 0(Rust に入れるなら同じ規則)。"""
    a = float(np.exp(-1.0 / (tau_s * 200.0)))
    ring = np.zeros((100, 24))
    pos, n, m = 0, 0, np.zeros(24)
    z = np.zeros_like(r)
    for t in range(r.shape[1]):
        if c0[t] < 1.0:
            continue
        if n == 0:
            ring[:] = r[:, t]
        else:
            ring[pos] = r[:, t]
        pos = (pos + 1) % 100
        s = ring.mean(0)
        m = s if n == 0 else a * m + (1 - a) * s
        n += 1
        z[:, t] = B @ (B.T @ (s - m))
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["devff2", "devreal"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    n0, ns = int(PRE * N.SR), int(SIL * N.SR)
    rng = np.random.default_rng(483)
    for s in a.sets:
        S = P.H / s
        W = P.H / s / "d483"
        W.mkdir(parents=True, exist_ok=True)
        ed = P.H / "e2e" / f"eval_{s}_sil"
        ed.mkdir(parents=True, exist_ok=True)
        pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"] if r.get("kind", "FF") != "AA" and (S / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav").exists()]
        jobs = []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            w = W / f"{k}.wav"
            if not w.exists():
                opw = S / "d56" / "_work" / f"op_{r['pi']:04d}.wav"
                pre = load48(opw if opw.exists() else sorted((S / "d56" / "_work").glob("ref_*.wav"))[0])[-n0:]
                pre = np.pad(pre, (n0 - len(pre), 0))
                sil = (rng.standard_normal(ns) * 10 ** (-70 / 20)).astype(np.float32)
                sf.write(w, np.concatenate([pre, sil, load48(S / "d56" / f"{k}_SRC.wav")[: 8 * N.SR]]).astype(np.float32), N.SR)
            jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["-i", w], ed / f"{k}.cond", ed / f"{k}.front"))
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        print("dumped", s, len(jobs), flush=True)
        st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 1.5
        P.slow_from_ckpt(st)
        B = P.SLOW["B"].numpy().astype(np.float64)
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, lg = P.load_dump(ed, k)
            spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
            T_op = P.op_tab(S, k)
            p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
            ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
            stt = P.spk_tensor(spk, dev)
            rr = ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10
            zs = {"OP15": 1.5 * P.opslow_causal(rr, 4.0),
                  "OP15G": 1.5 * torch.from_numpy(opslow_gated(rr[0].cpu().numpy().astype(np.float64), c[:, 0], 4.0, B)).float().to(dev)[None]}
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            exc = P.TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            with torch.no_grad():
                for v in ("OP15", "OP15G", "RND15C"):
                    if v == "RND15C":
                        P.OPSLOW["on"] = False
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt)
                        P.OPSLOW["on"] = True
                    else:
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, zs[v])
                    y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
                    y = y[n0 + ns:n0 + ns + 8 * N.SR]
                    (O / s / v).mkdir(parents=True, exist_ok=True)
                    sf.write(O / s / v / f"{k}_{v}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print("rendered", s, len(pairs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
