"""D565(推論だけ・診断): 男声 → 女声の本人らしさの律速が内容の経路(事後・C2)かを、同じ文・同じ目標を読む女声 TTS(devf)の事後・C2 に替えて測る。
女声の事後・C2 は全体の書き出し(eval_devf・devf/d56 の SRC)から C1 の対数事後の余弦の DTW で男声の時刻へ写す。写しの害の対照 RT = 男声 → 女声 → 男声の往復の写像で男声の事後・C2。
条件は定常状態(eval_dev_pre・前置き 12 s は男声のまま)・水準ずらし(D558)・遅い揺らぎ 0(P57S0 と同じ)。版の説明は OVERNIGHT の D565 の規則。
    uv run python d565_content_swap.py --ckpt ../results/diag_ppgvoc57/product_002500_m0.pt --out ../results/conv_p0/d565
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48

PRE = 12.0
FEM = 4.40
NP = 2400
VARS = ("NONE", "RT", "PF", "CF", "PCF", "PCFF", "FEMIN")


def align(la: np.ndarray, lb: np.ndarray) -> np.ndarray:
    a = la / (np.linalg.norm(la, axis=1, keepdims=True) + 1e-9)
    b = lb / (np.linalg.norm(lb, axis=1, keepdims=True) + 1e-9)
    _, wp = librosa.sequence.dtw(C=1 - a @ b.T)
    idx = np.zeros(len(la))
    cnt = np.zeros(len(la))
    for i, j in wp[::-1]:
        idx[i] += j
        cnt[i] += 1
    return np.round(idx / np.maximum(cnt, 1)).astype(int)


def logsm(lg: np.ndarray) -> np.ndarray:
    return torch.from_numpy(lg).float().log_softmax(-1).numpy()


def shift_c0(c: np.ndarray) -> np.ndarray:
    pre = c[:NP, 0]
    mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else FEM
    c = c.copy()
    c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + FEM - mu, c[:, 0])
    return c


def cv(w: Path, T: int, dev: str) -> torch.Tensor:
    return P.cvin_frames(torch.from_numpy(TC.prime(load48(w).astype(np.float32)))[None].to(dev), T)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192))
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 0.0
    P.slow_from_ckpt(st)
    S, SF = P.H / "dev", P.H / "devf"
    E = P.H / "e2e"
    n0 = int(PRE * N.SR)
    O = Path(a.out)
    rows = []
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        cm, lm = P.load_dump(E / "eval_dev_pre", k)
        cmf, lmf = P.load_dump(E / "eval_devf", k)
        _, lmm = P.load_dump(E / "eval_dev", k)
        for w, n in ((S / "d56" / f"{k}_SRC.wav", len(lmm)), (SF / "d56" / f"{k}_SRC.wav", len(lmf))):
            assert abs(sf.info(str(w)).frames * N.SR // sf.info(str(w)).samplerate // N.HOP - n) <= 3, (w, n)
        imf = align(logsm(lmm), logsm(lmf))
        ifm = align(logsm(lmf), logsm(lmm))
        T = min(len(lm) - NP, len(lmm))
        jf = imf[:T]
        tr = ifm[jf]
        u = slice(NP, NP + T)
        cv0 = cv(S / "d467" / f"{k}.wav", len(lm), dev)
        cvm = cv(S / "d56" / f"{k}_SRC.wav", len(lmm), dev)
        cvff = cv(SF / "d56" / f"{k}_SRC.wav", len(lmf), dev)
        spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        c0 = shift_c0(cm)
        rows.append({"k": k, "T": T, "rt_err_frames": float(np.mean(np.abs(tr - np.arange(T))))})
        for v in VARS:
            c, lg, cvf, Sd = c0.copy(), lm.copy(), cv0.clone(), S
            if v in ("RT", "CF"):
                lg[u] = lmm[tr]
            if v in ("PF", "PCF", "PCFF"):
                lg[u] = lmf[jf]
            if v in ("RT", "PF"):
                cvf[..., u] = cvm[..., tr]
            if v in ("CF", "PCF", "PCFF"):
                cvf[..., u] = cvff[..., jf]
            if v == "PCFF":
                vf = cmf[jf, 26]
                c[u, 26] = vf
                c[u, 25] = np.where(vf > 0.5, cmf[jf, 25], 0.0)
            if v == "FEMIN":
                cf_, lf_ = P.load_dump(E / "eval_devf_pre", k)
                c, lg, Sd = shift_c0(cf_), lf_, SF
                cvf = cv(SF / "d467" / f"{k}.wav", len(lf_), dev)
            y = P.render(g, c, lg, spk, dev, T_op=P.op_tab(Sd, k), cvf=cvf)[n0:n0 + 8 * N.SR]
            od = O / v / "dev"
            od.mkdir(parents=True, exist_ok=True)
            sf.write(od / f"{k}_{v}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        print(k, rows[-1], flush=True)
    (O / "info.json").write_text(json.dumps({"n": len(rows), "rt_err_frames_median": float(np.median([x["rt_err_frames"] for x in rows])), "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
