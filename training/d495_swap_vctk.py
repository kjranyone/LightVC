"""D495(D491 を実男声で・VCTK の内容をそろえた組 devvctkm: 男声 p01NN ↔ 同じ区間・同じ目標の女声 p00NN): 条件を 1 つずつ女声の版に入れ替える。以下 D491 の説明: dev(男声 TTS v4L)と devf(同じ TTS・同じ文・同じ目標の女声)の Rust の書き出しで、女声の版の条件を男声の時刻へ DTW(対数事後の余弦)で写し、
男声の条件を 1 つずつ入れ替えて候補(P43OP)で描画する。版 = NONE・PF(事後)・CF(c0n)・FF0(log f0 = 目標の声域へ移した軌跡)・VF(有声)・ALLF(全部 = 女声の変換を男声の時刻へ)。
評価は d488_paired_windows(窓ごとの同じ判別器の対の差)。整列の誤差を含む診断(製品の手ではない)。
    uv run python d491_swapcarrier.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d491
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
import train_ppgvoc as P

VARS = ("NONE", "PF", "CF", "FF0", "VF", "ALLF")


def align(lm: np.ndarray, lf: np.ndarray) -> np.ndarray:
    """男声のフレーム t → 女声のフレームの番号(DTW の経路の平均・単調)。"""
    import librosa
    a = lm / (np.linalg.norm(lm, axis=1, keepdims=True) + 1e-9)
    b = lf / (np.linalg.norm(lf, axis=1, keepdims=True) + 1e-9)
    C = 1 - a @ b.T
    _, wp = librosa.sequence.dtw(C=C, subseq=False)
    wp = wp[::-1]
    idx = np.zeros(len(lm))
    cnt = np.zeros(len(lm))
    for i, j in wp:
        idx[i] += j
        cnt[i] += 1
    return np.round(idx / np.maximum(cnt, 1)).astype(int)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    n = 0
    rows = json.loads((P.H / "devvctkm" / "devset.json").read_text())["pairs"]
    fem = {r["pi"] % 100: r for r in rows if r["kind"].startswith("F")}
    for r in rows:
        if not r["kind"].startswith("M") or (r["pi"] % 100) not in fem:
            continue
        rf = fem[r["pi"] % 100]
        if rf["ti"] != r["ti"] or rf.get("block") != r.get("block"):
            continue
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        kf = f"p{rf['pi']:04d}_t{rf['ti']:03d}"
        ed = P.H / "e2e" / "eval_devvctkm"
        if not (ed / f"{k}.cond").exists() or not (ed / f"{kf}.cond").exists():
            continue
        cm, lm = P.load_dump(ed, k)
        cf, lf = P.load_dump(ed, kf)
        idx = align(torch.from_numpy(lm).float().log_softmax(-1).numpy(), torch.from_numpy(lf).float().log_softmax(-1).numpy())
        cfa, lfa = cf[idx], lf[idx]
        spk = P.load_enroll(P.H / "devvctkm" / "d85_work" / f"en_{r['ti']:03d}")
        for v in VARS:
            cc, ll = cm.copy(), lm.copy()
            if v in ("PF", "ALLF"):
                ll = lfa
            if v in ("CF", "ALLF"):
                cc[:, 0] = cfa[:, 0]
            if v in ("FF0", "ALLF"):
                both = (cm[:, 26] > 0.5) & (cfa[:, 26] > 0.5)
                cc[:, 25] = np.where(both, cfa[:, 25], cc[:, 25])
            if v in ("VF", "ALLF"):
                cc[:, 26] = cfa[:, 26]
                cc[:, 25] = np.where(cfa[:, 26] > 0.5, np.where(cc[:, 25] != 0, cc[:, 25], cfa[:, 25]), 0.0)
            if v == "ALLF":
                cc[:, 1:25] = cfa[:, 1:25]
            y = P.render(g, cc, ll, spk, dev, T_op=P.op_tab(P.H / "devvctkm", k) if P.OPSLOW["on"] else None)
            (O / v).mkdir(parents=True, exist_ok=True)
            sf.write(O / v / f"{k}_{v}.wav", y, N.SR)
        n += 1
    print("rendered", n, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
