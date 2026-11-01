"""D532(推論だけの検査・学習なし): 学習(自己再構成)の f0・有声 = オフラインの教師(f0hi.teacher_f0 = harvest + DIO48)・推論 = 因果の F1(有声 = ロジット > 0)の不一致の帰属。
候補 P43OP を devff5 の定常状態の条件(eval_devff5_pre)で、f0・有声だけを入力(前置き 12 s + 元の 8 s)の教師の f0 に差し替えて描画する(TF0 = 神託・非因果・診断のみ)。
声域の移しは発話ごとに 条件の f0 / 教師の f0 の中央値(両方有声)・時刻のずれは ±4 フレームで条件の log f0 との差の分散が最小のもの。周期性の行(P43 は 目標の定数 × 有声)は差し替えた有声に従う。
TF0 ≫ F1 なら F1 の有声・f0 の不一致が息の端のピッチ・かすれの一因 = 学習で F1 を使う(学習と推論の一致)か F1 の判定を直す。評価は d467_paired_v1・d524・d533(息の周期性)で別に。
    uv run python d532_oracle_f0.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d532
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
import f0hi as FH
import nvoc as N
import train_ppgvoc as P
from d6_unit_cov import load48

PRE = 12.0


def oracle(c: np.ndarray, x48: np.ndarray) -> tuple:
    T = len(c)
    ft, _ = FH.teacher_f0(x48.astype(np.float64), n=T + 8)
    vc = c[:, 26] > 0.5
    best = None
    for L in range(-4, 5):
        fs = np.zeros(T, np.float32)
        lo, hi = max(0, -L), min(T, len(ft) - L)
        fs[lo:hi] = ft[lo + L:hi + L]
        m = vc & (fs > 0)
        if m.sum() < 50:
            continue
        d = c[m, 25] - np.log(fs[m])
        s = float(np.var(d))
        if best is None or s < best[0]:
            best = (s, L, fs, float(np.median(d)))
    _, L, fs, off = best
    c2 = c.copy()
    v2 = fs > 0
    c2[:, 26] = v2.astype(np.float32)
    c2[:, 25] = np.where(v2, np.log(np.maximum(fs, 1e-3)) + off, c[:, 25])
    return c2, L, float(np.mean(v2 != vc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff5")
    ap.add_argument("--name", default="TF0")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    od = Path(a.out) / a.name
    od.mkdir(parents=True, exist_ok=True)
    info = {}
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        c2, L, dv = oracle(c, load48(S / "d467" / f"{k}.wav"))
        info[k] = {"lag": L, "voicing_changed": round(dv, 4)}
        spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        y = P.render(g, c2, lg, spk, dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
        sf.write(od / f"{k}_{a.name}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
    (Path(a.out) / f"{a.name}_info.json").write_text(json.dumps(info, indent=1))
    lags = [v["lag"] for v in info.values()]
    print("rendered", len(info), "lag", np.bincount(np.array(lags) + 4).tolist(), "voicing_changed median", np.median([v["voicing_changed"] for v in info.values()]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
