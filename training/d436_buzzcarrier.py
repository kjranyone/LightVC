"""D436: 実音声の入力の変換でだけ出るフレーム周期(200 Hz)の包絡変調の線(D435)を、どの条件のフレームごとの揺れが運ぶか。
devff(実女声 → 女声・目標ごとに 1 本)を候補(既定 D2)で、条件を 1 つずつ因果な 1 次の平滑(係数 a)にかけて描画し amline(eval_nvoc)の中央値を比べる。
あわせて条件のフレームごとの揺れ(事後の L1・c0n・log f0 の |Δ|・有声の反転 / 秒)を実女声の入力(devff)と TTS 男声の入力(dev)で並べる。
    uv run python d436_buzzcarrier.py --ckpt ../results/diag_ppgvoc41/product_002500_m0.pt --out ../results/conv_p0/d436
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
from eval_nvoc import amline

VARS = ("ORIG", "P_S", "C0_S", "F0_S", "VV_S", "ALL_S")


def ema(x: np.ndarray, a: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = x[0]
    for t in range(len(x)):
        acc = a * acc + (1 - a) * x[t]
        y[t] = acc
    return y


def items_of(s: str) -> list:
    seen, out = set(), []
    for f in sorted((P.H / "e2e" / f"eval_{s}").glob("p0*.cond")):
        ti = int(f.stem.split("_t")[1][:3])
        if ti not in seen:
            seen.add(ti)
            out.append((f.stem, ti))
    return out


def jitter(c: np.ndarray, lg: np.ndarray) -> dict:
    p = torch.from_numpy(lg).float().softmax(-1).numpy()
    v = c[:, 26] > 0.5
    vv = v[1:] & v[:-1]
    return {"post_l1": float(np.abs(np.diff(p, axis=0)).sum(1).mean()), "c0n": float(np.abs(np.diff(c[:, 0])).mean()),
            "lf_voiced": float(np.abs(np.diff(c[:, 25]))[vv].mean()) if vv.any() else float("nan"), "vv_flips_per_s": float(np.abs(np.diff(v.astype(int))).sum() / (len(v) / 200))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=float, default=0.6)
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
    rep: dict = {"a": a.a, "jitter": {}}
    for s in ("devff", "dev"):
        J = [jitter(*P.load_dump(P.H / "e2e" / f"eval_{s}", k)) for k, _ in items_of(s)]
        rep["jitter"][s] = {k: round(float(np.nanmedian([j[k] for j in J])), 4) for k in J[0]}
        print("揺れ", s, rep["jitter"][s], flush=True)
    vals = {v: [] for v in VARS}
    ed = P.H / "e2e" / "eval_devff"
    for k, ti in items_of("devff"):
        c, lg = P.load_dump(ed, k)
        spk = P.load_enroll(P.H / "devff" / "d85_work" / f"en_{ti:03d}")
        for v in VARS:
            cc, ll = c.copy(), lg.copy()
            if v in ("P_S", "ALL_S"):
                ll = ema(ll, a.a)
            if v in ("C0_S", "ALL_S"):
                cc[:, 0] = ema(cc[:, 0], a.a)
            if v in ("F0_S", "ALL_S"):
                cc[:, 25] = ema(cc[:, 25], a.a)
            if v in ("VV_S", "ALL_S"):
                cc[:, 26] = (ema(cc[:, 26], a.a) > 0.5).astype(np.float32)
            y = P.render(g, cc, ll, spk, dev)
            sf.write(O / v / f"{k}_{v}.wav", y, N.SR)
            vals[v].append(amline(y[: 8 * N.SR]))
    rep["amline_median"] = {v: round(float(np.median(x)), 3) for v, x in vals.items()}
    rep["amline_frac_above_real_p85"] = {v: round(float(np.mean(np.array(x) > 0.539)), 3) for v, x in vals.items()}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (O / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
