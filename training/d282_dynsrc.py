"""D282: 包絡の動きの供給源 = 操作者の包絡の、操作者の表からのずれ(diag_ppgvoc1 の 5k・学習なし・D281 で生成の欠けの主体は包絡の動きと判明)。
包絡 = Σp T_目標 / 10 + γ·(操作者の包絡 − Σp T_操作者 / 10)(線形・γ = 0.35 / 0.7 / 1)と、γ 1 を下限つきの log-mel で足す版(fl1: 操作者の包絡 + 表の差 Δ を
log-mel の各帯の下限の重み clamp((lm − ln1e-5 − 1), 0, 1) で足して下限で止め DCT に戻す・c0 も更新)。PV = dev の 60 組(操作者の表 = d85_work/op_PPPP)・
SR = 保留 30 人(操作者 = 本人・表は同じ登録 → γ 1 は本物の包絡 = D281 の env)。
a<γ> = モデルの出した包絡(文脈の符号器のずれを含む)に γ·(操作者のずれ)を足す(g<γ> は包絡の行を表で上書きする = 文脈の符号器のずれが消える・D318 で判明)/ wct<γ> = 操作者のずれを白色化して目標の残差の共分散で色づけ(C_目標^½ C_操作者^−½・登録の res_cov・SR は恒等)して γ 倍(D283)。
    uv run python d282_dynsrc.py --ckpt ../results/diag_ppgvoc1/last_ema.pt --out ../results/conv_p0/d282
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import pae as PA
import train_ppgvoc as P

LF = math.log(1e-5)


def floor_add(c0n: np.ndarray, env: np.ndarray, d_env: np.ndarray) -> tuple:
    """c0n [T](条件の単位)・env / d_env [T, 24](条件の単位 = 生 / 10)→ 下限つきで足した (c0n, env)。"""
    _, dct = PA._np_consts()
    D = dct[:25]
    raw = np.concatenate([(c0n * 10 + PA.C0_SIL)[:, None], env * 10], 1)
    lm = raw @ D
    dl = np.concatenate([np.zeros((len(env), 1)), d_env * 10], 1) @ D
    w = np.clip(lm - LF - 1.0, 0.0, 1.0)
    lm2 = np.maximum(lm + w * dl, LF)
    r2 = lm2 @ D.T
    return ((r2[:, 0] - PA.C0_SIL) / 10).astype(np.float32), (r2[:, 1:25] / 10).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="g035,g070,g100,fl1")
    ap.add_argument("--set", default="dev", help="PV の組(dev = 男声 TTS v4L・devff = 実女声の操作者・devtts = TTS v3 の女声 p00 / 男声 p01)")
    ap.add_argument("--no_sr", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    held = P.e2e_held()
    sr_items, pv_items = P.eval_items(held)
    if a.set != "dev":
        pv_items = []
        for f in sorted((P.H / "e2e" / f"eval_{a.set}").glob("p0*.cond")):
            ti = int(f.stem.split("_t")[1][:3])
            c, lg = P.load_dump(P.H / "e2e" / f"eval_{a.set}", f.stem)
            pv_items.append((f"{f.stem}_PV", c, lg, P.load_enroll(P.H / a.set / "d85_work" / f"en_{ti:03d}")))
    if a.no_sr:
        sr_items = []
    _, dct = PA._np_consts()
    raw0 = np.zeros(25); raw0[0] = PA.C0_SIL
    assert np.abs((raw0 @ dct[:25]) @ dct[:25].T - raw0).max() < 1e-6 * abs(PA.C0_SIL), "DCT の往復(float32 の基底の相対誤差)"
    ed = P.H / "e2e" / "eval_dev"
    import json
    op_tab, cov = {}, {}

    def sqrtm(C: np.ndarray, pw: float) -> np.ndarray:
        w, V = np.linalg.eigh((C + C.T) / 2)
        return (V * np.maximum(w, 1e-8 * w.max()) ** pw) @ V.T

    def rc(d: Path) -> np.ndarray:
        return np.array(json.loads((d / "enroll.json").read_text())["res_cov"], np.float64).reshape(24, 24)
    for nm, *_ in pv_items:
        pi, ti = int(nm[1:5]), int(nm.split("_t")[1][:3])
        if pi not in op_tab:
            op_tab[pi] = P.load_enroll(P.H / a.set / "d85_work" / f"op_{pi:04d}")[0]
        if (pi, ti) not in cov:
            cov[(pi, ti)] = sqrtm(rc(P.H / a.set / "d85_work" / f"en_{ti:03d}"), 0.5) @ sqrtm(rc(P.H / a.set / "d85_work" / f"op_{pi:04d}"), -0.5)
    groups = []
    for v in a.variants.split(","):
        for grp, items in (("SR", sr_items), ("PV", pv_items)):
            if not items:
                continue
            od = O / v / grp
            od.mkdir(parents=True, exist_ok=True)
            for nm, c, lg, spk in items:
                T_t = spk[0]
                T_o = T_t if grp == "SR" else op_tab[int(nm[1:5])]
                with torch.no_grad():
                    cond, exc, stt = P.cond_exc(g, c, lg, spk, dev)
                    p = torch.from_numpy(lg).softmax(1).numpy()
                    tab_t, tab_o = p @ T_t.T / 10, p @ T_o.T / 10
                    e_op = c[:, 1:25]
                    if v == "fl1":
                        c0n, env = floor_add(c[:, 0], e_op, tab_t - tab_o)
                        cond[0, 0] = torch.from_numpy(c0n).to(dev)
                    elif v.startswith("wct"):
                        gm = int(v[3:]) / 100
                        Wc = np.eye(24) if grp == "SR" else cov[(int(nm[1:5]), int(nm.split("_t")[1][:3]))]
                        env = (tab_t + gm * (e_op - tab_o) @ Wc.T).astype(np.float32)
                    elif v.startswith("a"):
                        gm = int(v[1:]) / 100
                        env = (cond[0, 1:25].cpu().numpy().T + gm * (e_op - tab_o)).astype(np.float32)
                    else:
                        gm = int(v[1:]) / 100
                        env = (tab_t + gm * (e_op - tab_o)).astype(np.float32)
                    cond[0, 1:25] = torch.from_numpy(env.T.copy()).to(dev)
                    y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            groups.append(f"{v}_{grp}={od}/*.wav")
        print("rendered", v, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
