"""D180(学習なし・GPU 推論・D129 の素朴な読みを実女声の開発用の組へ): 入力・表・f0・周期性・出力部は d139 と同じ。包絡 c1..c24 と相対 c0 を、
男声の内容の鍵(C1 の事後の平方根 + 直前 4 フレームの平均・正規化 = d106 の ctxkey)で目標の参照 20 秒を柔らかい注意(温度 τ)で読んで作る(写しの監視 = 有効フレーム数 eff ≥ 8)。
  XRD<τ> = 読んだ包絡 + 読んだ c0・XRD<τ>E = 読んだ包絡 + 男声の c0・XRD<τ>C = 表 + 読んだ c0(τ = 0.05 → 05・0.02 → 02)。c0 の水準 = 参照の有声平均を引き男声(操作者)の有声平均を足す。
出力 = <s>/d180/pNNNN_tNNN_<cond>.wav(+ TABR・SRC)・<s>/d180/_read_log.json。事前登録 = results/conv_p0/d180_prereg.yaml。
    uv run python d180_read_real.py --s <dev>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = S / "d180"; od.mkdir(exist_ok=True)
    FR = S / "rr_real"

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy(), "p4": d["p4"].float().numpy()}


    def ctxkey(p: np.ndarray) -> np.ndarray:
        sp = np.sqrt(np.maximum(p, 0))
        cs = np.cumsum(np.pad(sp, ((0, 0), (5, 0))), 1)
        prev = (cs[:, 4:-1] - cs[:, :-5]) / 4
        k_ = np.concatenate([sp, prev], 0)
        return k_ / (np.linalg.norm(k_, axis=0, keepdims=True) + 1e-9)

    def attend(q: np.ndarray, Kr: np.ndarray, vals: np.ndarray, tau: float) -> tuple:
        qt = torch.from_numpy(q.T).float().to(dev); kt = torch.from_numpy(Kr).float().to(dev)
        w = torch.softmax(qt @ kt / tau, 1)
        eff = float((1.0 / (w ** 2).sum(1)).median())
        return (w @ torch.from_numpy(vals.T).float().to(dev)).cpu().numpy().T.astype(np.float64), eff
    sr_ = torch.load(ROOT / a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        n = min(r["env"].shape[1], 5000)
        T = C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)
        rc0 = (r["env"][0, :n] - RRm.C0_SIL) / 10
        vr = r["f1"][:n] > 0
        tgt[t] = {"Kc": ctxkey(r["p"][:, :n]), "vals": np.concatenate([r["env"][1:25, :n], (rc0 - rc0[vr].mean())[None]], 0), "T": T,
                  "mu": float(np.median(np.log(r["ft"][r["ft"] > 0]))), "p4": r["p4"][:, r["f1"] > 0].mean(1)}
    rlog = []
    with torch.no_grad():
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if (od / f"{k}_XRD02C.wav").exists():
                continue
            b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
            vo = o["f1"] > 0
            sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
            vM = b["f1"] > 0
            tg = tgt[ti]
            tab = (tg["T"] @ b["p"]).astype(np.float32)
            f0o = np.where(vM, b["f1"] * np.exp(tg["mu"] - sM["lf"]), 0.0).astype(np.float64)
            per = (b["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
            outs = {"TABR": (b["c0f"].astype(np.float64), tab.astype(np.float64))}
            qk = ctxkey(b["p"])
            for nm_, tau in (("XRD05", 0.05), ("XRD02", 0.02)):
                rd, eff = attend(qk, tg["Kc"], tg["vals"], tau)
                c0r = rd[24] + sM["c0"]
                outs[nm_] = (c0r, rd[:24])
                outs[nm_ + "E"] = (b["c0f"].astype(np.float64), rd[:24])
                outs[nm_ + "C"] = (c0r, tab.astype(np.float64))
                rlog.append({"pair": k, "cond": nm_, "eff": round(eff, 1)})
            for c, (cc, ee) in outs.items():
                sf.write(od / f"{k}_{c}.wav", render(cc, ee, f0o, per), N.SR)
            shutil.copy(f, od / f.name)
            print(k, rlog[-2:], flush=True)
    (od / "_read_log.json").write_text(json.dumps(rlog, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
