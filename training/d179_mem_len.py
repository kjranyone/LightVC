"""D179(学習なし・GPU 推論): 実女声の開発用の組(d139 と同じ入力・表・f0・周期性・出力部)で R の記憶だけを変える。
  RRM05 / RRM10 / RRM20 = 参照の特徴の先頭 5 / 10 / 20 秒(全体 ≤ 25 秒)を記憶に・RRSWP = 別の目標((ti + 1) mod 30)の参照全体を記憶に(表は正しい目標)。
出力 = <s>/d179/pNNNN_tNNN_<cond><sfx>.wav(+ TABR・SRC)。事前登録 = results/conv_p0/d179_prereg.yaml。
    uv run python d179_mem_len.py --s <dev> --tag diag_rr1c
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    ap.add_argument("--sfx", default="")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = S / "d179"; od.mkdir(exist_ok=True)
    FR = S / "rr_real"

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy(), "p4": d["p4"].float().numpy()}

    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    net = RRm.RR(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk")}, **{k_: tuple(ck["cfg"][k_]) for k_ in ("qd", "kd") if k_ in ck["cfg"]}).to(dev).eval()
    net.load_state_dict(ck["net"])
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
        tgt[t] = {"rp": r["p"][:, :n].astype(np.float32), "rv": np.concatenate([r["env"][1:25, :n] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32), "T": T, "env": r["env"][:, :n],
                  "mu": float(np.median(np.log(r["ft"][r["ft"] > 0]))), "p4": r["p4"][:, r["f1"] > 0].mean(1)}
    with torch.no_grad():
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if (od / f"{k}_RRSWP{a.sfx}.wav").exists():
                continue
            b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
            vo = o["f1"] > 0
            sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
            vM = b["f1"] > 0
            q = np.concatenate([b["p"], (b["e3"] - sM["e3"][:, None]) / 10, (b["c0f"] - sM["c0"])[None],
                                np.where(vM, np.log(np.maximum(b["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
            if ck["cfg"].get("no_op_env"):
                q[RRm.K:RRm.K + 24] = 0
            tg = tgt[ti]
            tab = (tg["T"] @ b["p"]).astype(np.float32)
            f0o = np.where(vM, b["f1"] * np.exp(tg["mu"] - sM["lf"]), 0.0).astype(np.float64)
            per = (b["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
            outs = {"TABR": (b["c0f"].astype(np.float64), tab.astype(np.float64))}
            for c, (src, m) in {"RRM05": (ti, 1000), "RRM10": (ti, 2000), "RRM20": (ti, 5000), "RRSWP": ((ti + 1) % 30, 5000)}.items():
                e_ = tgt[src]["env"][:, :m]
                rc0 = (e_[0] - RRm.C0_SIL) / 10
                rv = np.concatenate([e_[1:25] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
                rp = tgt[src]["rp"][:, :m]
                mem = net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev))
                env, dc0, _ = net(torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab)[None].to(dev), mem, torch.ones(1, rp.shape[1], dtype=torch.bool, device=dev))
                outs[c + a.sfx] = ((b["c0f"] + dc0[0].cpu().numpy()).astype(np.float64), env[0].cpu().numpy().astype(np.float64))
            for c, (cc, ee) in outs.items():
                sf.write(od / f"{k}_{c}.wav", render(cc, ee, f0o, per), N.SR)
            shutil.copy(f, od / f.name)
            print(k, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
