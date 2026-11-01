"""D150(学習なし・開発用の組): 分布の生成器 F(train_dynflow.py)の標本で変換する。実女声 30 目標 × 男声 60 組(D143)・操作者の時間のまま(DTW なし)。
入力特徴 = <dev>/rr_real(prep_rr)。出力 = 包絡 = 表 Σp_M T_目標 + 10·x[:24]・c0 = 男声の F2 c0 + x[24](相対)・f0 / 周期性は d139 と同じ。製品の出力部 rvtab_x で鳴らす。
  FL10 = 温度 1.0・FL07 = 温度 0.7(雑音を 0.7 倍)。6 段の Euler・雑音は AR(1)・乱数は組ごとに固定。出力 = <dev>/d139/pNNNN_tNNN_{FL10, FL07}.wav。
    uv run python d150_flow_dev.py --s <dev> --tag diag_flow1
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
import train_dynflow as FL
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--nopast", action="store_true", help="過去フレームを使わない標本(FN10・FN07)")
    ap.add_argument("--pre", default="", help="条件名の接頭辞(既定 FL / FN)")
    ap.add_argument("--temps", default="1.0,0.7", help="温度(条件名 = 接頭辞 + 温度 ×10 の 2 桁)")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = S / "d139"; od.mkdir(exist_ok=True)
    FR = S / "rr_real"

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy(), "p4": d["p4"].float().numpy()}

    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    net = FL.Flow(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk", "K", "hid")}).to(dev).eval()
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
        rc0 = (r["env"][0, :n] - RRm.C0_SIL) / 10
        tgt[t] = {"rp": r["p"][:, :n].astype(np.float32), "rv": np.concatenate([r["env"][1:25, :n] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32),
                  "T": C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32), "mu": float(np.median(np.log(r["ft"][r["ft"] > 0]))), "p4": r["p4"][:, r["f1"] > 0].mean(1)}
    with torch.no_grad():
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            pre = a.pre or ("FN" if a.nopast else "FL")
            temps = [float(t) for t in a.temps.split(",")]
            if (od / f"{k}_{pre}{int(round(temps[-1] * 10)):02d}.wav").exists():
                continue
            b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
            vo = o["f1"] > 0
            sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
            vM = b["f1"] > 0
            q = np.concatenate([b["p"], (b["e3"] - sM["e3"][:, None]) / 10, (b["c0f"] - sM["c0"])[None],
                                np.where(vM, np.log(np.maximum(b["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
            tg = tgt[ti]
            tab = torch.from_numpy((tg["T"] @ b["p"]).astype(np.float32))[None].to(dev)
            mem = net.rr.memory(torch.from_numpy(tg["rp"])[None].to(dev), torch.from_numpy(tg["rv"])[None].to(dev))
            h, _ = net.context(torch.from_numpy(q)[None].to(dev), mem, torch.ones(1, tg["rp"].shape[1], dtype=torch.bool, device=dev))
            f0o = np.where(vM, b["f1"] * np.exp(tg["mu"] - sM["lf"]), 0.0).astype(np.float64)
            per = (b["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
            for nm, temp in [(pre + f"{int(round(t * 10)):02d}", t) for t in temps]:
                g = torch.Generator(device=dev).manual_seed(pi)
                smp = net.sample_nopast if a.nopast else net.sample
                x = smp(h, tab, steps=a.steps, gen=g, temp=temp)[0].cpu().numpy().astype(np.float64)
                env = tab[0].cpu().numpy().astype(np.float64) + 10 * x[:24]
                c0n = b["c0f"].astype(np.float64) + x[24]
                sf.write(od / f"{k}_{nm}.wav", render(c0n, env, f0o, per), N.SR)
            print(k, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
