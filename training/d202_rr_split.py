"""D202(学習なし・帰属): D198 の一般の分 G = R̄ − 表(候補外 8 人の記憶の平均)を、発話の中の C1 の単位ごとの平均(G_st)と単位の中の動き(G_dy)・包絡と c0 に分けて実女声で鳴らす。
RRD<sfx> = 表 + S + G_dy・c0 = 男声の c0 + s0 + g0_dy・RRM<sfx> = 表 + S + G_st・c0 = 男声の c0 + s0 + g0_st・RRE<sfx> = 表 + S + G・c0 = 男声の c0 + s0(S = R_A − R̄)。
発話の中の単位ごとの平均を使う診断(製品の推論経路には置けない)。その他は d198_rr_contrast.py と同じ。事前登録 = results/conv_p0/d202_prereg.yaml
    uv run python d202_rr_split.py --s <scratchpad>/dev --tag diag_rr14c --sfx 14C --device cpu
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
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def anchors(n: int) -> list:
    base = [m for m in json.loads((ROOT / "data/realx/items_clone.json").read_text()) if not m["val"] and m["ref_sec"] >= 20.0]
    base = sorted({m["target"]: m for m in base}.values(), key=lambda m: m["target"])
    return [base[int(i)] for i in np.linspace(0, len(base) - 1, n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    ap.add_argument("--sfx", default="14C")
    ap.add_argument("--n_anchor", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    S = Path(a.s)
    dev = a.device
    od = S / "d139"; od.mkdir(exist_ok=True)
    FR = S / "rr_real"

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy(), "p4": d["p4"].float().numpy()}

    def mem_in(p: np.ndarray, env: np.ndarray) -> tuple:
        n = min(env.shape[1], 5000)
        rc0 = (env[0, :n] - RRm.C0_SIL) / 10
        return p[:, :n].astype(np.float32), np.concatenate([env[1:25, :n] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)

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

    FEAT = ROOT / "data/rr_feat"
    man = {it["stem"]: it for it in json.loads((FEAT / "real_fem/manifest.json").read_text())["items"]}
    amem = []
    for A in anchors(a.n_anchor):
        ds = [torch.load(FEAT / man[s]["pt"], weights_only=False) for s in A["ref_stems"] if s in man]
        p = np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1)
        env = np.concatenate([d["env"].float().numpy() for d in ds], 1)
        rp, rv = mem_in(p, env)
        with torch.no_grad():
            amem.append((net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1]))
        print("anchor", A["target"], len(ds), rp.shape[1], flush=True)

    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        T = C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)
        rp, rv = mem_in(r["p"], r["env"])
        tgt[t] = {"rp": rp, "rv": rv, "T": T, "mu": float(np.median(np.log(r["ft"][r["ft"] > 0]))), "p4": r["p4"][:, r["f1"] > 0].mean(1)}
    with torch.no_grad():
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if (od / f"{k}_RRE{a.sfx}.wav").exists():
                continue
            b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
            vo = o["f1"] > 0
            sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
            vM = b["f1"] > 0
            q = np.concatenate([b["p"], (b["e3"] - sM["e3"][:, None]) / 10, (b["c0f"] - sM["c0"])[None],
                                np.where(vM, np.log(np.maximum(b["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
            tg = tgt[ti]
            tab = (tg["T"] @ b["p"]).astype(np.float32)
            qt, tt = torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab)[None].to(dev)

            def run(mem, m: int) -> tuple:
                e, d0, _ = net(qt, tt, mem, torch.ones(1, m, dtype=torch.bool, device=dev))
                return e[0].cpu().numpy().astype(np.float64), d0[0].cpu().numpy().astype(np.float64)

            eA, dA = run(net.memory(torch.from_numpy(tg["rp"])[None].to(dev), torch.from_numpy(tg["rv"])[None].to(dev)), tg["rp"].shape[1])
            outs = [run(mm, m) for mm, m in amem]
            eG = np.mean([x[0] for x in outs], 0); dG = np.mean([x[1] for x in outs], 0)
            c0b = b["c0f"].astype(np.float64)
            f0o = np.where(vM, b["f1"] * np.exp(tg["mu"] - sM["lf"]), 0.0).astype(np.float64)
            per = (b["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
            G, g0 = eG - tab.astype(np.float64), dG
            u = b["u"][: G.shape[1]]
            Gst, g0st = np.zeros_like(G), np.zeros_like(g0)
            for v in np.unique(u):
                m = u == v
                Gst[:, m] = G[:, m].mean(1, keepdims=True); g0st[m] = g0[m].mean()
            Sx = tab.astype(np.float64) + eA - eG
            s0 = c0b + dA - dG
            conds = {f"RRD{a.sfx}": (s0 + g0 - g0st, Sx + G - Gst), f"RRM{a.sfx}": (s0 + g0st, Sx + Gst), f"RRE{a.sfx}": (s0, Sx + G)}
            for c, (cc, ee) in conds.items():
                sf.write(od / f"{k}_{c}.wav", render(cc, ee, f0o, per), N.SR)
            print(k, len(np.unique(u)), float(np.abs(G).mean()), float(np.abs(Gst).mean()), float(np.abs(G - Gst).mean()), float(np.abs(g0).mean()), float(np.abs(g0st).mean()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
