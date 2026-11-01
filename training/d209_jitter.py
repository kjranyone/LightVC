"""D209(学習なし・数値だけ): d208_rr_natural.py と同じ R の形を作り(音声は作らない)、有声の隣り合うフレームの包絡の揺れ J = 中央値_t mean_d |x_t − x_{t−1}| を
表・R_A・S 形(表 + R_A − R̄)・T 形(S 形 + G_rel)で比べる。c0 は記述(s0 と男声の c0f)。事前登録 = results/conv_p0/d209_prereg.yaml
    uv run python d209_jitter.py --s <scratchpad>/dev --tag diag_rr14c --out ../results/conv_p0/d209dev.json
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import scipy.signal as ss
    import train_rvoc as TR
    from d6_unit_cov import load48
    S = Path(a.s)
    dev = a.device
    od = S / "d139"; od.mkdir(exist_ok=True)
    FR = S / "rr_real"
    sr = N.SR

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

    cf = 1000.0 * 2.0 ** (np.arange(-13, 15) / 3.0)
    cf = cf[(cf >= 60) & (cf <= 20000)]

    def ltas(x: np.ndarray) -> np.ndarray:
        n = len(x) // 240
        e = 10 * np.log10((x[: n * 240].reshape(n, 240) ** 2).mean(1) + 1e-12)
        act = e > e.max() - 40
        fq, _, Zs = ss.stft(x, sr, nperseg=2048, noverlap=2048 - 240)
        P = np.abs(Zs[:, :n]) ** 2
        P = P[:, act[: P.shape[1]]].mean(1)
        L = 10 * np.log10(np.array([P[(fq >= c * 2 ** (-1 / 6)) & (fq < c * 2 ** (1 / 6))].sum() + 1e-20 for c in cf]))
        return L - L.mean()

    def eqfir(g: np.ndarray) -> tuple:
        g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        g = np.clip(a.strength * (g - np.median(g)), -12, 12)
        fr = np.r_[0.0, cf / (sr / 2), 1.0]
        gl = 10 ** (np.r_[g[0], g, g[-1]] / 20)
        h = ss.minimum_phase(ss.firwin2(1023, np.clip(fr, 0, 1), gl ** 2, fs=2.0), method="homomorphic")
        _, gdel = ss.group_delay((h, 1), w=np.linspace(50, 1000, 50), fs=sr)
        return h, float(np.median(gdel)) / sr * 1000, g

    FEAT = ROOT / "data/rr_feat"
    man = {it["stem"]: it for it in json.loads((FEAT / "real_fem/manifest.json").read_text())["items"]}
    amem = []
    for A in anchors(a.n_anchor):
        ds = [torch.load(FEAT / man[s]["pt"], weights_only=False) for s in A["ref_stems"] if s in man]
        p = np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1)
        env = np.concatenate([d["env"].float().numpy() for d in ds], 1)
        rp, rv = mem_in(p, env)
        uk = np.concatenate([d["pk"][:, 0].long().numpy() for d in ds])
        Tk = C0.table(env[1:25], uk, RRm.K, Cb).astype(np.float32)
        with torch.no_grad():
            amem.append((net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1], Tk))
        print("anchor", A["target"], len(ds), rp.shape[1], flush=True)

    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        T = C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)
        rp, rv = mem_in(r["p"], r["env"])
        vt = r["ft"] > 0
        tgt[t] = {"rp": rp, "rv": rv, "T": T, "mu": float(np.median(np.log(r["ft"][vt]))), "p4": r["p4"][:, r["f1"] > 0].mean(1),
                  "c0T": float(((r["env"][0, vt] - RRm.C0_SIL) / 10).mean()), "eT": r["env"][1:25, vt].mean(1).astype(np.float64)}

    @torch.no_grad()
    def convert(src: dict, o: dict, sM: dict, tg: dict) -> tuple:
        vo = o["f1"] > 0
        vM = src["f1"] > 0
        q = np.concatenate([src["p"], (src["e3"] - sM["e3"][:, None]) / 10, (src["c0f"] - sM["c0"])[None],
                            np.where(vM, np.log(np.maximum(src["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
        tab = (tg["T"] @ src["p"]).astype(np.float32)
        qt, tt = torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab)[None].to(dev)

        def run(mem, m: int, tb) -> tuple:
            e, d0, _ = net(qt, tb, mem, torch.ones(1, m, dtype=torch.bool, device=dev))
            return e[0].cpu().numpy().astype(np.float64), d0[0].cpu().numpy().astype(np.float64)

        eA, dA = run(net.memory(torch.from_numpy(tg["rp"])[None].to(dev), torch.from_numpy(tg["rv"])[None].to(dev)), tg["rp"].shape[1], tt)
        outs = [run(mm, m, tt) for mm, m, _ in amem]
        eG = np.mean([x[0] for x in outs], 0); dG = np.mean([x[1] for x in outs], 0)
        rel = []
        for mm, m, Tk in amem:
            tk = (Tk @ src["p"]).astype(np.float32)
            rel.append(run(mm, m, torch.from_numpy(tk)[None].to(dev))[0] - tk)
        Grel = np.mean(rel, 0)
        Sx = tab.astype(np.float64) + eA - eG
        s0 = src["c0f"].astype(np.float64) + dA - dG
        f0o = np.where(vM, src["f1"] * np.exp(tg["mu"] - sM["lf"]), 0.0).astype(np.float64)
        per = (src["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
        return {"S": (s0, Sx), "T": (s0, Sx + Grel), "A": (s0, eA)}, f0o, per, vM

    def jit(x: np.ndarray, v: np.ndarray) -> float:
        d = np.abs(np.diff(x, axis=-1))
        d = d.mean(0) if d.ndim == 2 else d
        m = v[1:] & v[:-1]
        return float(np.median(d[m]))

    rows = []
    for f in sorted((S / "d56").glob("p*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
        vo = o["f1"] > 0
        sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
        tg = tgt[ti]
        fb_, _, _, vM = convert(b, o, sM, tg)
        tab = (tg["T"] @ b["p"]).astype(np.float64)
        s0, Sx = fb_["S"]; _, Tx = fb_["T"]
        eA = fb_["A"][1]
        rows.append({"key": f.stem[:-4], "J_tab": jit(tab, vM), "J_RA": jit(eA, vM), "J_S": jit(Sx, vM), "J_T": jit(Tx, vM), "J_SminusTab": jit(Sx - tab, vM),
                     "J_c0_s0": jit(s0, vM), "J_c0_male": jit(b["c0f"].astype(np.float64), vM)})
        print(rows[-1]["key"], {k: round(v, 4) for k, v in rows[-1].items() if k != "key"}, flush=True)
    med = {k: round(float(np.median([r[k] for r in rows])), 4) for k in rows[0] if k != "key"}
    ratio = [r["J_S"] / r["J_T"] for r in rows]
    rep = {"n": len(rows), "median": med, "ratio_S_over_T_median": round(float(np.median(ratio)), 4), "ratio_S_over_T_p10_p90": np.round(np.percentile(ratio, [10, 90]), 4).tolist(), "rows": rows}
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
