"""D203(学習なし・帰属): 一般の分を相対で作る。候補外 8 人(D198 と同じ)を各自の表 T_k と各自の記憶で読み、G_rel = mean_k (R(問い, T_k の表, k の記憶) − T_k の表)。
RRT<sfx> = A の表 + S + G_rel・c0 = 男声の c0 + s0(S = R_A − R̄・R̄ は A の表と候補外の記憶 = D198 と同じ)。その他は d202_rr_split.py と同じ。事前登録 = results/conv_p0/d203_prereg.yaml
    uv run python d203_rr_rel.py --s <scratchpad>/dev --tag diag_rr14c --sfx 14C --device cpu
    uv run python d203_rr_rel.py --s <scratchpad>/dev --tag diag_rr14c --sfx 14C --n_rel 32 --osfx RRT32(D204・results/conv_p0/d204_prereg.yaml)
    uv run python d203_rr_rel.py --s <scratchpad>/dev --tag diag_rr14c --sfx 14C --map --osfx RRTW(D205・results/conv_p0/d205_prereg.yaml の amendment)
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
    ap.add_argument("--n_rel", type=int, default=0, help="> 0 で G_rel だけをこの人数の候補外で作る(S は --n_anchor のまま・D204)")
    ap.add_argument("--osfx", default="", help="出力の名前(既定 RRT<sfx>)")
    ap.add_argument("--map", action="store_true", help="k のずれを k の表 → A の表の周波数の伸縮 W(α)(1 パラメータ・声道長)で A の尺度へ写してから平均する(G_map・D205)")
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
        uk = np.concatenate([d["pk"][:, 0].long().numpy() for d in ds])
        Tk = C0.table(env[1:25], uk, RRm.K, Cb).astype(np.float32)
        with torch.no_grad():
            amem.append((net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1], Tk, np.bincount(uk, minlength=RRm.K)))
        print("anchor", A["target"], len(ds), rp.shape[1], flush=True)

    rmem = amem
    if a.n_rel > 0:
        rmem = []
        for A in anchors(a.n_rel):
            ds = [torch.load(FEAT / man[s]["pt"], weights_only=False) for s in A["ref_stems"] if s in man]
            p = np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1)
            env = np.concatenate([d["env"].float().numpy() for d in ds], 1)
            rp, rv = mem_in(p, env)
            uk = np.concatenate([d["pk"][:, 0].long().numpy() for d in ds])
            Tk = C0.table(env[1:25], uk, RRm.K, Cb).astype(np.float32)
            with torch.no_grad():
                rmem.append((net.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1], Tk, np.bincount(uk, minlength=RRm.K)))
        print("rel anchors", len(rmem), flush=True)
    on = a.osfx or f"RRT{a.sfx}"
    Mc: dict = {}

    import librosa
    import zsvc as Z
    D25 = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)[:25]
    fc = librosa.mel_frequencies(n_mels=N.N_MEL + 2, fmin=0.0, fmax=N.FMAX)[1:-1]
    ALPHAS = np.geomspace(0.75, 1.0 / 0.75, 61)

    def warpmat(al: float) -> np.ndarray:
        src = np.clip(fc / al, fc[0], fc[-1])
        i1 = np.clip(np.searchsorted(fc, src), 1, len(fc) - 1)
        w = (src - fc[i1 - 1]) / (fc[i1] - fc[i1 - 1])
        I = np.zeros((len(fc), len(fc)))
        I[np.arange(len(fc)), i1 - 1] = 1 - w
        I[np.arange(len(fc)), i1] = w
        return (D25 @ I @ D25.T)[1:, 1:]

    WM = {al: warpmat(al) for al in ALPHAS}
    assert np.abs(warpmat(1.0) - np.eye(24)).max() < 1e-6

    def fitmap(TA: np.ndarray, nA: np.ndarray, Tk: np.ndarray, nk: np.ndarray) -> tuple:
        j = (nA > 0) & (nk > 0)
        w = np.sqrt(np.minimum(nA, nk)[j].astype(np.float64))
        best = (np.inf, 1.0)
        for al, W in WM.items():
            R = TA[:, j] - W @ Tk[:, j]
            R = R - (R * w).sum(1, keepdims=True) / w.sum()
            e = float(((R ** 2).sum(0) * w).sum())
            best = min(best, (e, al))
        return WM[best[1]], best[1]
    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        T = C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)
        rp, rv = mem_in(r["p"], r["env"])
        tgt[t] = {"n": np.bincount(r["u"], minlength=RRm.K), "rp": rp, "rv": rv, "T": T, "mu": float(np.median(np.log(r["ft"][r["ft"] > 0]))), "p4": r["p4"][:, r["f1"] > 0].mean(1)}
    with torch.no_grad():
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if (od / f"{k}_{on}.wav").exists():
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
            outs = [run(mm, m) for mm, m, *_ in amem]
            eG = np.mean([x[0] for x in outs], 0); dG = np.mean([x[1] for x in outs], 0)
            c0b = b["c0f"].astype(np.float64)
            f0o = np.where(vM, b["f1"] * np.exp(tg["mu"] - sM["lf"]), 0.0).astype(np.float64)
            per = (b["p4"] - o["p4"][:, vo].mean(1)[:, None] + tg["p4"][:, None]).astype(np.float64)
            rel = []
            for ki, (mm, m, Tk, nk) in enumerate(rmem):
                tk = (Tk @ b["p"]).astype(np.float32)
                e_, _, _ = net(qt, torch.from_numpy(tk)[None].to(dev), mm, torch.ones(1, m, dtype=torch.bool, device=dev))
                dv = e_[0].cpu().numpy().astype(np.float64) - tk
                if a.map:
                    if (ti, ki) not in Mc:
                        Mc[(ti, ki)] = fitmap(tg["T"], tg["n"], Tk, nk)
                    dv = Mc[(ti, ki)][0] @ dv
                rel.append(dv)
            Grel = np.mean(rel, 0)
            Sx = tab.astype(np.float64) + eA - eG
            s0 = c0b + dA - dG
            conds = {on: (s0, Sx + Grel)}
            for c, (cc, ee) in conds.items():
                sf.write(od / f"{k}_{c}.wav", render(cc, ee, f0o, per), N.SR)
            print(k, float(np.abs(eG - tab).mean()), float(np.abs(Grel).mean()), float(np.abs(eG - tab - Grel).mean()), flush=True)
    if a.map and Mc:
        st_ = [{"t": t, "k": k, "alpha": float(al), "dist_I": float(np.linalg.norm(M - np.eye(M.shape[0])))} for (t, k), (M, al) in sorted(Mc.items())]
        (S / f"d205_maps_{on}.json").write_text(json.dumps(st_, indent=1))
        al_ = np.array([x["alpha"] for x in st_])
        print("maps", len(st_), "alpha mean", float(al_.mean()), "min", float(al_.min()), "max", float(al_.max()), "at_edge", int(((al_ <= ALPHAS[0] + 1e-9) | (al_ >= ALPHAS[-1] - 1e-9)).sum()), "dist_I", float(np.mean([x["dist_I"] for x in st_])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
