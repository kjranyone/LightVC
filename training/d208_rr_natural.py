"""D208(学習なし): 製品の経路の自然さの手当て(登録の定数 = c0 のずらし Z・包絡の較正 B・出力の半分の等化 H)を R 系の出力(d139 の経路)に同じ定義で載せる。
各組 (操作者 p, 目標 A) で、操作者の登録の参照 op_PPPP を同じ R の形で変換した包絡 E_op・c0 C_op から:
  Z = 目標の参照の有声の教師の c0n の平均((env0 − C0_SIL)/10)− C_op の有声の平均・B = 目標の参照の有声の教師の包絡 c1..24 の平均 − E_op の有声の平均(評価の発話の有声フレームに足す)・
  H = 目標の参照の 1/3 oct の長時間スペクトル − Z だけを載せた操作者の参照の変換音の同じ量 → 平滑・中央値を引く・× 0.5・±12dB → 最小位相 FIR 1023 タップ(d141_out_eq.py と同じ)。
出力 = <s>/d139/pNNNN_tNNN_RRSN<sfx>.wav(話者の分だけの形)・RRTN<sfx>.wav(+ 相対の一般の分 G_rel)。R・候補外 8 人・問いは d203_rr_rel.py と同じ。事前登録 = results/conv_p0/d208_prereg.yaml
    uv run python d208_rr_natural.py --s <scratchpad>/dev --tag diag_rr14c --sfx 14C
    uv run python d208_rr_natural.py --s <scratchpad>/dev --tag diag_rr14c --sfx 14C --seed 1(D210・T 形だけを乱数 1 で・RRTN14Cs1)
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
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0, help="出力部の励振の乱数(0 以外は出力名に s<seed>・D210)")
    ap.add_argument("--attr", action="store_true", help="D213: T 形の部品を R なしに替えた RRTNC(c0 = 男声のみ)・RRTNE(包絡 = 表のみ)・RRTNB(両方)だけを書く")
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
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(a.seed)))[0].cpu().numpy()[N.DELAY:]
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
        c0m = src["c0f"].astype(np.float64)
        return {"S": (s0, Sx), "T": (s0, Sx + Grel), "TC": (c0m, Sx + Grel), "TE": (s0, tab.astype(np.float64)), "TB": (c0m, tab.astype(np.float64))}, f0o, per, vM

    names = {"S": f"RRSN{a.sfx}", "T": f"RRTN{a.sfx}"} if a.seed == 0 else {"T": f"RRTN{a.sfx}s{a.seed}"}
    if a.attr:
        names = {"TC": f"RRTNC{a.sfx}", "TE": f"RRTNE{a.sfx}", "TB": f"RRTNB{a.sfx}"}
    tl: dict = {}
    log = []
    for f in sorted((S / "d56").glob("p*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        if all((od / f"{k}_{nm}.wav").exists() for nm in names.values()):
            continue
        b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
        vo = o["f1"] > 0
        sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
        tg = tgt[ti]
        if ti not in tl:
            tl[ti] = ltas(load48(S / "d37/_work" / f"ref_{ti:03d}.wav").astype(np.float64))
        fo, f0op, perop, vop = convert(o, o, sM, tg)
        fb_, f0o, per, vM = convert(b, o, sM, tg)
        for form, nm in names.items():
            c0op, eop = fo[form]
            z = tg["c0T"] - float(c0op[vop].mean())
            bias = tg["eT"] - eop[:, vop].mean(1)
            ycal = render(c0op + z, eop, f0op, perop)
            h, gd, g = eqfir(tl[ti] - ltas(ycal.astype(np.float64)))
            c0, env = fb_[form]
            env2 = env.copy()
            env2[:, vM] += bias[:, None]
            y = ss.lfilter(h, 1.0, render(c0 + z, env2, f0o, per).astype(np.float64))
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), sr)
            log.append({"key": k, "form": form, "z": round(z, 4), "bias_l2": round(float(np.linalg.norm(bias)), 4), "eq_db_range": [round(float(g.min()), 2), round(float(g.max()), 2)], "gd_ms": round(gd, 3)})
        print(k, {x["form"]: (x["z"], x["bias_l2"], x["gd_ms"]) for x in log[-2:]}, flush=True)
    if log:
        (S / f"d208_consts_{a.sfx}{'' if a.seed == 0 else f's{a.seed}'}{'_attr' if a.attr else ''}.json").write_text(json.dumps(log, indent=1))
        for form in names:
            zs = np.array([x["z"] for x in log if x["form"] == form]); gds = np.array([x["gd_ms"] for x in log if x["form"] == form])
            print(form, "z median", round(float(np.median(zs)), 3), "p10/p90", np.round(np.percentile(zs, [10, 90]), 3).tolist(), "gd ms median", round(float(np.median(gds)), 3), "p95", round(float(np.percentile(gds, 95)), 3), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
