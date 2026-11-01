"""D215(学習なし・results/conv_p0/d215_prereg.yaml): R の話者の分を製品の経路の条件(G03S10Z + 較正 B の Rust の書き出し)に足して鳴らす。
RRPS<sfx> = 包絡 += (R_A − R̄)/10・c0 += dc0_A − dc̄0 / RRPT<sfx> = 包絡 += (R_A − R̄ + G_rel)/10・c0 は同じ。
R の差分の有声平均は操作者の参照の変換で測って引く(包絡は有声フレーム・c0 は全フレーム)。f0・周期性は製品のまま。Python の出力部 + 製品の等化 H(d141 と同じ・強さ 0.5)。
出力 = <s>/d139/pNNNN_tNNN_RRP{S,T}<sfx>.wav。R・候補外 8 人・問い・G_rel は d208_rr_natural.py と同じ。
D216(results/conv_p0/d216_prereg.yaml): --attr = S 形の包絡の差分だけ(<pfx>SE)・c0 の差分だけ(<pfx>SC)/ --base G00S10Z --pfx RRP0 = 操作者の残差 0 の土台(d114 の較正 B・Z と等化 H は同じ定義)。
diag_rr18p(results/diag_rr18p/prereg.yaml): --pin = R の入力の土台(tab)を製品の条件の包絡の行(RUSTB − 較正 B・操作者の参照は _cal の書き出し)にする。RRQS = 土台 + R_A − R̄・RRQA = 土台 + (R_A − 土台)。
D218(results/conv_p0/d218_prereg.yaml): --pint = --pin の T 形だけ。RRQT = 土台 + R_A − R̄ + 平均_k[R(土台_k) − 土台_k](土台_k = 候補外 k の登録の製品の包絡の行・pbase.py・front は d218_front.py)。
    uv run python d215_rr_product.py --s <dev> --tag diag_rr16c --sfx 16C
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm
from d114_env_bias_calib import BIN, CFG, EXTRA
from d208_rr_natural import anchors
from d214_path import rust

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sfx", default="16C")
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    ap.add_argument("--n_anchor", type=int, default=8)
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", default="G03S10Z")
    ap.add_argument("--pfx", default="RRP")
    ap.add_argument("--attr", action="store_true")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--pint", action="store_true")
    ap.add_argument("--anchor_en", default="/home/kojirotanaka/lightvc_scratch_20261013/d218_anchor_en")
    a = ap.parse_args()
    a.pin = a.pin or a.pint
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import scipy.signal as ss
    import train_rvoc as TR
    from d6_unit_cov import load48
    S = Path(a.s)
    dev = a.device
    od = S / "d139"; od.mkdir(exist_ok=True)
    rd = S / "d214"; rd.mkdir(exist_ok=True)
    FR = S / "rr_real"
    sr = N.SR
    work = S / "d85_work"
    cal = S / f"d114/_cal_{a.base}"
    shift = json.loads((S / f"d114/_c0shift_{a.base}.json").read_text())
    rk = "RUSTB" if a.base == "G03S10Z" else f"RUSTB_{a.base}"
    pairs = sorted((S / "d56").glob("p*_SRC.wav"))
    jobs = []
    for f in pairs:
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        if (rd / f"{k}_{rk}.f32").exists():
            continue
        cmd = [BIN, "artic", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3"]
        cmd += EXTRA + CFG[a.base] + ["--c0-shift", f"{shift[f'{pi}_{ti}']:.4f}", "-i", f, "-o", rd / f"{k}_{rk}.wav", "--env-bias", cal / f"bias_{pi:04d}_t{ti:03d}.f32"]
        jobs.append((cmd, str(rd / f"{k}_{rk}.f32")))
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(rust, jobs))
    print("rust dumps", len(jobs), flush=True)

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
    net = RRm.build(ck["cfg"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    sr_ = torch.load(ROOT / a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()

    @torch.no_grad()
    def render(c: np.ndarray) -> np.ndarray:
        f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0)
        cond = torch.from_numpy(c.T.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float64)

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

    def eqfir(g: np.ndarray) -> np.ndarray:
        g = np.convolve(np.pad(g, 1, mode="edge"), np.ones(3) / 3, mode="valid")
        g = np.clip(a.strength * (g - np.median(g)), -12, 12)
        fr = np.r_[0.0, cf / (sr / 2), 1.0]
        gl = 10 ** (np.r_[g[0], g, g[-1]] / 20)
        return ss.minimum_phase(ss.firwin2(1023, np.clip(fr, 0, 1), gl ** 2, fs=2.0), method="homomorphic")

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

    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        tgt[t] = {"rp": mem_in(r["p"], r["env"])[0], "rv": mem_in(r["p"], r["env"])[1], "T": C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)}

    @torch.no_grad()
    def deltas(src: dict, sM: dict, tg: dict, tabx: np.ndarray | None = None, gtab: list | None = None) -> dict:
        vM = src["f1"] > 0
        q = np.concatenate([src["p"], (src["e3"] - sM["e3"][:, None]) / 10, (src["c0f"] - sM["c0"])[None],
                            np.where(vM, np.log(np.maximum(src["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
        tab = (tg["T"] @ src["p"]).astype(np.float32) if tabx is None else tabx.astype(np.float32)
        qt, tt = torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab)[None].to(dev)

        def run(mem, m: int, tb) -> tuple:
            e, d0, _ = net(qt, tb, mem, torch.ones(1, m, dtype=torch.bool, device=dev))
            return e[0].cpu().numpy().astype(np.float64), d0[0].cpu().numpy().astype(np.float64)

        eA, dA = run(net.memory(torch.from_numpy(tg["rp"])[None].to(dev), torch.from_numpy(tg["rv"])[None].to(dev)), tg["rp"].shape[1], tt)
        outs = [run(mm, m, tt) for mm, m, _ in amem]
        eG = np.mean([x[0] for x in outs], 0); dG = np.mean([x[1] for x in outs], 0)
        if gtab is None:
            Grel = np.mean([run(mm, m, torch.from_numpy((Tk @ src["p"]).astype(np.float32))[None].to(dev))[0] - (Tk @ src["p"]) for mm, m, Tk in amem], 0)
        else:
            Grel = np.mean([run(mm, m, torch.from_numpy(g.astype(np.float32))[None].to(dev))[0] - g.astype(np.float32) for (mm, m, _), g in zip(amem, gtab)], 0)
        return {"S": eA - eG, "T": eA - eG + Grel, "A": eA - tab, "c0": dA - dG, "c0A": dA, "v": vM}

    names = {"E": f"{a.pfx}SE{a.sfx}", "C": f"{a.pfx}SC{a.sfx}"} if a.attr else {"T": f"RRQT{a.sfx}"} if a.pint else {"S": f"RRQS{a.sfx}", "A": f"RRQA{a.sfx}"} if a.pin else {"S": f"{a.pfx}S{a.sfx}", "T": f"{a.pfx}T{a.sfx}"}

    def fit(x: np.ndarray, m: int) -> np.ndarray:
        assert abs(x.shape[1] - m) <= 2, (x.shape, m)
        return np.pad(x, ((0, 0), (0, max(0, m - x.shape[1]))), mode="edge")[:, :m]
    tl: dict = {}
    log = []
    chkmax: list = []
    if a.pint:
        import pbase as PB
        Wp = PB.warp_matrix(1.3)
        aen = [PB.load_enroll(Path(a.anchor_en) / A["target"]) for A in anchors(a.n_anchor)]
    for f in pairs:
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
        h = eqfir(tl[ti] - ltas(load48(cal / f"op{pi:04d}_t{ti:03d}.wav").astype(np.float64)))
        C = np.fromfile(S / "d214" / f"{k}_{rk}.f32", "<f4").reshape(-1, 31).astype(np.float64)
        if a.pin:
            bv = np.fromfile(cal / f"bias_{pi:04d}_t{ti:03d}.f32", "<f4").astype(np.float64)
            Co = np.fromfile(cal / f"op{pi:04d}_t{ti:03d}.f32", "<f4").reshape(-1, 31).astype(np.float64)
            gop = gsr = None
            if a.pint:
                opn = PB.load_enroll(work / f"op_{pi:04d}")
                chk = PB.base_env(*PB.read_front(rd / f"{k}_FRONT.f32", opn["k"]), (en_t := PB.load_enroll(work / f"en_{ti:03d}")), opn, Wp, PB.color(en_t, opn, Wp))
                cho = PB.base_env(*PB.read_front(rd / f"op{pi:04d}_FRONT.f32", opn["k"]), en_t, opn, Wp, PB.color(en_t, opn, Wp))
                for x, ref in ((chk, C[:, 1:25] - bv[None] * (C[:, 26:27] > 0.5)), (cho, Co[:, 1:25])):
                    mx = float(np.abs(x[: min(len(x), len(ref))] - ref[: min(len(x), len(ref))]).max()); chkmax.append(mx)
                    assert abs(len(x) - len(ref)) <= 2 and mx < 1e-3, (k, mx, x.shape, ref.shape)
                gb = lambda fr, m: [fit(PB.base_env(*PB.read_front(fr, en["k"]), en, opn, Wp, PB.color(en, opn, Wp)).T * 10, m) for en in aen]
                gop, gsr = gb(rd / f"op{pi:04d}_FRONT.f32", o["p"].shape[1]), gb(rd / f"{k}_FRONT.f32", b["p"].shape[1])
            dop = deltas(o, sM, tg, fit(Co[:, 1:25].T * 10, o["p"].shape[1]), gop)
            dsr = deltas(b, sM, tg, fit((C[:, 1:25] - bv[None] * (C[:, 26:27] > 0.5)).T * 10, b["p"].shape[1]), gsr)
        else:
            dop = deltas(o, sM, tg)
            dsr = deltas(b, sM, tg)
        n = min(len(C), dsr["c0"].shape[0])
        assert abs(len(C) - dsr["c0"].shape[0]) <= 2, (len(C), dsr["c0"].shape)
        mc0 = float(dop["c0"].mean())
        vC = C[:n, 26] > 0.5
        for form, nm in names.items():
            fm = "S" if form in "EC" else form
            me = dop[fm][:, dop["v"]].mean(1)
            ck = "c0A" if fm == "A" else "c0"
            c = C.copy()
            if form != "E":
                c[:n, 0] += dsr[ck][:n] - float(dop[ck].mean())
            if form != "C":
                c[:n, 1:25] += dsr[fm][:, :n].T / 10
                c[:n][vC, 1:25] -= me / 10
            y = ss.lfilter(h, 1.0, render(c))
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), sr)
            log.append({"key": k, "form": form, "c0_mean_op": round(mc0, 4), "env_mean_op_l2": round(float(np.linalg.norm(me)), 4),
                        "dc0_rms": round(float(np.sqrt(np.mean((dsr["c0"] - mc0) ** 2))), 4), "denv_rms": round(float(np.sqrt(np.mean(dsr[fm] ** 2))), 4)})
        print(k, log[-1], flush=True)
    if log:
        (S / f"d215_consts_{a.sfx}{'' if a.base == 'G03S10Z' else '_' + a.base}{'_attr' if a.attr else ''}{'_pint' if a.pint else '_pin' if a.pin else ''}.json").write_text(json.dumps(log, indent=1))
        if chkmax:
            print("base check max|diff|", max(chkmax), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
