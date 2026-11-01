"""D579(推論だけ): 開発の組(dev2 / dev3・D570 と同じ組)で、製品 P57S0 に既存の学習した読み R(diag_rr19L・S 形)の話者の分を足し、固定の半音 k で鳴らす。規則は OVERNIGHT の D579。
段 rr: R の差分 = R(目標の記憶)− 候補外 8 人の記憶の平均(d220 / D577 と同じ)。問い = 操作者の SRC 8 s の C1 の事後(上位 16)・F3 の包絡・F2 の c0・F1 の f0(操作者の参照の登録の平均を引く)・
表 = 目標の参照の CheapTrick の表 × 操作者の事後・記憶 = 目標の参照(<組>/d37/_work/ref_TTT.wav・CheapTrick c0〜c24・25 s まで)。→ <組>/d579_rr/<pNNNN_tTTT>.npz。
段 render: D570 の OURS_k(e2e/eval_<組>_pre_nof0 の書き出し・前置き 12 s・c0 の水準ずらし・操作者の遅い揺らぎ 0・有声の log f0 + k 半音)の条件の包絡の行に
z = dS / 10 を前置きの後ろのフレームだけ足す(D578 と同じ口)→ <組>/d570/OURS<k>R/<key>_OURS<k>R.wav(8 s・ピーク 0.9)。
    uv run python d579_rr_dev.py --sets dev2 dev3 --stage rr
    uv run python d579_rr_dev.py --sets dev2 dev3 --stage render
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
CKPT = ROOT / "results/diag_ppgvoc57/product_002500_m0.pt"
PRE = 12.0
KS = (5, 7, 12)


def top16(p: np.ndarray) -> np.ndarray:
    idx = np.argsort(-p, axis=0)[:16]
    out = np.zeros_like(p)
    np.put_along_axis(out, idx, np.take_along_axis(p, idx, 0), 0)
    return out


def pairs_of(S: Path) -> list:
    return [r for r in json.loads((S / "devset.json").read_text())["pairs"]
            if (S / "d570" / "OURS7" / f"p{r['pi']:04d}_t{r['ti']:03d}_OURS7.wav").exists()]


def stage_rr(sets: list, dev: str, tag: str, gpath: str = "", rrdir: str = "d579_rr", mem_only: bool = False, mem_f1: bool = False, limit: int = 0) -> None:
    import c1_content as CC
    import conv_c0 as C0
    import f0est as FE
    import f0hi as FH
    import lvl as LV
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_c3 as C3
    import train_rr as RRm
    import train_rvoc as TR
    from d208_rr_natural import anchors
    from d6_unit_cov import load48
    torch.set_num_threads(4)
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    ld = lambda p: torch.load(ROOT / p, map_location="cpu", weights_only=False)
    s1 = ld("results/f0est3/last.pt"); f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(s1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    s2 = ld("results/f2_2/last.pt"); f2 = LV.Lvl(s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev); f2.load_state_dict(s2["net"]); f2.eval()
    s3 = ld("results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    sc = ld("results/c1_1/last.pt"); c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev); c1.load_state_dict(sc["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt", map_location="cpu").cpu().numpy()
    K = Cb.shape[0]
    ckr = ld(f"results/{tag}/last.pt")
    rnet = RRm.build(ckr["cfg"]).to(dev).eval()
    rnet.load_state_dict(ckr["net"])

    @torch.no_grad()
    def front(x: np.ndarray) -> dict:
        n = len(x) // N.HOP
        x = x[: n * N.HOP].astype(np.float32)
        xt = torch.from_numpy(x)[None].to(dev)
        v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
        f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
        m0 = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1))
        c0f = C3.causal_avg(f2(m0)[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
        e3 = 10.0 * f3(m0)[0].cpu().numpy()[1:25, :n].astype(np.float64)
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        ft, _ = FH.teacher_f0(x, n)
        return {"x": x, "n": n, "f0": pad1(f0f), "c0": pad1(c0f), "e3": pad(e3), "p": pad(p), "u": pad(p).argmax(0), "ft": pad1(ft[:n].astype(np.float64))}

    def cheap25(fr: dict) -> np.ndarray:
        xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, fr["f0"] if mem_f1 else fr["ft"]), ((0, 0), (1, 1)), mode="edge")
        return (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :fr["n"]][:25].astype(np.float64)

    @torch.no_grad()
    def rmem(p_: np.ndarray, env: np.ndarray) -> tuple:
        n_ = min(env.shape[1], 5000)
        rc0 = (env[0, :n_] - RRm.C0_SIL) / 10
        rp = p_[:, :n_].astype(np.float32)
        rv = np.concatenate([env[1:25, :n_] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
        return rnet.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1]

    FEAT = ROOT / "data/rr_feat"
    man = {it_["stem"]: it_ for it_ in json.loads((FEAT / "real_fem/manifest.json").read_text())["items"]}
    amem = []
    anc = anchors(8)
    for A_ in anc:
        ds = [torch.load(FEAT / man[s_]["pt"], weights_only=False) for s_ in A_["ref_stems"] if s_ in man]
        amem.append(rmem(np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1), np.concatenate([d["env"].float().numpy() for d in ds], 1)))
    anc_t = {A_["target"] for A_ in anc}
    gnet, gkv = None, None
    if gpath:
        import copy
        sg = torch.load(gpath, map_location="cpu", weights_only=False)
        gnet = copy.deepcopy(rnet)
        gnet.load_state_dict(sg["g"])
        gnet.eval()
        gkv = [z.to(dev) for z in sg["kv"]]

    @torch.no_grad()
    def rr_delta(fm: dict, sM: dict, tab: np.ndarray, mA: tuple) -> tuple:
        vM = fm["f0"] > 0
        q = np.concatenate([top16(fm["p"]), (fm["e3"] - sM["e3"][:, None]) / 10, (fm["c0"] - sM["c0"])[None],
                            np.where(vM, np.log(np.maximum(fm["f0"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float64)], 0).astype(np.float32)
        qt, tt = torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab.astype(np.float32))[None].to(dev)

        def run(mm: tuple) -> tuple:
            e_, d0_, _ = rnet(qt, tt, mm[0], torch.ones(1, mm[1], dtype=torch.bool, device=dev))
            return e_[0].cpu().numpy().astype(np.float64), d0_[0].cpu().numpy().astype(np.float64)
        eA, dA = run(mA)
        if gnet is not None:
            L = len(gkv) // 2
            e_, d0_, _ = gnet(qt, tt, [(gkv[2 * l], gkv[2 * l + 1]) for l in range(L)], torch.ones(1, gkv[0].shape[-1], dtype=torch.bool, device=dev))
            return eA - e_[0].cpu().numpy().astype(np.float64), dA - d0_[0].cpu().numpy().astype(np.float64)
        outs = [run(mm) for mm in amem]
        return eA - np.mean([o_[0] for o_ in outs], 0), dA - np.mean([o_[1] for o_ in outs], 0)

    for s in sets:
        S = H / s
        od = S / rrdir
        od.mkdir(exist_ok=True)
        ops: dict = {}
        tgs: dict = {}
        n_done = 0
        for r in (pairs_of(S)[:limit] if limit else pairs_of(S)):
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if (od / f"{k}.npz").exists() and not mem_only:
                continue
            if mem_only and (S / "rr_mem" / f"en_{r['ti']:03d}" / "mem_v1.f32").exists():
                continue
            if r["pi"] not in ops:
                opw = S / "d56" / "_work" / f"op_{r['pi']:04d}.wav"
                fr = front(load48(opw).astype(np.float32))
                v = fr["f0"] > 0
                ops[r["pi"]] = {"e3": fr["e3"][:, v].mean(1), "c0": float(fr["c0"][v].mean()), "lf": float(np.median(np.log(fr["f0"][v])))}
            if r["ti"] not in tgs:
                assert r.get("target") not in anc_t
                fr = front(load48(S / "d37" / "_work" / f"ref_{r['ti']:03d}.wav").astype(np.float32))
                env = cheap25(fr)
                T = C0.table(env[1:25], fr["u"], K, Cb)
                tgs[r["ti"]] = (T, rmem(top16(fr["p"]), env))
                if mem_only:
                    md = S / "rr_mem" / f"en_{r['ti']:03d}"
                    md.mkdir(parents=True, exist_ok=True)
                    for l, (kk_, vv_) in enumerate(tgs[r["ti"]][1][0]):
                        kk_[0].cpu().numpy().astype("<f4").tofile(md / f"mem_k{l}.f32")
                        vv_[0].cpu().numpy().astype("<f4").tofile(md / f"mem_v{l}.f32")
            if mem_only:
                continue
            fm = front(load48(S / "d56" / f"{k}_SRC.wav").astype(np.float32)[: 8 * N.SR])
            T, mA = tgs[r["ti"]]
            dS, dC = rr_delta(fm, ops[r["pi"]], T @ fm["p"], mA)
            np.savez(od / f"{k}.npz", dS=dS.astype(np.float32), dC=dC.astype(np.float32))
            n_done += 1
        print(s, "rr deltas", n_done, flush=True)


def stage_render(sets: list, dev: str, gain: float = 1.0, ks: tuple = KS, ckpt: str = str(CKPT), wsfx: str = "", rrdir: str = "d579_rr", osfx: str = "", h1fix: float = 0.0, exc_h1: float = 0.0) -> None:
    import nvoc as N
    import train_c1 as TC
    import exc_h1 as X
    import train_ppgvoc as P
    import train_rvoc as TR
    from d6_unit_cov import load48
    X.install()
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 0.0
    P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    n0 = int(PRE * N.SR)
    f0 = n0 // N.HOP
    for s in sets:
        S = P.H / s
        ED = P.H / "e2e" / f"eval_{s}_pre_nof0"
        od = S / "d570"
        n_done = 0
        for r in pairs_of(S):
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            sx = "" if gain == 1.0 else f"g{int(round(gain * 100))}"
            outs = {kk: od / f"OURS{kk}R{sx}{wsfx}{osfx}" / f"{k}_OURS{kk}R{sx}{wsfx}{osfx}.wav" for kk in ks}
            if all(o.exists() for o in outs.values()):
                continue
            rr = np.load(S / rrdir / f"{k}.npz")
            c, lg = P.load_dump(ED, k)
            pre = c[:2400, 0]
            mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
            c = c.copy()
            c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
            vv = c[:, 26] > 0.5
            T = len(c)
            m = min(rr["dS"].shape[1], T - f0)
            assert abs((T - f0) - rr["dS"].shape[1]) <= 3, (k, T - f0, rr["dS"].shape[1])
            z = np.zeros((24, T), np.float32)
            z[:, f0:f0 + m] = gain * rr["dS"][:, :m] / 10
            cvf = P.cvin_frames(torch.from_numpy(TC.prime(load48(S / "d467" / f"{k}.wav").astype(np.float32)))[None].to(dev), len(lg))
            en = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
            P.H1FIX.update(on=h1fix > 0, sigma=h1fix, taper=False, ft=float(np.exp(json.loads((S / "d85_work" / f"en_{r['ti']:03d}" / "enroll.json").read_text())["mu"])))
            X.EXH1.update(on=exc_h1 > 0, gain_db=exc_h1, ft=P.H1FIX["ft"])
            todo = [(kk, o, z) for kk, o in outs.items() if not o.exists()]
            if n_done == 0:
                todo.append((7, None, np.zeros_like(z)))
            for kk, o, zz in todo:
                ck = c.copy()
                ck[:, 25] = np.where(vv, c[:, 25] + kk * np.log(2) / 12, c[:, 25])
                if P.C0LOW["on"]:
                    ck = P.c0low_np(ck)
                with torch.no_grad():
                    p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                    ct = torch.from_numpy(ck.T.copy()).float().to(dev)[None]
                    stt = P.spk_tensor(en, dev)
                    if P.PERDYN["on"] and isinstance(stt, tuple):
                        stt = (stt[0], ct[:, 27:31])
                    cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, torch.from_numpy(zz)[None].to(dev), cvf=cvf)
                    fz = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                    y = g(cond, TR.excitation(fz, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)), stt)[0].cpu().numpy()[N.DELAY:]
                y = y[n0:n0 + 8 * N.SR]
                if o is None:
                    ref = load48(od / f"OURS7{wsfx}" / f"{k}_OURS7{wsfx}.wav").astype(np.float32)
                    y0 = (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
                    dd = float(np.abs(y0[: len(ref)] - ref[: len(y0)]).max())
                    print("self-check z=0 vs D570 OURS7", k, dd, flush=True)
                    assert dd < 2e-3, dd
                    continue
                o.parent.mkdir(parents=True, exist_ok=True)
                sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            n_done += 1
        print(s, "rendered", n_done, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--stage", required=True, choices=("rr", "render", "mem"))
    ap.add_argument("--rr_tag", default="diag_rr19L")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--g", default="", help="d587 の G(R̄ の近似)の g.pt = S = R(目標) − G")
    ap.add_argument("--rrdir", default="d579_rr")
    ap.add_argument("--osfx", default="", help="出力の名前の末尾(R の差分の出どころの区別・例 G)")
    ap.add_argument("--mem_f1", action="store_true", help="目標の記憶の包絡を F1 の f0 の CheapTrick で(Rust の登録と同じ・既定は教師の f0)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=list(KS))
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--wsfx", default="", help="別の重みの版の名前の末尾(OURS<k>R<g><wsfx>・z = 0 の自己検査は OURS7<wsfx> と)")
    ap.add_argument("--exc_h1", type=float, default=0.0, help="D618 の励起の第 1 倍音の重み dB(d570_render --exc_h1 と同じ)")
    ap.add_argument("--h1fix", type=float, default=0.0, help="D611 の有声の枠の基音の傾き σ(d570_render --h1fix と同じ・自己検査の OURS7<wsfx> も同じ σ で描いたもの)")
    a = ap.parse_args()
    if a.stage in ("rr", "mem"):
        stage_rr(a.sets, a.device, a.rr_tag, a.g, a.rrdir, a.stage == "mem", a.mem_f1, a.limit)
    else:
        stage_render(a.sets, a.device, a.gain, tuple(a.ks), a.ckpt, a.wsfx, a.rrdir, a.osfx, a.h1fix, a.exc_h1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
