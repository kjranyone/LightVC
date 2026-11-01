"""D571(学習なし・オラクル・current/artic_morph.md 段 1): 固定の半音 k で、目標本人の実現のどの層(動き方・声の源・句の c0)が独立の本人らしさを運ぶか。規則は OVERNIGHT の D571。
組 = d165_real_parallel の select で作った男声 TTS → 実女声の並行の組(<s>/items.json・<s>/pNNNN_tTTT_SRC.wav・cen_TTT.wav)。前段・表・整列・出力部は D165 の convert と同じ(コードを写した)。
高さ = 操作者の F1 の f0 × 2^(k/12)(全版共通)。版 = TAB0・CONV・XDYN・XSRC・XC0・XDS・XCDS・GU(WORLD の一律のフォルマントシフト α)。
較正(物差しの天井): REALW = 目標本人の録音の WORLD 再合成(本人の高さ)・REALK = 同じ録音の高さだけを操作者 + k の中央値へ一律に移した WORLD 再合成(本人の声道・動き方のまま固定の半音の高さ)。
    uv run python d571_oracle.py --out <s> --k 7
    uv run python d571_oracle.py --out <s> --k 7 --stage eval --res ../results/conv_p0/d571.json
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

ROOT = Path(__file__).resolve().parent.parent
MALE = ROOT / "data/male_tts_v4l"
VERS = ("TAB0", "CONV", "XDYN", "XSRC", "XC0", "XDS", "XCDS", "GU", "REALW", "REALK")
EVAL_EXTRA = ("OURS7", "XDYNL", "XDYNH", "XRT", "XRTO", "XDYNC", "XDYNCH", "XDYNHLQ", "XDYNHHQ", "XDYNCHS", "XRE", "XREC", "OURS7R", "OURS7RC", "OURS7NC", "OURS7NV", "OURS7RNC", "OURS7Y58", "OURS7RY58", "OURS7Y59", "OURS7RY59")


def gu(x: np.ndarray, k: float, a: float, ratio: float | None = None) -> np.ndarray:
    import pyworld as pw
    import scipy.signal as ss
    fp = 5.0
    x = x.astype(np.float64)
    f0, t = pw.harvest(ss.resample_poly(x, 1, 3), 16000, f0_floor=50.0, f0_ceil=800.0, frame_period=fp)
    sp = pw.cheaptrick(x, f0, t, 48000)
    ap = pw.d4c(x, f0, t, 48000)
    fq = np.linspace(0, 24000, sp.shape[1])
    sp2 = np.stack([np.interp(fq / a, fq, r) for r in sp]) if a != 1.0 else sp
    ap2 = np.stack([np.interp(fq / a, fq, r) for r in ap]) if a != 1.0 else ap
    r = ratio if ratio is not None else 2 ** (k / 12)
    y = pw.synthesize(f0 * r, np.ascontiguousarray(sp2), np.ascontiguousarray(ap2), 48000, fp)
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def ctx_feats(p: np.ndarray) -> np.ndarray:
    c = np.concatenate([np.zeros((p.shape[0], 20)), p], 1).cumsum(1)
    n = p.shape[1]
    win = lambda lo, hi: (c[:, 20 - lo:20 - lo + n] - c[:, 20 - hi - 1:20 - hi - 1 + n]) / (hi - lo + 1)
    return np.concatenate([p, win(1, 3), win(4, 9), win(10, 19)], 0)


def top16(p: np.ndarray) -> np.ndarray:
    idx = np.argsort(-p, axis=0)[:16]
    out = np.zeros_like(p)
    np.put_along_axis(out, idx, np.take_along_axis(p, idx, 0), 0)
    return out


def ridge_fit(F: np.ndarray, R: np.ndarray) -> np.ndarray:
    n = F.shape[1]
    h = n // 2
    sc = np.trace(F @ F.T) / F.shape[0]
    best, bl = 1e18, None
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        err = 0.0
        for tr, te in ((slice(0, h), slice(h, n)), (slice(h, n), slice(0, h))):
            A = F[:, tr]
            B = np.linalg.solve(A @ A.T + lam * sc * np.eye(F.shape[0]) * (A.shape[1] / n), A @ R[:, tr].T).T
            err += float(((R[:, te] - B @ F[:, te]) ** 2).mean())
        if err < best:
            best, bl = err, lam
    return np.linalg.solve(F @ F.T + bl * sc * np.eye(F.shape[0]), F @ R.T).T


def evaluate(od: Path, k: int, res: Path) -> None:
    from transformers import WavLMForXVector
    import d394_idmargin as D
    import idloss as ID
    dev = "cuda"
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    ec = ID.Ecapa(dev)

    @torch.no_grad()
    def emb(p: Path) -> dict:
        x = torch.from_numpy(D.load16(p))[None].to(dev)
        w = torch.nn.functional.normalize(sv(input_values=x).embeddings, dim=-1)[0].cpu().numpy()
        x48, _ = sf.read(str(p), dtype="float32", always_2d=True)
        e = ec(torch.from_numpy(x48.mean(1))[None].to(dev))[0].cpu().numpy()
        return {"wsv": w, "ecapa": e / np.linalg.norm(e)}

    items = json.loads((od / "items.json").read_text())
    C = {ti: emb(od / f"cen_{ti:03d}.wav") for ti in sorted({it["ti"] for it in items})}
    per: dict = {}
    for it in items:
        pre = f"p{it['item']:04d}_t{it['ti']:03d}"
        for v in VERS + tuple(x.replace("OURS7", f"OURS{k}") for x in EVAL_EXTRA) + ("SRC", "REAL"):
            f = od / f"k{k}" / f"{pre}_{v}.wav" if v not in ("SRC", "REAL") else (od / f"{pre}_SRC.wav" if v == "SRC" else Path(it["clip"]))
            if not f.exists():
                continue
            e = emb(f)
            per.setdefault(v, {})[pre] = {m: float(e[m] @ C[it["ti"]][m] - np.mean([e[m] @ C[t][m] for t in C if t != it["ti"]])) for m in ("wsv", "ecapa")}
    rng = np.random.default_rng(0)
    rep = {"k": k, "margin": {v: {m: round(float(np.mean([r[m] for r in d.values()])), 4) for m in ("wsv", "ecapa")} | {"n": len(d)} for v, d in per.items()}, "paired": {}}
    PR = [(v, "TAB0") for v in VERS + EVAL_EXTRA if v != "TAB0"] + [("OURS7RY58", "XDS"), ("OURS7RY58", "XCDS"), ("OURS7RY58", "REALK"), ("OURS7R", "XDS"), ("XDS", "REALK"), ("OURS7Y59", "OURS7"), ("OURS7RY59", "OURS7R"), ("OURS7RY59", "OURS7RY58"), ("OURS7RY59", "GU"), ("OURS7RY59", "XRE"), ("OURS7Y58", "OURS7"), ("OURS7RY58", "OURS7R"), ("OURS7RY58", "OURS7Y58"), ("OURS7RY58", "GU"), ("OURS7RY58", "XRE"), ("OURS7RY58", "TAB0"), ("OURS7NC", "OURS7"), ("OURS7NV", "OURS7"), ("OURS7RNC", "OURS7R"), ("OURS7RNC", "XRE"), ("OURS7R", "OURS7"), ("OURS7RC", "OURS7R"), ("OURS7R", "GU"), ("OURS7R", "XRE"), ("XRE", "XDYN"), ("XREC", "XRE"), ("XDYNCH", "XDYNCHS"), ("XDYNC", "XDYN"), ("XDYNCH", "XDYNH"), ("XDYNHLQ", "XDYNHHQ"), ("XRT", "XRTO"), ("XDYNL", "XDYNH"), ("XRT", "XDYN"), ("OURS7", "GU"), ("OURS7", "CONV"), ("OURS7", "REALK"), ("XCDS", "GU"), ("CONV", "GU"), ("TAB0", "GU"), ("REALK", "GU"), ("XCDS", "REALK"), ("REALK", "REALW")]
    for a_, b_ in [(x.replace("OURS7", f"OURS{k}"), y.replace("OURS7", f"OURS{k}")) for x, y in PR]:
        keys = sorted(set(per.get(a_, {})) & set(per.get(b_, {})))
        if not keys:
            continue
        tg = np.array([q[-4:] for q in keys])
        us = sorted(set(tg))
        idx = {u: np.flatnonzero(tg == u) for u in us}
        out = {"n": len(keys)}
        for m in ("wsv", "ecapa"):
            d = np.array([per[a_][q][m] - per[b_][q][m] for q in keys])
            bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
            out[m] = {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
        rep["paired"][f"{a_}-{b_}"] = out
    res.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["margin"]), flush=True)
    for q, v in rep["paired"].items():
        print(q, v, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=1.12)
    ap.add_argument("--stage", default="render")
    ap.add_argument("--res", default="")
    ap.add_argument("--rvoc_ckpt", default=str(ROOT / "results/rvtab_f3/snap/ema_380k.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default="")
    ap.add_argument("--rr_tag", default="diag_rr19L")
    a = ap.parse_args()
    only = [v for v in a.only.split(",") if v]
    od = Path(a.out)
    if a.stage == "eval":
        evaluate(od, a.k, Path(a.res))
        return 0
    import librosa
    import scipy.signal as ss
    import c1_content as CC
    import conv_c0 as C0
    import f0est as FE
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    from d6_unit_cov import load48
    from d88_perturb_identity import msqrt, wmat
    dev = a.device
    torch.set_num_threads(4)
    sr_ = torch.load(a.rvoc_ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    ld = lambda p: torch.load(ROOT / p, map_location="cpu", weights_only=False)
    s1 = ld("results/f0est3/last.pt"); f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(s1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    s2 = ld("results/f2_2/last.pt"); f2 = LV.Lvl(s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev); f2.load_state_dict(s2["net"]); f2.eval()
    s3 = ld("results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    s4 = ld("results/f4_1/last.pt"); f4 = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev); f4.load_state_dict(s4["net"]); f4.eval()
    sc = ld("results/c1_1/last.pt"); c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev); c1.load_state_dict(sc["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt", map_location="cpu").cpu().numpy()
    K = Cb.shape[0]
    W = wmat(1.3)

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
        p4 = f4(m0)[0].cpu().numpy()[:, :n].astype(np.float64)
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        ft, _ = H.teacher_f0(x, n)
        return {"x": x, "n": n, "f0": pad1(f0f), "c0": pad1(c0f), "e3": pad(e3), "p4": pad(p4), "p": pad(p), "u": pad(p).argmax(0), "ft": pad1(ft[:n].astype(np.float64))}

    def cheap(fr: dict) -> np.ndarray:
        xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, fr["ft"]), ((0, 0), (1, 1)), mode="edge")
        return (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :fr["n"]][1:25].astype(np.float64)

    def enroll(frs: list, use_f3: bool) -> dict:
        E = np.concatenate([fr["e3"] if use_f3 else cheap(fr) for fr in frs], 1)
        U = np.concatenate([fr["u"] for fr in frs])
        T = C0.table(E, U, K, Cb)
        F0 = np.concatenate([fr["f0"] if use_f3 else fr["ft"] for fr in frs])
        v = F0 > 0
        r = E - T[:, U]
        rv = r[:, v] - r[:, v].mean(1, keepdims=True)
        P4 = np.concatenate([fr["p4"] for fr in frs], 1)
        sM = {"e3": E[:, v].mean(1), "c0": float(np.concatenate([fr["c0"] for fr in frs])[v].mean()), "lf": float(np.median(np.log(F0[v])))} if use_f3 else None
        return {"T": T, "mu": float(np.median(np.log(F0[v]))), "cov": rv @ rv.T / rv.shape[1], "p4mean": P4[:, v].mean(1), "sM": sM}

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def dtwj(X_: np.ndarray, Y_: np.ndarray) -> np.ndarray:
        _, wq = librosa.sequence.dtw(X=X_, Y=Y_, metric="euclidean")
        jq = np.zeros(X_.shape[1], int)
        for i_, j_ in wq[::-1]:
            jq[i_] = j_
        return jq

    bl, al = ss.butter(2, 3.0 / 100.0)
    lo = lambda z: ss.filtfilt(bl, al, z)

    items = json.loads((od / "items.json").read_text())
    use_rr = bool({"XRE", "XREC", "RRDUMP"} & set(only))
    (od / "rr_delta").mkdir(exist_ok=True)
    if use_rr:
        import train_rr as RRm
        from d208_rr_natural import anchors
        ckr = torch.load(ROOT / "results" / a.rr_tag / "last.pt", map_location="cpu", weights_only=False)
        rnet = RRm.build(ckr["cfg"]).to(dev).eval()
        rnet.load_state_dict(ckr["net"])

        def cheap25(fr: dict) -> np.ndarray:
            xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
            e = np.pad(PA.envelope(xa, fr["ft"]), ((0, 0), (1, 1)), mode="edge")
            return (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :fr["n"]][:25].astype(np.float64)

        def mem_in(p_: np.ndarray, env: np.ndarray) -> tuple:
            n_ = min(env.shape[1], 5000)
            rc0 = (env[0, :n_] - RRm.C0_SIL) / 10
            return p_[:, :n_].astype(np.float32), np.concatenate([env[1:25, :n_] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)

        @torch.no_grad()
        def rmem(p_: np.ndarray, env: np.ndarray) -> tuple:
            rp, rv = mem_in(p_, env)
            return rnet.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev)), rp.shape[1]

        FEAT = ROOT / "data/rr_feat"
        man = {it_["stem"]: it_ for it_ in json.loads((FEAT / "real_fem/manifest.json").read_text())["items"]}
        amem = []
        anc = anchors(8)
        assert not ({A_["target"] for A_ in anc} & {q["target"] for q in items})
        for A_ in anc:
            ds = [torch.load(FEAT / man[s_]["pt"], weights_only=False) for s_ in A_["ref_stems"] if s_ in man]
            amem.append(rmem(np.concatenate([RRm.dense(d["pk"], d["pv"]) for d in ds], 1), np.concatenate([d["env"].float().numpy() for d in ds], 1)))
        print("rr anchors", len(amem), flush=True)

        @torch.no_grad()
        def rr_delta(fm_: dict, sM: dict, tab_: np.ndarray, mA: tuple) -> tuple:
            vM = fm_["f0"] > 0
            pq = top16(fm_["p"])
            q = np.concatenate([pq, (fm_["e3"] - sM["e3"][:, None]) / 10, (fm_["c0"] - sM["c0"])[None],
                                np.where(vM, np.log(np.maximum(fm_["f0"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float64)], 0).astype(np.float32)
            qt, tt = torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab_.astype(np.float32))[None].to(dev)

            def run(mm: tuple) -> tuple:
                e_, d0_, _ = rnet(qt, tt, mm[0], torch.ones(1, mm[1], dtype=torch.bool, device=dev))
                return e_[0].cpu().numpy().astype(np.float64), d0_[0].cpu().numpy().astype(np.float64)
            eA, dA = run(mA)
            outs = [run(mm) for mm in amem]
            return eA - np.mean([o_[0] for o_ in outs], 0), dA - np.mean([o_[1] for o_ in outs], 0)
    ko = od / f"k{a.k}"
    ko.mkdir(parents=True, exist_ok=True)
    ops: dict = {}
    tgs: dict = {}
    for it in items:
        pre = f"p{it['item']:04d}_t{it['ti']:03d}"
        if not (od / f"{pre}_SRC.wav").exists():
            continue
        if (all((od / "rr_delta" / f"{pre}.npz").exists() if v == "RRDUMP" else (ko / f"{pre}_{v}.wav").exists() for v in only) if only else (ko / f"{pre}_GU.wav").exists()):
            continue
        if it["spk"] not in ops:
            mf, tot = [], 0.0
            for f in sorted(MALE.glob(f"{it['spk']}/t*.wav")):
                if tot >= 22.0:
                    break
                x = load48(f).astype(np.float32); mf.append(front(x)); tot += len(x) / 48000
            ops[it["spk"]] = enroll(mf, True)
        def tget(ti: int) -> tuple:
            if ti not in tgs:
                ref = next(q["ref"] for q in items if q["ti"] == ti)
                rf = [front(load48(r).astype(np.float32)) for r in ref]
                T0, T3 = enroll(rf, False), enroll(rf, True)
                Pr = np.concatenate([fr["p"] for fr in rf], 1)
                Rr = np.concatenate([fr["e3"] for fr in rf], 1) - T3["T"] @ Pr
                T3["B"] = ridge_fit(ctx_feats(Pr), Rr) if {"XRT", "XRTO"} & set(only) else None
                if use_rr:
                    T3["mem"] = rmem(top16(Pr), np.concatenate([cheap25(fr) for fr in rf], 1))
                tgs[ti] = (T0, T3)
            return tgs[ti]
        O_, (T_, TA3) = ops[it["spk"]], tget(it["ti"])
        xs = load48(od / f"{pre}_SRC.wav").astype(np.float32)
        fm = front(xs)
        fa = front(load48(it["clip"])[: 8 * 48000].astype(np.float32))
        j = dtwj(np.sqrt(np.maximum(fm["p"], 0)), np.sqrt(np.maximum(fa["p"], 0)))
        f0o = np.where(fm["f0"] > 0, fm["f0"] * 2 ** (a.k / 12), 0.0)
        per_c = fm["p4"] - O_["p4mean"][:, None] + T_["p4mean"][:, None]
        per_t = fa["p4"][:, j]
        M_ = msqrt(T_["cov"], 0.5) @ msqrt(W @ O_["cov"] @ W.T, -0.5)
        tab = T_["T"] @ fm["p"]
        dyn_c = 0.7 * M_ @ W @ (fm["e3"] - O_["T"] @ fm["p"])
        fa_dyn = 0.7 * (fa["e3"][:, j] - TA3["T"] @ fa["p"][:, j])
        cm, ca = fm["c0"], fa["c0"][j]
        if only:
            tis = sorted({q["ti"] for q in items})
            oth = tis[(tis.index(it["ti"]) + 1) % len(tis)]
            lp = np.stack([lo(r) for r in fa_dyn])
            fc_dyn = 0.7 * (cheap(fa)[:, j] - T_["T"] @ fa["p"][:, j])
            hc = fc_dyn - np.stack([lo(r) for r in fc_dyn])
            vi = np.flatnonzero(f0o > 0)
            hs = hc.copy()
            hs[:, vi] = np.roll(hc[:, vi], len(vi) // 2, axis=1)
            hq = np.zeros_like(fa_dyn)
            hq[8:] = (fa_dyn - lp)[8:]
            ext = {"XDYNL": (cm, tab + lp, f0o, per_c), "XDYNH": (cm, tab + fa_dyn - lp, f0o, per_c),
                   "XDYNC": (cm, tab + fc_dyn, f0o, per_c), "XDYNCH": (cm, tab + hc, f0o, per_c), "XDYNCHS": (cm, tab + hs, f0o, per_c),
                   "XDYNHLQ": (cm, tab + (fa_dyn - lp) - hq, f0o, per_c), "XDYNHHQ": (cm, tab + hq, f0o, per_c)}
            if use_rr:
                dS, dC = rr_delta(fm, O_["sM"], tab, TA3["mem"])
                ext |= {"XRE": (cm, tab + dS, f0o, per_c), "XREC": (cm + dC, tab + dS, f0o, per_c)}
            if {"XRT", "XRTO"} & set(only):
                Fm = ctx_feats(fm["p"])
                ext |= {"XRT": (cm, tab + 0.7 * TA3["B"] @ Fm, f0o, per_c), "XRTO": (cm, tab + 0.7 * tget(oth)[1]["B"] @ Fm, f0o, per_c)}
            for c in only:
                if c == "RRDUMP":
                    np.savez(od / "rr_delta" / f"{pre}.npz", dS=dS.astype(np.float32), dC=dC.astype(np.float32))
                    continue
                cc, ee, ff, pp = ext[c]
                sf.write(ko / f"{pre}_{c}.wav", render(cc, ee, ff, pp), N.SR)
            print(pre, flush=True)
            continue
        conds = {"TAB0": (cm, tab, f0o, per_c), "CONV": (cm, tab + dyn_c, f0o, per_c), "XDYN": (cm, tab + fa_dyn, f0o, per_c),
                 "XSRC": (cm, tab, f0o, per_t), "XC0": (ca, tab, f0o, per_c), "XDS": (cm, tab + fa_dyn, f0o, per_t), "XCDS": (ca, tab + fa_dyn, f0o, per_t)}
        for c, (cc, ee, ff, pp) in conds.items():
            sf.write(ko / f"{pre}_{c}.wav", render(cc, ee, ff, pp), N.SR)
        xt = load48(it["clip"]).astype(np.float32)[: 8 * 48000]
        rt = float(np.median(f0o[f0o > 0]) / np.median(fa["ft"][fa["ft"] > 0]))
        sf.write(ko / f"{pre}_REALW.wav", gu(xt, 0, 1.0, 1.0), N.SR)
        sf.write(ko / f"{pre}_REALK.wav", gu(xt, 0, 1.0, rt), N.SR)
        sf.write(ko / f"{pre}_GU.wav", gu(xs, a.k, a.alpha), N.SR)
        print(pre, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
