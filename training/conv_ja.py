"""日本語の目標話者での変換の同一性(C0-7 の日本語の部分・current/converter.md)。目標 = 評価話者(held)の日本語の実音声 23 人(参照 12〜25s・順位の基準の重心は参照と別のファイル)・
ソース = VCTK の評価男声(conv_c0 と同じ SENTS 3〜6)・各目標に 5 人。条件: SRC(元の男声)・TAB(G なし = P0C の因果版)・G ごと(--g 名前=ckpt)。
順位 = 23 人の目標の中の本人(偶然は 1/23)。ECAPA と WavLM-SV。目標ごとにまとめた検定(cluster)つき。
    uv run python conv_ja.py --ladder <scratchpad>/r4_spk --g c3_1=../results/c3_1/last.pt c3_2=../results/c3_2/last.pt --out ../results/conv_p0/ja.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--g", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--males_per", type=int, default=5)
    ap.add_argument("--work", default="/tmp/conv_ja_wav")
    ap.add_argument("--exclude_spk", nargs="*", default=[])
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--save_dir", default="")
    ap.add_argument("--ctx", default="", choices=["", "prev", "pos"], help="TABC: 文脈つきの表(prev = 直前の別の単位・pos = 単位の連続の中の位置 0 / 1-2 / 3+)。単位だけの表へ観測数で縮約")
    ap.add_argument("--ctx_lam", type=float, default=3.0)
    ap.add_argument("--src", default="vctk", choices=["vctk", "ja", "jaf", "self"], help="元の声(jaf = 評価外の日本語女声・self = 目標自身の別発話 = 自己変換): vctk = VCTK 英語の評価男声 / ja = 日本語の TTS 男声(VCTK 男声のクローン・製品の条件に近い)")
    ap.add_argument("--w_sm", type=int, default=0, help="表の引きの因果平均の幅(0 = 既定の W_SM)")
    ap.add_argument("--cen_skip", type=float, default=0.0, help="中心の埋め込みに使う別発話の先頭 s 秒を除く(self の元の声に使う・梯子の比較では全条件で同じ値にする)")
    ap.add_argument("--ta", action="store_true", help="TABD: 目標接近モデル(構音の動力学): ê = 話者平均 + κ(T[u] − 話者平均) を 1 次 IIR(係数 α)で追う。(κ, α)は話者ごとに参照から推定(要約 2 数・因果)")
    ap.add_argument("--f2", default="results/f2_2/last.pt")
    ap.add_argument("--f0_rho", type=float, default=1.0, help="f0 の動きの尺度: log f0_out = μ_目標 + ρ(log f0_src − 中央値)(診断・1 = 現行)")
    ap.add_argument("--oracle_units", action="store_true", help="TABO: 表の引きに ContentVec の単位(非因果・チェーン外)を使う条件を足す(単位の質の上限)")
    ap.add_argument("--ref_sec", type=float, default=12.0)
    ap.add_argument("--shrink", type=float, default=0.0, help="表を全話者の単位平均へ観測数で縮約(λ フレーム・0 = 現行)")
    a = ap.parse_args()
    import conv_c0 as C0
    import c1_content as CC
    import f0est as FE
    import f0hi as H
    import idloss as ID
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_f0est as TF
    import train_rvoc as TR
    from scipy.signal import resample_poly
    from scipy.stats import wilcoxon
    from math import gcd
    from train_dec2 import load48
    dev = "cuda"
    C0.LAD["path"] = a.ladder
    g = C0.G()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval().requires_grad_(False)
    f1front = FE.Front().to(dev)
    sf2 = torch.load(ROOT / a.f2, map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval().requires_grad_(False)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval().requires_grad_(False)
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").to(dev)
    K = Cb.shape[0]
    ecapa, wsv, cvec = ID.Ecapa(dev), ID.WavlmSV(dev), ID.ContentVec(dev)

    KAP = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    ALP = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9]

    def ta_run(Tuni, mean, u, kap, alp):
        tgt = mean[:, None] + kap * (Tuni[:, u] - mean[:, None])
        if alp == 0:
            return tgt
        from scipy.signal import lfilter
        return lfilter([1 - alp], [1, -alp], tgt, axis=1, zi=tgt[:, :1] * alp)[0]

    def ta_fit(Tuni, env24, u):
        mean = env24.mean(1)
        best = None
        for kap in KAP:
            for alp in ALP:
                e = np.abs(ta_run(Tuni, mean, u, kap, alp) - env24).mean()
                if best is None or e < best[0]:
                    best = (e, kap, alp)
        base = np.abs(C0.smooth(Tuni[:, u]) - env24).mean() if hasattr(C0, "smooth") else float("nan")
        return {"mean": mean, "kap": best[1], "alp": best[2], "fit_l1": float(best[0]), "base_l1": float(base)}

    def ctx_of(u, mode):
        n_ = len(u)
        rs = np.zeros(n_, int)
        for t_ in range(1, n_):
            rs[t_] = rs[t_ - 1] if u[t_] == u[t_ - 1] else t_
        if mode == "pos":
            ps = np.minimum(np.maximum(np.arange(n_) - rs, 0), 3)
            return np.where(ps == 0, 0, np.where(ps <= 2, 1, 2))
        return np.where(rs > 0, u[np.maximum(rs - 1, 0)], u)

    def ctx_stats(env24, u, mode):
        c_ = ctx_of(u, mode)
        nc = 3 if mode == "pos" else K
        sums = np.zeros((nc, K, 24)); cnt = np.zeros((nc, K))
        np.add.at(sums, (c_, u), env24.T)
        np.add.at(cnt, (c_, u), 1.0)
        return sums, cnt

    def ctx_lookup(st, Tuni, hard, mode, lam):
        sums, cnt = st
        c_ = ctx_of(hard, mode)
        sm_, ct_ = sums[c_, hard], cnt[c_, hard]
        prior = Tuni[:, hard].T
        return ((sm_ + lam * prior) / (ct_[:, None] + lam)).T

    def sm3(E, w=sm_w):
        e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
        return w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:]

    def load_cat(files):
        xs = []
        for w in files:
            x, sr = sf.read(str(w), dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 48000:
                gg = gcd(sr, 48000)
                x = resample_poly(x, 48000 // gg, sr // gg).astype(np.float32)
            xs.append(x)
        return np.concatenate(xs)

    mu_u = np.load(ROOT / "data/c3/targets.npz")["T"].astype(np.float64).mean(0) if a.shrink > 0 else None
    # 目標(評価話者の日本語の実音声)
    held = [s for s in sorted(TR.held_speakers()) if s != "unknown" and s not in a.exclude_spk and (ROOT / "female-dataset" / s).is_dir()]
    tg = {"spk": [], "T": [], "mu": [], "per": [], "cen_ec": [], "cen_wv": []}
    with torch.no_grad():
        for s in held:
            ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
            ref, tot = [], 0.0
            for w in ws:
                if tot >= a.ref_sec:
                    break
                ref.append(w)
                tot += sf.info(str(w)).duration
            rest = [w for w in ws if w not in ref]
            if not rest:
                continue
            x = load_cat(ref)[:int(max(a.ref_sec, 25) * 48000)]
            n = len(x) // N.HOP
            x = x[:n * N.HOP]
            f0r, _ = H.teacher_f0(x, n)
            xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
            env = sm3(PA.envelope(xa, f0r))
            per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0r)[None])[0].numpy()
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].T.cpu().numpy()
            T = C0.table(env[1:25], p.argmax(1), K, Cb.cpu().numpy())
            if a.shrink > 0:
                cnt = np.bincount(p.argmax(1), minlength=K).astype(np.float64)
                T = ((T * cnt[None] + a.shrink * mu_u) / (cnt[None] + a.shrink)).astype(np.float32)
            vo = f0r > 0
            cen_full = load_cat(rest)[:int((25 + a.cen_skip) * 48000)]
            if len(cen_full) < int((a.cen_skip + 8) * 48000):
                continue
            tg.setdefault("self_x", []).append(cen_full[:int(a.cen_skip * 48000)])
            cen = cen_full[int(a.cen_skip * 48000):]
            ct = torch.from_numpy(cen)[None].to(dev)
            if a.save_dir:
                sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
                sf.write(str(sd / f"cen_{len(tg['spk']):02d}.wav"), cen, 48000)
            if a.ta:
                tg.setdefault("ta", []).append(ta_fit(T.astype(np.float64), env[1:25, :len(p)].astype(np.float64), p.argmax(1)))
            if a.ctx:
                tg.setdefault("ctx", []).append(ctx_stats(env[1:25, :len(p)], p.argmax(1), a.ctx))
            tg["spk"].append(s); tg["T"].append(T.astype(np.float32)); tg["mu"].append(float(np.median(np.log(f0r[vo]))))
            tg["per"].append(per[:, vo].mean(1)); tg["cen_ec"].append(ecapa(ct)[0].cpu().numpy()); tg["cen_wv"].append(wsv(ct)[0].cpu().numpy())
    S = len(tg["spk"])
    print("targets", S, flush=True)
    Tt = torch.from_numpy(np.stack(tg["T"])).to(dev) / 10.0
    mu = torch.tensor(tg["mu"], dtype=torch.float32, device=dev)
    per_t = torch.from_numpy(np.stack(tg["per"]).astype(np.float32)).to(dev)
    summ = torch.cat([Tt.mean(2), mu[:, None], per_t], 1)
    cen_ec = torch.from_numpy(np.stack(tg["cen_ec"])).to(dev)
    cen_wv = torch.from_numpy(np.stack(tg["cen_wv"])).to(dev)
    chains = {}
    for spec in [("TAB", None)] + [tuple(x.split("=", 1)) for x in a.g]:
        name, path = spec
        sg = torch.load(path or (a.g[0].split("=", 1)[1]), map_location="cpu", weights_only=False)
        G = C3.ConvG(sg["cfg"]["K"], sg["cfg"]["ch"], tuple(sg["cfg"]["dils"]), sg["cfg"]["d_film"], sg["cfg"].get("m_basis", 0)).to(dev)
        G.load_state_dict(sg["G"], strict=False); G.eval()
        chains[name] = (C3.Chain(c1, mfront, Cb, f1, f1front, f2, G, Tt, mu, per_t, summ, dev), path is None)
    for _ch, _z in chains.values():
        _ch.rho = a.f0_rho
        if a.w_sm:
            _ch.w_sm = a.w_sm
    if a.ta:
        chains["TABD"] = (chains["TAB"][0], True)
        print("TA の推定(話者ごと κ, α, 当てはめの L1 / 固定平滑の L1):", [(d["kap"], d["alp"], round(d["fit_l1"], 3), round(d["base_l1"], 3)) for d in tg["ta"]], flush=True)
    if a.ctx:
        chains["TABC"] = (chains["TAB"][0], True)
    if a.oracle_units:
        chains["TABO"] = (chains["TAB"][0], True)
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    males = J["males"]
    if a.src == "self":
        males = ["self"]
        a.males_per = 1
    if a.src == "jaf":
        hs_ = TR.held_speakers()
        fl_ = sorted(p_.name for p_ in (ROOT / "female-dataset").iterdir() if p_.is_dir() and p_.name not in hs_ and p_.name not in {"fe9565ca1f33bf20", "ffb9b5647612b32b"})
        import random as _r
        males = _r.Random(0).sample(fl_, 20)
        jsrc = {}
        for m_ in males:
            fs_, tot_ = [], 0.0
            for w_ in sorted((ROOT / "female-dataset" / m_).glob("*.wav")):
                if tot_ >= 9.0:
                    break
                fs_.append(str(w_)); tot_ += sf.info(str(w_)).duration
            jsrc[m_] = fs_
    if a.src == "ja":
        mrows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["src"] == "tts_m"]
        jby: dict = {}
        for r in mrows:
            jby.setdefault(r["spk"], []).append(r)
        males = sorted(jby)
        jsrc = {}
        for m_ in males:
            fs_, tot_ = [], 0.0
            for r in sorted(jby[m_], key=lambda r: r["wav"]):
                if tot_ >= 9.0:
                    break
                fs_.append(r["wav"]); tot_ += r["dur"]
            jsrc[m_] = fs_
    pairs = [(males[(i * a.males_per + j) % len(males)], i) for i in range(S) for j in range(a.males_per)]
    names = ["SRC"] + list(chains) + ([f"{nm}_SHUF" for nm in chains if nm != "TAB"] if a.shuffle else [])
    ranks = {nm: {"ec": [], "wv": []} for nm in names}
    agree = {nm: [] for nm in chains}
    am_ = {nm: [] for nm in chains}
    import eval_nvoc as E
    with torch.no_grad():
        for pi, (m, ti) in enumerate(pairs):
            if a.src == "self":
                xs = g.norm(g.trim(tg["self_x"][ti].astype(np.float64))).astype(np.float32)
            elif a.src in ("ja", "jaf"):
                xs = np.concatenate([g.norm(g.trim(load48(w).astype(np.float64))) for w in jsrc[m]]).astype(np.float32)
            else:
                us = [u for u in C0.SENTS if C0.utt(m, u).exists()]
                xs = np.concatenate([g.norm(g.trim(load48(str(C0.utt(m, u))).astype(np.float64))) for u in us]).astype(np.float32)
            n = len(xs) // N.HOP
            xs = xs[:n * N.HOP]
            xg = torch.from_numpy(xs)[None].to(dev)
            f0s = TF.infer(f1front, f1, xs, dev)
            v10 = f0s[:2000][f0s[:2000] > 0]
            med = torch.tensor([float(np.log(np.median(v10)))], device=dev)
            ux = (cvec(xg) @ Cb.T).argmax(-1)[0]
            outs = {"SRC": xg[:, :8 * 48000]}
            for nm, (ch, zero) in chains.items():
                ho = None
                if nm == "TABO":
                    jj = ((torch.arange(n, device=dev).float() - 5.5) / 4).round().long().clamp(0, len(ux) - 1)
                    ho = ux[jj][None]
                if nm == "TABD":
                    q = ch.prep(xg, torch.tensor([ti], device=dev), med)
                    d_ = tg["ta"][ti]
                    e2 = ta_run(tg["T"][ti].astype(np.float64), d_["mean"], q["hard"][0].cpu().numpy(), d_["kap"], d_["alp"]) / 10
                    e_s2 = torch.from_numpy(e2.astype(np.float32))[None].to(dev)
                    cond = C3.build_cond(q["c0n"], e_s2, torch.zeros_like(e_s2), q["lf"], q["vo"], q["pm"], torch.zeros(1, 4, e_s2.shape[-1], device=dev))
                    f0o, p = q["f0o"], q["p_soft"]
                elif nm == "TABC":
                    q = ch.prep(xg, torch.tensor([ti], device=dev), med)
                    e2 = ctx_lookup(tg["ctx"][ti], tg["T"][ti].astype(np.float64), q["hard"][0].cpu().numpy(), a.ctx, a.ctx_lam) / 10
                    e_s2 = C3.causal_avg(torch.from_numpy(e2.astype(np.float32))[None].to(dev), C3.W_SM)
                    cond = C3.build_cond(q["c0n"], e_s2, torch.zeros_like(e_s2), q["lf"], q["vo"], q["pm"], torch.zeros(1, 4, e_s2.shape[-1], device=dev))
                    f0o, p = q["f0o"], q["p_soft"]
                else:
                    cond, f0o, dc, dp, p = ch(xg, torch.tensor([ti], device=dev), med, zero, ho)
                y = gen(cond, TR.excitation(f0o, n * N.HOP, torch.Generator().manual_seed(0)))[:, N.DELAY:]
                y = y / (y.abs().max() + 1e-9) * 0.9
                outs[nm] = y[:, :8 * 48000]
                uy = (cvec(y) @ Cb.T).argmax(-1)[0]
                k = min(len(ux), len(uy))
                agree[nm].append(float((ux[:k] == uy[:k]).float().mean()))
                am_[nm].append(E.amline(y[0].cpu().numpy()))
            if a.shuffle:
                ts = (ti + 1 + (pi % (S - 1))) % S
                for nm, (ch, zero) in chains.items():
                    if nm == "TAB":
                        continue
                    cond, f0o, dc, dp, p = ch(xg, torch.tensor([ts], device=dev), med, zero)
                    y = gen(cond, TR.excitation(f0o, n * N.HOP, torch.Generator().manual_seed(0)))[:, N.DELAY:]
                    y = y / (y.abs().max() + 1e-9) * 0.9
                    outs[f"{nm}_SHUF"] = y[:, :8 * 48000]
            if a.save_dir:
                sd = Path(a.save_dir); sd.mkdir(parents=True, exist_ok=True)
                for nm, y in outs.items():
                    sf.write(str(sd / f"p{pi:03d}_t{ti:02d}_{nm}.wav"), y[0].cpu().numpy(), 48000)
            for nm, y in outs.items():
                se, sw = (ecapa(y) @ cen_ec.T)[0], (wsv(y) @ cen_wv.T)[0]
                ranks[nm]["ec"].append(int((se > se[ti]).sum()) + 1)
                ranks[nm]["wv"].append(int((sw > sw[ti]).sum()) + 1)
    tgt = np.array([ti for _, ti in pairs])
    rep = {"n_targets": S, "n_pairs": len(pairs), "chance_top1": round(1 / S, 3)}
    for ver in ("ec", "wv"):
        for nm in names:
            r = np.array(ranks[nm][ver])
            d = {"top1": round(float((r == 1).mean()), 3), "top5": round(float((r <= 5).mean()), 3), "mean_rank": round(float(r.mean()), 2)}
            if nm not in ("SRC", "TAB"):
                b = np.array(ranks["TAB"][ver])
                ma = np.array([r[tgt == t].mean() for t in range(S)]); mb = np.array([b[tgt == t].mean() for t in range(S)])
                try:
                    d["cluster_wilcoxon_vs_TAB(改善の片側)"] = float(wilcoxon(ma, mb, alternative="less").pvalue)
                    d["cluster_wilcoxon_vs_TAB(悪化の片側)"] = float(wilcoxon(ma, mb, alternative="greater").pvalue)
                except ValueError:
                    pass
                d["targets_better/worse"] = [int((ma < mb).sum()), int((ma > mb).sum())]
            rep[f"{ver}_{nm}"] = d
    rep["pair_ranks"] = {f"{v}_{nm}": ranks[nm][v] for nm in names for v in ("ec", "wv")}
    rep["pair_target"] = tgt.tolist()
    rep["cv_unit_agree"] = {nm: round(float(np.mean(v)), 4) for nm, v in agree.items()}
    rep["am_line_db"] = {nm: round(float(np.mean(v)), 3) for nm, v in am_.items()}
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
