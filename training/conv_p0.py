"""P0: 変換の製品経路(出力部を通す)の同一性の基準(current/converter.md 4d・学習なし)。conv_c0 / conv_c1 と同じ 225 組・同じ材料・同じ採点。
  P0_TAB   元男声(S0)→ C1 の単位の事後(硬い引き)→ 目標の表 T[u](参照 ≥ 10s の CheapTrick 包絡 c1..c24 [.25,.5,.25] 平滑の単位平均)→ ê_t(w=5 の一様平滑)
           + c0 = 元の c0(F1 の f0 で分析)+ log f0_out = log f0_src(F1)− 発話の中央値 + 目標の中央値 + 有声 = F1 + 周期性 = 目標の有声フレームの 4 帯の平均
           → 出力部 rvoc3hi 330k → 波形。診断のため c0 と f0 の中央値は発話単位(製品は因果の走行推定)・周期性は目標の平均で G なし(Δ = 0)。
  P0_COPY  目標の女声そのもの(参照 REF)を出力部で写し合成 = 出力部が運べる同一性の天井(f0 = f0hi の教師・包絡 = REF 自身)
  TAB_C1_S0 / R0 / ORACLE  既存のキャッシュ(STFT 入れ替え)
    uv run python conv_p0.py --ladder <scratchpad>/r4_spk --work <scratchpad>/c0_vctk --c1 ../results/c1_1 --out ../results/conv_p0/p0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import conv_c0 as C0

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--c1", required=True)
    ap.add_argument("--rvoc", default=str(ROOT / "results/rvoc3hi/snap/ema_330k.pt"))
    ap.add_argument("--f1", default=str(ROOT / "results/f0est3/last.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--males_per", type=int, default=5)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tag", default="P0")
    ap.add_argument("--causal", action="store_true", help="表の引きの平滑を因果の 5 フレーム平均にする(C3 の学習と同じ・製品の経路)")
    ap.add_argument("--f2", default=str(ROOT / "results/f2_1/last.pt"), help="--causal の c0(F2・因果)")
    ap.add_argument("--parts", default="avg,c0,med", help="--causal の部品: avg=表の引きの因果平均・c0=F2・med=f0 中央値を最初の 10s(分解診断用)")
    ap.add_argument("--g", default="", help="C3 の G(train_c3 の last.pt)を通す。--causal が前提・条件名は <tag>_G")
    a = ap.parse_args()
    import torch
    import c1_content as CC
    import f0est as FE
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_f0est as TF
    import train_rvoc as TR
    from train_dec2 import load48
    C0.LAD["path"] = a.ladder
    g = C0.G()
    wd = Path(a.work)
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    fems, males = J["fems"], J["males"]
    pairs = [(males[(i * a.males_per + j) % len(males)], f) for i, f in enumerate(fems) for j in range(a.males_per)]
    if a.n:
        pairs = pairs[:a.n]
    dev = "cuda"
    c1d = Path(a.c1)
    st = torch.load(c1d / "last.pt", map_location="cpu", weights_only=False)
    net = CC.C1(st["cfg"]["k"], st["cfg"]["ch"], tuple(st["cfg"]["dils"])).to(dev).eval()
    net.load_state_dict(st["net"])
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(c1d / "codebook.pt").to(dev)
    K = Cb.shape[0]
    sf1 = torch.load(a.f1, map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(sf1["net"])
    f1.eval()
    f1front = FE.Front().to(dev)
    sr_ = torch.load(a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"])
    gen.eval()
    sm_w = TR.ckpt_env_smooth(sr_, None)

    f2 = None
    parts = set(a.parts.split(",")) if (a.causal or a.g) else set()
    if a.g:
        parts = {"avg", "c0", "med"}
    if "c0" in parts:
        import lvl as LV
        sf2 = torch.load(a.f2, map_location="cpu", weights_only=False)
        f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev)
        f2.load_state_dict(sf2["net"])
        f2.eval()

    @torch.no_grad()
    def c1_post(x: np.ndarray, n: int) -> np.ndarray:
        xp = torch.from_numpy(T1.prime(x.astype(np.float32)))[None].to(dev)
        lg = net(mfront(xp))[..., -(len(x) // N.HOP):]
        p = lg.softmax(1)[0].T.cpu().numpy()
        return np.pad(p, ((0, max(0, n - len(p))), (0, 0)), mode="edge")[:n]

    def smooth3(E: np.ndarray) -> np.ndarray:
        e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
        return sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]

    def analyse(x: np.ndarray, f0: np.ndarray):
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        env = smooth3(PA.envelope(xa, f0))
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()
        return env, per

    def render(env_c: np.ndarray, f0: np.ndarray, per: np.ndarray) -> np.ndarray:
        """env_c [25, n](c0 = 生の値)・f0 [n](Hz・0 = 無声)・per [4, n] → 波形(DELAY 除去)。"""
        cc = env_c.copy()
        cc[0] -= PA.C0_SIL
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        with torch.no_grad():
            y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()
        return y[N.DELAY:]

    # 目標側(女声 45 人): 参照 REF から表・f0 の中央値・周期性の平均・写し合成(P0_COPY)
    tg: dict = {}
    for s in fems:
        RF = np.load(wd / "sig" / f"{s}__REF.npz")
        x = RF["x"].astype(np.float32)
        n = len(x) // N.HOP
        x = x[:n * N.HOP]
        f0r, _ = H.teacher_f0(x.astype(np.float64).astype(np.float32), n)
        env, per = analyse(x, f0r)
        p = c1_post(x.astype(np.float64), n)
        T = C0.table(env[1:25], p.argmax(1), K, Cb.cpu().numpy())
        vo = f0r > 0
        tg[s] = {"T": T, "mu": float(np.median(np.log(f0r[vo]))), "per": per[:, vo].mean(1) if vo.any() else per.mean(1)}
        fo = wd / "wav" / f"{s}__{a.tag}_COPY.wav"
        if not fo.exists():
            sf.write(fo, g.norm(render(env, f0r, per)).astype(np.float32), 48000)
    gnet = None
    if a.g:
        import train_c3 as C3
        sg = torch.load(a.g, map_location="cpu", weights_only=False)
        gnet = C3.ConvG(sg["cfg"]["K"], sg["cfg"]["ch"], tuple(sg["cfg"]["dils"]), sg["cfg"]["d_film"], sg["cfg"].get("m_basis", 0)).to(dev)
        gnet.load_state_dict(sg["G"], strict=False)
        gnet.eval()
        a.causal = True
    names = ["R0", "TAB_C1_S0", f"{a.tag}_TAB", f"{a.tag}_COPY"] + ([f"{a.tag}_G"] if a.g else [])
    for m, f in pairs:
        us = [u for u in C0.SENTS if C0.utt(m, u).exists() and C0.utt(f, u).exists()]
        xs = np.concatenate([g.norm(g.trim(load48(str(C0.utt(m, u))).astype(np.float64))) for u in us]).astype(np.float32)
        n = len(xs) // N.HOP
        xs = xs[:n * N.HOP]
        f0s = TF.infer(f1front, f1, xs, dev)
        f0s = np.pad(f0s, (0, max(0, n - len(f0s))))[:n].astype(np.float32)
        env_s, _ = analyse(xs, f0s)
        if f2 is not None:
            with torch.no_grad():
                c0n = f2(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(xs)[None].to(dev)], 1)))[0].cpu().numpy()
            env_s = env_s.copy()
            env_s[0] = c0n[:env_s.shape[1]] * 10 + PA.C0_SIL
        p = c1_post(xs.astype(np.float64), n)
        if "avg" in parts:
            import train_c3 as C3
            seq = C3.causal_avg(torch.from_numpy(tg[f]["T"][:, p.argmax(1)].astype(np.float32))[None], C3.W_SM)[0].numpy()
        else:
            seq = C0.smooth(tg[f]["T"][:, p.argmax(1)])
        env_c = np.concatenate([env_s[:1], seq], 0)
        vo = f0s > 0
        if "med" in parts:
            v10 = vo[:2000]
            med = float(np.median(np.log(f0s[:2000][v10]))) if v10.any() else 0.0
        else:
            med = float(np.median(np.log(f0s[vo]))) if vo.any() else 0.0
        f0o = np.where(vo, np.exp(np.log(np.maximum(f0s, 1.0)) - med + tg[f]["mu"]), 0.0).astype(np.float32)
        per_o = np.where(vo[None], tg[f]["per"][:, None], 0.0).astype(np.float32)
        fo = wd / "wav" / f"{m}__{f}__{a.tag}_TAB.wav"
        sf.write(fo, g.norm(render(env_c, f0o, per_o)).astype(np.float32), 48000)
        if gnet is not None:
            import train_c3 as C3
            with torch.no_grad():
                dv = lambda z: torch.from_numpy(np.asarray(z, np.float32)).to(dev)
                e_s = dv(seq / 10)[None]
                c0n = dv((env_s[0] - PA.C0_SIL) / 10)[None, None]
                lf = dv(np.where(f0o > 0, np.log(np.maximum(f0o, 1.0) / 200.0), 0.0))[None, None]
                vv = dv((f0o > 0).astype(np.float32))[None, None]
                pm = dv(tg[f]["per"])[None, :, None].expand(-1, -1, n)
                film = torch.cat([dv(tg[f]["T"] / 10).mean(1)[None], dv([tg[f]["mu"]])[None], dv(tg[f]["per"])[None]], 1)
                dc, dp = gnet(e_s, dv(p.T)[None], lf, c0n, vv, film)
                cond = C3.build_cond(c0n, e_s, dc, lf, vv, pm, dp)
                fo_t = dv(f0o)[None]
                y = gen(cond, TR.excitation(fo_t, n * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
            sf.write(wd / "wav" / f"{m}__{f}__{a.tag}_G.wav", g.norm(y).astype(np.float32), 48000)
        cp = wd / "wav" / f"{m}__{f}__{a.tag}_COPY.wav"
        if not cp.exists():
            cp.write_bytes((wd / "wav" / f"{f}__{a.tag}_COPY.wav").read_bytes())
    print("render done", flush=True)
    import idloss as ID
    cvm = ID.ContentVec(dev)
    agree: dict = {nm: [] for nm in names if nm not in ("R0", f"{a.tag}_COPY")}
    with torch.no_grad():
        for m, f in pairs:
            us = [u for u in C0.SENTS if C0.utt(m, u).exists() and C0.utt(f, u).exists()]
            xs = np.concatenate([g.norm(g.trim(load48(str(C0.utt(m, u))).astype(np.float64))) for u in us]).astype(np.float32)
            ux = (cvm(torch.from_numpy(xs)[None].to(dev)) @ Cb.T).argmax(-1)[0]
            for nm in agree:
                y, _ = sf.read(wd / "wav" / f"{m}__{f}__{nm}.wav", dtype="float32")
                uy = (cvm(torch.from_numpy(y)[None].to(dev)) @ Cb.T).argmax(-1)[0]
                k = min(len(ux), len(uy))
                agree[nm].append(float((ux[:k] == uy[:k]).float().mean()))
    cva = {nm: round(float(np.mean(v)), 4) for nm, v in agree.items()}
    print("ContentVec 単位の一致率(出力 vs 元の男声):", cva, flush=True)
    rep = {"n_pairs": len(pairs), "rvoc": a.rvoc, "f1": a.f1, "c1_step": st["step"], "cv_unit_agree": cva}
    import eval_nvoc as E
    am = {}
    for nm in names:
        if nm in ("R0", "TAB_C1_S0"):
            continue
        am[nm] = round(float(np.mean([E.amline(sf.read(wd / "wav" / f"{m}__{f}__{nm}.wav", dtype="float32")[0]) for m, f in pairs[::5]])), 3)
    rep["am_line_db(1/5 の組)"] = am
    print("変調の線:", am, flush=True)
    rep.update(C0.score(wd, pairs, fems, names, (f"{a.tag}_TAB", "TAB_C1_S0", "R0")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k in names}, ensure_ascii=False)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
