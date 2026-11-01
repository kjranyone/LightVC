"""D20: c0 の細かい動き(≥ 4Hz)を目標の単位ごとの強さ(構音目標表の c0 の行・参照の要約)から作る(学習なし)。
女声 → 女声(x): TAB = 製品(c0 は元の声)・C0LP = c0 を元の声の 4Hz 低域通過だけ(元の声の細かい動きを消す)・C0T = 4Hz 低域(元の声)+ 4Hz 高域(目標の c0 表を元の単位で引いた列)・C0TF = 元の声の平均 + 目標の c0 表の列(元の声の動きなし)。
c0 の表の列は包絡と同じ因果 3 平均(平滑なしは単位の切り替えで段差 → 線 +1.35dB で傷)。
自己変換(self): COPY・C0T(4Hz 低域は本人・高域は本人の参照の c0 表)・TAB・TAB_C0T。低域通過は診断のため零位相(製品は因果に置き換え)。
旧: D18: 女声 → 女声の変換で、元の声の内容に揃った韻律を、目標の参照の要約統計(平均・標準偏差のみ)で寄せる効果(学習なし)。
TAB = 製品(f0 は中央値の移動・c0 は元の声のまま)・TAB_C0M = c0 の平均(活性フレーム)を目標の参照の平均へ・TAB_F0R = 有声の log f0 の標準偏差を目標の参照へ・TAB_BOTH = 両方。
旧: D16 = artic_vc rev0 レビューの P-1(学習なし・推論のみ・参照のフレームは使わない)。作り方の不整合(平滑の段差)を入れないよう、全て平滑なしの値で組む。
自己変換(目標自身の別発話・包絡だけ替え、c0・f0・周期性は実の分析):
  COPY = 実の包絡 er・TAB = 表の引き(因果 3 平均 = 製品)・TAB_RAW = 表の引き(平滑なし)
  P1a = er + (T − INS)[u](COPY + 表の誤差・平滑なし)・P1b = er + B(別の発話の平滑の段差 B = 因果5(INS[u]) − INS[u])
  P1c = TAB + g(g = その発話の er − T[u] の時間平均 = 発話全体のずれ)
女声 → 女声(元の声 = 次の目標の話者の発話・f0 は中央値の移動・c0 は元の声の実の c0・周期性は目標の平均):
  TAB = 表の引き(因果 3 平均)・P1d = T_tgt[u_src] + (er_src − INS_src[u_src])(元の声の自然な動きを目標の表へ載せ替え・平滑なし)
出力: --out_dir/self と --out_dir/x に d4b の形式(cen_XXX.wav・pYYYY_tXXX_<cond>.wav)。
    uv run python d16_p1_checks.py --out_dir <scratchpad>/d16 --out ../results/conv_p0/d16.json
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
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_tg", type=int, default=150)
    ap.add_argument("--n_utt", type=int, default=3)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import eval_nvoc as E
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    dev = "cuda"
    od_s, od_x = Path(a.out_dir) / "self", Path(a.out_dir) / "x"
    od_s.mkdir(parents=True, exist_ok=True); od_x.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]

    def sm3(Ev):
        e = np.pad(Ev, ((0, 0), (1, 1)), mode="edge")
        return sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]

    def analyse(x, f0):
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        return sm3(PA.envelope(xa, f0)), PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()

    @torch.no_grad()
    def units(x):
        n = len(x) // N.HOP
        u = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].argmax(1)[0].cpu().numpy()
        return np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]

    ca = lambda z, w: C3.causal_avg(torch.from_numpy(z.astype(np.float32))[None], w)[0].numpy()

    @torch.no_grad()
    def render(c0, env24, f0, per):
        cc = np.concatenate([c0[None] - PA.C0_SIL, env24], 0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    hs = TR.held_speakers()
    allspk = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in (hs | TTS_LEAK))
    lda_tr = {allspk[i] for i in np.random.RandomState(0).permutation(len(allspk))[:1200]}
    cand = [s_ for s_ in allspk if s_ not in lda_tr]
    pool = [cand[i] for i in np.random.RandomState(5).permutation(len(cand))[: a.n_tg * 2]]
    tg: list = []
    for s in pool:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        if len(rest) < a.n_utt + 2:
            continue
        xr = np.concatenate([load48(w) for w in ref]); xr = xr[: len(xr) // N.HOP * N.HOP]
        f0r, _ = H.teacher_f0(xr, len(xr) // N.HOP)
        envr, perr = analyse(xr, f0r)
        ur = units(xr)
        T = C0.table(envr[1:25, : len(ur)], ur, K, Cb)
        Tc0 = C0.table(envr[0:1, : len(ur)] - envr[0, : len(ur)].mean(), ur, K, Cb)[0]
        vo = f0r > 0
        utts = []
        for w in rest[: a.n_utt]:
            x = load48(w)[: 8 * 48000]
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            if n < 200:
                continue
            f0t, _ = H.teacher_f0(x, n)
            env, per = analyse(x, f0t)
            u = units(x)
            er = env[1:25, :n]
            INS = C0.table(er, u, K, Cb)
            utts.append({"x": x, "er": er, "c0": env[0, :n], "f0": f0t[:n], "per": per[:, :n], "u": u, "INS": INS})
        if not utts:
            continue
        cen = np.concatenate([load48(w) for w in rest[a.n_utt:]])[: 25 * 48000]
        act_r = (envr[0] - PA.C0_SIL) / 10 > 1.0
        tg.append({"spk": s, "T": T, "mu": float(np.median(np.log(f0r[vo]))), "sd": float(np.log(f0r[vo]).std()), "c0m": float(envr[0][act_r].mean()), "Tc0": Tc0, "per_mean": perr[:, vo].mean(1), "utts": utts, "cen": cen})
        print("prep", len(tg), s, flush=True)
        if len(tg) >= a.n_tg:
            break
    S = len(tg)
    am: dict = {}
    pi = 0
    from scipy.signal import butter, sosfiltfilt
    lp = lambda z, fc: sosfiltfilt(butter(4, fc, "lowpass", fs=200, output="sos"), z)
    for ti, t in enumerate(tg):
        sf.write(od_x / f"cen_{ti:03d}.wav", t["cen"], 48000)
        sf.write(od_s / f"cen_{ti:03d}.wav", t["cen"], 48000)
        T, Tc0 = t["T"], t["Tc0"]
        src = tg[(ti + 1) % S]
        for k_ in range(len(t["utts"])):
            su = src["utts"][k_ % len(src["utts"])]
            vo_s = su["f0"] > 0
            lfs = np.log(np.maximum(su["f0"], 1.0))
            med = float(np.median(lfs[vo_s])) if vo_s.any() else t["mu"]
            f0m = np.where(vo_s, np.exp(lfs - med + t["mu"]), 0.0)
            perc = np.repeat(t["per_mean"][:, None], len(su["u"]), 1)
            tab = ca(T[:, su["u"]], 3)
            c0s = su["c0"]
            tseq = ca(Tc0[su["u"]][None], 3)[0]
            c0_lp = lp(c0s, 4.0)
            conds = {"TAB": c0s, "C0LP": c0_lp, "C0T": c0_lp + (tseq - lp(tseq, 4.0)), "C0TF": c0s.mean() + tseq}
            for c, c0_ in conds.items():
                y = render(c0_, tab, f0m, perc)
                sf.write(od_x / f"p{pi:04d}_t{ti:03d}_{c}.wav", y, 48000)
                am.setdefault("x_" + c, []).append(E.amline(y))
            sf.write(od_x / f"p{pi:04d}_t{ti:03d}_SRC.wav", su["x"], 48000)
            ut = t["utts"][k_]
            useq = ca(Tc0[ut["u"]][None], 3)[0]
            c0o = lp(ut["c0"], 4.0) + (useq - lp(useq, 4.0))
            tabs = ca(T[:, ut["u"]], 3)
            sconds = {"COPY": (ut["er"], ut["c0"]), "C0T": (ut["er"], c0o), "TAB": (tabs, ut["c0"]), "TAB_C0T": (tabs, c0o)}
            for c, (ev, c0_) in sconds.items():
                y = render(c0_, ev, ut["f0"], ut["per"])
                sf.write(od_s / f"p{pi:04d}_t{ti:03d}_{c}.wav", y, 48000)
                am.setdefault("s_" + c, []).append(E.amline(y))
            sf.write(od_s / f"p{pi:04d}_t{ti:03d}_SRC.wav", ut["x"], 48000)
            pi += 1
        print("render", ti, {k: round(float(np.mean(v)), 3) for k, v in am.items()}, flush=True)
    rep = {"n_targets": S, "n_utts": pi, "am_line_db": {k: round(float(np.mean(v)), 3) for k, v in am.items()}}
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
