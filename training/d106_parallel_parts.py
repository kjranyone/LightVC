"""D106(学習なし・診断専用): 男声 → 女声で残る差の帰属。パラレル文(D80: 男声 M の fK と女声 TTS A の tK_neutral = 同じ文)で、
製品の変換(M → A・γ0.7 + 色・周期性 F4・Python の製品経路・出力部 --rvoc_ckpt)の 1 要素だけを A の本物(DTW で M の時間へ写す・診断専用)に替える:
  AWARP = A 自身の全部(F3 の包絡・F2 の c0・教師 f0・F4)を M の時間へ伸縮(男声の情報は時間の流れだけ)・ASELF = A 自身の全部(伸縮なし = 製品の写し合成)・CONV = 製品の変換・XF0 = f0 を A の教師 f0(A の有声)・XC0 = c0 を A の F2・XPER = 周期性を A の F4・XDYN = 包絡の動きを A の F3 − 表 A(伸縮・色づけなし)・XALL = 4 つとも。
C0E(D119)= c0 を「男声の各フレームのエネルギー(F2 の c0 + F3 の形)を保ち、出力の包絡の形と整合する水準」に解き直す: c0_out = c0_M + (√N/2)(log Σ_b e^{2 形_M} − log Σ_b e^{2 形_out})(因果・定数なし)・C0EX = 同じ式で形_out の代わりに A の本物の包絡(診断: 整合の相手が正しい形なら)。
XCD・XCF・…・XDFP(D126)= {c0・動き・f0・周期性} のうち名前の文字(c・d・f・p)を A の本物に替えた 2 つ組・3 つ組(全要因の残り 10 通り)。
XSURD / XSURCD(D149 = 分布の生成器の事前検査 PC-F1)= 動き(と c0)を A 自身のものの位相だけ乱したもの(統計は本人・時間の並びは内容と揃わない・同じ乱数を全次元に)に替える。D = 動きだけ・CD = c0 と動き。XDYN(本人の実際の動き)/ XCD と比べ、統計だけで同一性が出るかを見る。
XREAD / XREADS(D129・オーナー 10-12: 参照を注意機構で読むのは生成に限り可 = その情報の上限の検査・学習なし)= 男声の内容の鍵(C1 の事後の平方根 + 直前 4 フレームの平均・正規化)で目標の登録の各フレームの鍵へ柔らかい注意(温度 0.05 / 0.02)を掛け、目標の F3 包絡と相対 c0 を読み出す(c0 = 読み + 操作者の平均水準)。E = 包絡だけ・C = c0 だけ。有効フレーム数(1/Σw²)と、読み出しと A の本物(DTW)の相関を _read_log.json に。
XMEQ / XMEQC / XMEQD(D128)= 変調の等化: 操作者の c0 と(伸縮 + 色づけ後の)動きの軌道を、因果な最小位相 FIR(各次元の変調のパワースペクトルの比の平方根・登録の有声区間・Welch 64 フレーム・利得 0.25〜4)で目標の登録の変調スペクトルへ合わせる(時間の並びと位相は操作者のまま・登録の定数だけ)。C = c0 だけ・D = 動きだけ。
XCDB / XCB / XDB(D127)= c0・動き(両方 / c0 だけ / 動きだけ)を目標 A ではなく第三の女声 B(同じ文・DTW で M の時間へ・B の表は B の別の 9 文の F3)から = 『目標の時間パターン』でなく『女声一般の時間パターン』で足りるか。
--align(D155)= DTW の特徴(lm = 既定の log-mel・lmn = 発話ごとに平均分散をそろえた log-mel・c1 = C1 の事後の平方根・c1lm = 両方・rr = train_rr の交差ペアと同じ log-mel の DCT 1..20 のコサイン)。--closure_only = 描画せず 4 つの特徴の三角閉包の誤差(M→B→A→M の有声の平均ずれ)と kNN の組のずれだけを _closure_*.json に。
XSELFK / XSELFRT(D154・整列の交絡の検査)= 操作者自身の動き dyn_c を DTW の経路で引き直す。K = M と kNN-VC(M)(--knn_dir・同じ時刻)の log-mel DTW(一定のずれは中央値で除く)= 真の対応が恒等の組で測った異性間 DTW の誤差だけ。RT = M→B→A→M の 3 回の DTW の合成。…CD = c0 も同じ経路で引き直す(D157)。TAB0 = 表だけ(動きなし・c0 は操作者)。XC0PD / XC0PD2(D168)= 操作者の c0 + g ×(目標ごとのリッジ − 話者一般のリッジ)・g = 1 / 2。XC0P / XC0PI(D167)= c0 の低域を、目標の 9 文(XC0P)/ 目標以外の 30 人(XC0PI・文 k を除く)で当てはめたリッジ(C1 の事後とその因果な移動平均 → 低域)で操作者の内容から予測。XC0T(D166)= 句の始まりからの経過 τ の c0 の型を操作者のものから目標のもの(登録の参照)へ差し替え(因果)。XC0SS / XC0SC(D164)= 操作者の c0 の低域を登録の参照の低域の振れ幅の比で伸縮(SS = ゼロ位相・SC = 因果)。XC0S / XC0F(D162)= c0 を 3 Hz で低域・高域に分け、目標本人の低域 + 操作者の高域 / 操作者の低域 + 目標本人の高域。XCD_STY<sty> / ASELF_STY<sty>(D161)= 目標の同じ文の別スタイルの c0 + 動きを写す / そのスタイルの写し合成(--only に _STY を含める)。経路の統計を _self_log.json に。
C0TAB(D121)= c0 も包絡と同じ『目標の表 + 操作者の残差』に: c0_out = c0_M − Σp T0_操作者[u] + Σp T0_目標[u](T0 = 単位ごとの c0 の話者平均からの相対値・目標 = CheapTrick・操作者 = F2)。
DTW = 因果 log-mel(MelFront)の系列のユークリッド距離・M のフレームごとに対応する A のフレームの値。出力 = <out>/<M>_<A>_tK_<cond>.wav(D105 --pool tts で距離を測る)。
    uv run python d106_parallel_parts.py --out_dir <scratchpad>/d106 --rvoc_ckpt ../results/rvtab_f3/snap/ema_380k.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"
MALE = ROOT / "data/male_tts_v4l"
MPAR = ROOT / "data/male_tts_v4l_par"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rvoc_ckpt", required=True)
    ap.add_argument("--n_male", type=int, default=24)
    ap.add_argument("--texts", default="0,1,2,3,4")
    ap.add_argument("--only", default="", help="この条件だけ書く(, 区切り)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--knn_dir", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--align", default="lm", choices=["lm", "lmn", "c1", "c1lm", "rr"])
    ap.add_argument("--closure_only", action="store_true")
    ap.add_argument("--closure_hows", default="lm,lmn,c1,c1lm")
    a = ap.parse_args()
    import librosa
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
    torch.set_num_threads(8)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
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
        lm = m0[0].cpu().numpy()[:, :n].astype(np.float64)
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        ft, _ = H.teacher_f0(x, n)
        return {"x": x, "n": n, "f0": pad1(f0f), "c0": pad1(c0f), "e3": pad(e3), "p4": pad(p4), "p": pad(p), "u": pad(p).argmax(0), "lm": pad(lm), "ft": pad1(ft[:n].astype(np.float64))}

    def enroll(xs: list, use_f3: bool) -> dict:
        frs = [front(x) for x in xs]
        E = np.concatenate([fr["e3"] if use_f3 else cheap(fr) for fr in frs], 1)
        U = np.concatenate([fr["u"] for fr in frs])
        T = C0.table(E, U, K, Cb)
        F0 = np.concatenate([fr["f0"] if use_f3 else fr["ft"] for fr in frs])
        v = F0 > 0
        r = E - T[:, U]
        rv = r[:, v] - r[:, v].mean(1, keepdims=True)
        P4 = np.concatenate([fr["p4"] for fr in frs], 1)
        C0v = np.concatenate([fr["c0"] for fr in frs])[v]
        C0a = np.concatenate([fr["c0"] if use_f3 else cheap0(fr) for fr in frs])
        T0 = C0.table((C0a - C0a.mean())[None], U, K, Cb)[0]
        Vf = np.concatenate([fr["f0"] if use_f3 else fr["ft"] for fr in frs]) > 0
        mod_c0 = modpsd(C0a[None], Vf)[0]
        mod_r = modpsd(E - T[:, U], Vf)
        Pc = np.concatenate([fr["p"] for fr in frs], 1)
        Kc = np.concatenate([ctxkey(fr["p"]) for fr in frs], 1)
        return {"E": E, "P": Pc, "Kc": Kc, "C0a": C0a, "R": E - T[:, U], "V": Vf, "mod_c0": mod_c0, "mod_r": mod_r, "T0": T0, "T": T, "mu": float(np.median(np.log(F0[v]))), "cov": rv @ rv.T / rv.shape[1], "p4mean": P4[:, v].mean(1), "c0m": float(C0v.mean()), "c0s": float(C0v.std())}

    import scipy.signal as ss
    NP_ = 64

    def ctxkey(p: np.ndarray) -> np.ndarray:
        sp = np.sqrt(np.maximum(p, 0))
        cs = np.cumsum(np.pad(sp, ((0, 0), (5, 0))), 1)
        prev = (cs[:, 4:-1] - cs[:, :-5]) / 4
        k_ = np.concatenate([sp, prev], 0)
        return k_ / (np.linalg.norm(k_, axis=0, keepdims=True) + 1e-9)

    def attend(q: np.ndarray, Kr: np.ndarray, vals: np.ndarray, tau: float) -> tuple:
        qt = torch.from_numpy(q.T).float().to(dev); kt = torch.from_numpy(Kr).float().to(dev)
        w = torch.softmax(qt @ kt / tau, 1)
        eff = float((1.0 / (w ** 2).sum(1)).median())
        return (w @ torch.from_numpy(vals.T).float().to(dev)).cpu().numpy().T.astype(np.float64), eff

    def modpsd(z: np.ndarray, v: np.ndarray) -> np.ndarray:
        acc, cnt = np.zeros((z.shape[0], NP_ // 2 + 1)), 0
        t = 0
        while t < len(v):
            if not v[t]:
                t += 1
                continue
            u = t
            while u < len(v) and v[u]:
                u += 1
            if u - t >= NP_:
                _, P = ss.welch(z[:, t:u] - z[:, t:u].mean(1, keepdims=True), nperseg=NP_, axis=1)
                acc += P * (u - t); cnt += u - t
            t = u
        return acc / max(cnt, 1)

    def meq(z: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        out = np.empty_like(z)
        fq = np.linspace(0, 1, NP_ // 2 + 1)
        for d in range(z.shape[0]):
            g = np.sqrt(np.maximum(dst[d], 1e-12) / np.maximum(src[d], 1e-12)).clip(0.25, 4.0)
            h = ss.minimum_phase(ss.firwin2(127, fq, g ** 2), method="homomorphic")
            m = z[d].mean()
            out[d] = ss.lfilter(h, 1.0, z[d] - m) + m
        return out

    def cheap0(fr: dict) -> np.ndarray:
        xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, fr["ft"])[:1], ((0, 0), (1, 1)), mode="edge")
        return ((sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[0, :fr["n"]] - PA.C0_SIL).astype(np.float64) / 10

    def cheap(fr: dict) -> np.ndarray:
        xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, fr["ft"]), ((0, 0), (1, 1)), mode="edge")
        return (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :fr["n"]][1:25].astype(np.float64)

    dctm = np.asarray(PA._np_consts()[1])[1:25]
    sq = np.sqrt(N.N_MEL)

    def c0e(src: np.ndarray, out: np.ndarray) -> np.ndarray:
        ls = lambda e: np.log(np.exp(2 * (dctm.T @ e)).sum(0))
        return sq / 2 * (ls(src) - ls(out)) / 10

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    STY = ["neutral", "breathy", "low_tension", "soft", "warm"]
    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)[: a.n_male]
    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STY))
    random.Random(80).shuffle(fems)
    texts = [int(t) for t in a.texts.split(",")]
    rlog: list = []
    P_si: dict = {}
    slog: list = []
    si, sk = (int(z) for z in a.shard.split("/"))

    def akey(fr: dict, how: str) -> np.ndarray:
        lmn = (fr["lm"] - fr["lm"].mean(1, keepdims=True)) / (fr["lm"].std(1, keepdims=True) + 1e-6)
        sp = np.sqrt(np.maximum(fr["p"], 0))
        if how == "rr":
            import scipy.fft as sfft
            return sfft.dct(fr["lm"], axis=0, norm="ortho")[1:21]
        return {"lm": fr["lm"], "lmn": lmn, "c1": sp, "c1lm": np.concatenate([sp, lmn / np.sqrt(lmn.shape[0])], 0)}[how]

    def dtwp(X_: np.ndarray, Y_: np.ndarray, how: str = "") -> np.ndarray:
        if (how or a.align) == "rr":
            un = lambda z: z / (np.linalg.norm(z, axis=0, keepdims=True) + 1e-9)
            return librosa.sequence.dtw(C=1 - un(X_).T @ un(Y_))[1]
        return librosa.sequence.dtw(X=X_, Y=Y_, metric="euclidean")[1]

    def dtwj(X_: np.ndarray, Y_: np.ndarray, how: str = "") -> np.ndarray:
        wq = dtwp(X_, Y_, how)
        jq = np.zeros(X_.shape[1], int)
        for i_, j_ in wq[::-1]:
            jq[i_] = j_
        return jq

    clog: list = []
    for mi, M in enumerate(males):
        if mi % sk != si:
            continue
        A = fems[mi]
        mfiles, tot = [], 0.0
        for f in sorted(MALE.glob(f"{M}/t*.wav")):
            if tot >= 22.0:
                break
            x = load48(f).astype(np.float32); mfiles.append(x); tot += len(x) / 48000
        O_ = None if a.closure_only else enroll(mfiles, use_f3=True)
        if O_ is not None and ("XC0SS" in (a.only or "") or "XC0SC" in (a.only or "") or "XC0T" in (a.only or "")):
            O_["_frs"] = [front(z) for z in mfiles]
        for k in texts:
            tk = f"t{k:02d}"
            if a.only and all((od / f"{M}_{A}_{tk}_{c_}.wav").exists() for c_ in a.only.split(",")):
                continue
            if a.closure_only:
                fm = front(load48(MPAR / M / f"f{k:02d}.wav").astype(np.float32))
                fa = front(load48(FEM / A / f"{tk}_neutral.wav").astype(np.float32))
                fb = front(load48(FEM / fems[len(males) + mi] / f"{tk}_neutral.wav").astype(np.float32))
                fk = front(load48(Path(a.knn_dir) / f"{M}_{A}_{tk}_KNN.wav").astype(np.float32))
                ii, vm = np.arange(fm["n"]), fm["f0"] > 0
                rec = {"pair": f"{M}_{A}_{tk}"}
                for how in a.closure_hows.split(","):
                    km, ka, kb = akey(fm, how), akey(fa, how), akey(fb, how)
                    rt = dtwj(ka, km, how)[dtwj(kb, ka, how)[dtwj(km, kb, how)]]
                    jk = dtwj(km, akey(fk, how), how)
                    jk = jk - int(np.median(jk - ii))
                    rec[how] = {"rt_mae_v": round(float(np.abs(rt - ii)[vm].mean()), 2), "rt_p90_v": round(float(np.percentile(np.abs(rt - ii)[vm], 90)), 1),
                                "k_mae_v": round(float(np.abs(jk - ii)[vm].mean()), 2)}
                clog.append(rec)
                print(rec, flush=True)
                (od / f"_closure_{si}-{sk}.json").write_text(json.dumps(clog, indent=1))
                continue
            T_ = enroll([load48(FEM / A / f"t{i:02d}_neutral.wav").astype(np.float32) for i in range(10) if i != k], use_f3=False)
            TA3 = enroll([load48(FEM / A / f"t{i:02d}_neutral.wav").astype(np.float32) for i in range(10) if i != k], use_f3=True)
            fm = front(load48(MPAR / M / f"f{k:02d}.wav").astype(np.float32))
            fa = front(load48(FEM / A / f"{tk}_neutral.wav").astype(np.float32))
            wp = dtwp(akey(fm, a.align), akey(fa, a.align))
            wp = wp[::-1]
            j = np.zeros(fm["n"], int)
            for i_, j_ in wp:
                j[i_] = j_
            n = fm["n"]
            f0o = np.where(fm["f0"] > 0, fm["f0"] * np.exp(T_["mu"] - O_["mu"]), 0.0)
            per_c = fm["p4"] - O_["p4mean"][:, None] + T_["p4mean"][:, None]
            M_ = msqrt(T_["cov"], 0.5) @ msqrt(W @ O_["cov"] @ W.T, -0.5)
            tab = T_["T"] @ fm["p"]
            dyn_c = 0.7 * M_ @ W @ (fm["e3"] - O_["T"] @ fm["p"])
            fa_f0 = np.where(fa["ft"][j] > 0, fa["ft"][j], 0.0)
            fa_c0 = fa["c0"][j]
            fa_per = fa["p4"][:, j]
            fa_dyn = 0.7 * (fa["e3"][:, j] - TA3["T"] @ fa["p"][:, j])
            conds = {"CONV": (fm["c0"], tab + dyn_c, f0o, per_c), "XF0": (fm["c0"], tab + dyn_c, fa_f0, per_c), "XC0": (fa_c0, tab + dyn_c, f0o, per_c),
                     "XPER": (fm["c0"], tab + dyn_c, f0o, fa_per), "XDYN": (fm["c0"], tab + fa_dyn, f0o, per_c), "XALL": (fa_c0, tab + fa_dyn, fa_f0, fa_per)}
            c0map = (fm["c0"] - O_["c0m"]) / O_["c0s"] * TA3["c0s"] + TA3["c0m"]
            conds["AWARP"] = (fa["c0"][j], fa["e3"][:, j], fa_f0, fa_per)
            conds["ASELF"] = (fa["c0"], fa["e3"], np.where(fa["ft"] > 0, fa["ft"], 0.0), fa["p4"])
            conds["C0MAP"] = (c0map, tab + dyn_c, f0o, per_c)
            conds["C0E"] = (fm["c0"] + c0e(fm["e3"], tab + dyn_c), tab + dyn_c, f0o, per_c)
            from itertools import combinations
            parts = {"c": (fm["c0"], fa_c0), "d": (tab + dyn_c, tab + fa_dyn), "f": (f0o, fa_f0), "p": (per_c, fa_per)}
            for r_ in (2, 3):
                for sub in combinations("cdfp", r_):
                    conds["X" + "".join(sub).upper()] = tuple(parts[q][1 if q in sub else 0] for q in "cdfp")
            if any(c_ in (a.only or "XCDB") for c_ in ("XCDB", "XCB", "XDB")):
                Bs = fems[len(males) + mi]
                TB3 = enroll([load48(FEM / Bs / f"t{i:02d}_neutral.wav").astype(np.float32) for i in range(10) if i != k], use_f3=True)
                fb = front(load48(FEM / Bs / f"{tk}_neutral.wav").astype(np.float32))
                wpb = dtwp(akey(fm, a.align), akey(fb, a.align))
                jb = np.zeros(fm["n"], int)
                for i_, j_ in wpb[::-1]:
                    jb[i_] = j_
                fb_c0 = fb["c0"][jb]
                fb_dyn = 0.7 * (fb["e3"][:, jb] - TB3["T"] @ fb["p"][:, jb])
                conds["XCDB"] = (fb_c0, tab + fb_dyn, f0o, per_c)
                conds["XCB"] = (fb_c0, tab + dyn_c, f0o, per_c)
                conds["XDB"] = (fm["c0"], tab + fb_dyn, f0o, per_c)
            if "XSELF" in (a.only or ""):
                ii = np.arange(n)
                fk = front(load48(Path(a.knn_dir) / f"{M}_{A}_{tk}_KNN.wav").astype(np.float32))
                jk = dtwj(akey(fm, a.align), akey(fk, a.align))
                off = int(np.median(jk - ii))
                jk = np.clip(jk - off, 0, n - 1)
                rt = dtwj(akey(fa, a.align), akey(fm, a.align))[dtwj(akey(fb, a.align), akey(fa, a.align))[jb]]
                rd = lambda z: float(np.sqrt(((z - dyn_c) ** 2).mean() / (dyn_c ** 2).mean()))
                st = lambda q: {"ne": round(float((q != ii).mean()), 3), "mae": round(float(np.abs(q - ii).mean()), 2), "bias": round(float((q - ii).mean()), 2),
                                "p90": round(float(np.percentile(np.abs(q - ii), 90)), 1), "rep": round(float((np.diff(q) == 0).mean()), 3), "skip": round(float((np.diff(q) > 1).mean()), 3),
                                "mae_v": round(float(np.abs(q - ii)[f0o > 0].mean()), 2), "p90_v": round(float(np.percentile(np.abs(q - ii)[f0o > 0], 90)), 1)}
                slog.append({"pair": f"{M}_{A}_{tk}", "n": n, "knn_off": off, "K": st(jk), "RT": st(rt),
                             "B": {"rep": round(float((np.diff(jb) == 0).mean()), 3), "skip": round(float((np.diff(jb) > 1).mean()), 3)},
                             "drel_K": round(rd(dyn_c[:, jk]), 3), "drel_RT": round(rd(dyn_c[:, rt]), 3), "drel_B": round(rd(fb_dyn), 3), "drel_A": round(rd(fa_dyn), 3)})
                print(slog[-1], flush=True)
                (od / f"_self_log_{si}-{sk}.json").write_text(json.dumps(slog, indent=1))
                conds["XSELFK"] = (fm["c0"], tab + dyn_c[:, jk], f0o, per_c)
                conds["XSELFRT"] = (fm["c0"], tab + dyn_c[:, rt], f0o, per_c)
                conds["XSELFKCD"] = (fm["c0"][jk], tab + dyn_c[:, jk], f0o, per_c)
                conds["XSELFRTCD"] = (fm["c0"][rt], tab + dyn_c[:, rt], f0o, per_c)
            if "XMEQ" in (a.only or ""):
                dyn_meq = 0.7 * meq(M_ @ W @ (fm["e3"] - O_["T"] @ fm["p"]), modpsd(M_ @ W @ O_["R"], O_["V"]), TA3["mod_r"])
                c0_meq = meq(fm["c0"][None], O_["mod_c0"][None], TA3["mod_c0"][None])[0]
                conds["XMEQ"] = (c0_meq, tab + dyn_meq, f0o, per_c)
                conds["XMEQC"] = (c0_meq, tab + dyn_c, f0o, per_c)
                conds["XMEQD"] = (fm["c0"], tab + dyn_meq, f0o, per_c)
            if "XREAD" in (a.only or ""):
                qm = ctxkey(fm["p"])
                for nm_, tau in (("XREAD", 0.05), ("XREADS", 0.02)):
                    rd, eff = attend(qm, TA3["Kc"], np.concatenate([TA3["E"], (TA3["C0a"] - TA3["C0a"].mean())[None]], 0), tau)
                    c0r = rd[24] + O_["c0m"]
                    conds[nm_] = (c0r, rd[:24], f0o, per_c)
                    conds[nm_ + "E"] = (fm["c0"], rd[:24], f0o, per_c)
                    conds[nm_ + "C"] = (c0r, tab + dyn_c, f0o, per_c)
                    ev = fa["e3"][:, j]
                    rlog.append({"pair": f"{M}_{A}_{tk}", "cond": nm_, "eff": round(eff, 1),
                                 "corr_read": round(float(np.mean([np.corrcoef(rd[d], ev[d])[0, 1] for d in range(24)])), 3),
                                 "corr_tab": round(float(np.mean([np.corrcoef(tab[d], ev[d])[0, 1] for d in range(24)])), 3),
                                 "corr_conv": round(float(np.mean([np.corrcoef((tab + dyn_c)[d], ev[d])[0, 1] for d in range(24)])), 3)})
                    print(rlog[-1], flush=True)
                (od / "_read_log.json").write_text(json.dumps(rlog, indent=1))
            if "XSUR" in (a.only or ""):
                def surr(r_: np.ndarray, seed: int) -> np.ndarray:
                    n_ = r_.shape[1]
                    mu_ = r_.mean(1, keepdims=True)
                    Fr = np.fft.rfft(r_ - mu_, axis=1)
                    ph = np.exp(1j * np.random.default_rng(seed).uniform(0, 2 * np.pi, Fr.shape[1])); ph[0] = 1.0
                    return np.fft.irfft(Fr * ph[None], n=n_, axis=1) + mu_
                da = fa["e3"] - TA3["T"] @ fa["p"]
                ca = fa["c0"] - fa["c0"].mean()
                sd_ = surr(np.concatenate([da, ca[None]], 0), 1000 * mi + k)
                conds["XSURD"] = (fm["c0"], tab + 0.7 * sd_[:24][:, j], f0o, per_c)
                conds["XSURCD"] = (sd_[24][j] + fa["c0"].mean(), tab + 0.7 * sd_[:24][:, j], f0o, per_c)
            conds["TAB0"] = (fm["c0"], tab, f0o, per_c)
            if "XC0S" in (a.only or "") or "XC0F" in (a.only or ""):
                bl, al = ss.butter(2, 3.0 / 100.0)
                lo = lambda z: ss.filtfilt(bl, al, z)
                cm, ca = fm["c0"], fa_c0
                conds["XC0S"] = (lo(ca) + cm - lo(cm), tab + dyn_c, f0o, per_c)
                conds["XC0F"] = (lo(cm) + ca - lo(ca), tab + dyn_c, f0o, per_c)
            if "XC0SS" in (a.only or "") or "XC0SC" in (a.only or ""):
                bl, al = ss.butter(2, 3.0 / 100.0)
                lo = lambda z: ss.filtfilt(bl, al, z)
                loc = lambda z: ss.lfilter(bl, al, z - z[0]) + z[0]
                def slow_sd(fr_list: list, f: object) -> float:
                    return float(np.mean([np.std((f(fr["c0"]) - f(fr["c0"]).mean())[fr["f0"] > 0]) for fr in fr_list if (fr["f0"] > 0).sum() > 50]))
                if "_slow" not in O_:
                    O_["_slow"] = (slow_sd(O_["_frs"], lo), slow_sd(O_["_frs"], loc))
                tfr = [front(load48(FEM / A / f"t{i:02d}_neutral.wav").astype(np.float32)) for i in range(10) if i != k]
                aT = slow_sd(tfr, lo) / O_["_slow"][0]
                aTc = slow_sd(tfr, loc) / O_["_slow"][1]
                cm = fm["c0"]
                sm, smc = lo(cm), loc(cm)
                conds["XC0SS"] = (sm.mean() + aT * (sm - sm.mean()) + cm - sm, tab + dyn_c, f0o, per_c)
                conds["XC0SC"] = (smc.mean() + aTc * (smc - smc.mean()) + cm - smc, tab + dyn_c, f0o, per_c)
                ca_ = fa_c0
                vv = fm["f0"] > 0
                slog.append({"pair": f"{M}_{A}_{tk}", "aT": round(aT, 3), "aTc": round(aTc, 3),
                             "corr_slow": round(float(np.corrcoef(lo(ca_)[vv], sm[vv])[0, 1]), 3),
                             "slope_slow": round(float(np.polyfit(sm[vv] - sm[vv].mean(), lo(ca_)[vv] - lo(ca_)[vv].mean(), 1)[0]), 3),
                             "sd_ratio_true": round(float(np.std(lo(ca_)[vv]) / np.std(sm[vv])), 3)})
                (od / f"_slow_log_{si}-{sk}.json").write_text(json.dumps(slog, indent=1))
            if "XC0T" in (a.only or ""):
                def tau_of(f0v: np.ndarray) -> np.ndarray:
                    t_, gap, cur = np.full(len(f0v), -1), 10 ** 6, -1
                    for q in range(len(f0v)):
                        if f0v[q]:
                            if gap >= 40:
                                cur = 0
                            elif cur >= 0:
                                cur += 1
                            gap = 0
                            t_[q] = cur
                        else:
                            gap += 1
                            if cur >= 0:
                                cur += 1
                    return t_

                def tmpl(frs: list) -> np.ndarray:
                    acc, cnt = np.zeros(41), np.zeros(41)
                    vm_ = np.concatenate([fr["c0"][fr["f0"] > 0] for fr in frs]).mean()
                    for fr in frs:
                        tt = tau_of(fr["f0"] > 0)
                        ok = (tt >= 0) & (fr["f0"] > 0)
                        b_ = np.minimum(tt[ok] // 10, 40)
                        np.add.at(acc, b_, fr["c0"][ok] - vm_); np.add.at(cnt, b_, 1)
                    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)

                if "_tmpl" not in O_:
                    O_["_tmpl"] = tmpl(O_["_frs"])
                tT = tmpl([front(load48(FEM / A / f"t{i:02d}_neutral.wav").astype(np.float32)) for i in range(10) if i != k])
                tt = tau_of(fm["f0"] > 0)
                ok = (tt >= 0) & (fm["f0"] > 0)
                c0t = fm["c0"].copy()
                bb = np.minimum(tt[ok] // 10, 40)
                c0t[ok] += tT[bb] - O_["_tmpl"][bb]
                conds["XC0T"] = (c0t, tab + dyn_c, f0o, per_c)
            if "XC0P" in (a.only or ""):
                from sklearn.linear_model import Ridge
                bl, al = ss.butter(2, 3.0 / 100.0)
                lo = lambda z: ss.filtfilt(bl, al, z)

                def pfeat(fr: dict) -> np.ndarray:
                    sp = np.sqrt(np.maximum(fr["p"], 0))
                    cs = np.cumsum(np.pad(sp, ((0, 0), (1, 0))), 1)
                    ma = lambda w: (cs[:, 1:] - cs[:, np.maximum(np.arange(1, sp.shape[1] + 1) - w, 0)]) / np.minimum(np.arange(1, sp.shape[1] + 1), w)
                    return np.concatenate([sp, ma(10), ma(40), ma(100)], 0).T

                def ptarget(fr: dict) -> np.ndarray:
                    return lo(fr["c0"]) - fr["c0"][fr["f0"] > 0].mean()

                tfr = [front(load48(FEM / A / f"t{i:02d}_neutral.wav").astype(np.float32)) for i in range(10) if i != k]
                rs = Ridge(alpha=30.0).fit(np.concatenate([pfeat(f_) for f_ in tfr]), np.concatenate([ptarget(f_) for f_ in tfr]))
                if ("si", k) not in P_si:
                    if "_fr" not in P_si:
                        P_si["_fr"] = {(o, i): front(load48(FEM / o / f"t{i:02d}_neutral.wav").astype(np.float32)) for o in fems[48:78] for i in range(10)}
                    P_si[("si", k)] = Ridge(alpha=30.0).fit(np.concatenate([pfeat(f_) for (o, i), f_ in P_si["_fr"].items() if i != k]),
                                                            np.concatenate([ptarget(f_) for (o, i), f_ in P_si["_fr"].items() if i != k]))
                Xm = pfeat(fm)
                base_ = fm["c0"] - lo(fm["c0"]) + fm["c0"][fm["f0"] > 0].mean()
                conds["XC0P"] = (base_ + rs.predict(Xm), tab + dyn_c, f0o, per_c)
                conds["XC0PI"] = (base_ + P_si[("si", k)].predict(Xm), tab + dyn_c, f0o, per_c)
                dd = rs.predict(Xm) - P_si[("si", k)].predict(Xm)
                conds["XC0PD"] = (fm["c0"] + dd, tab + dyn_c, f0o, per_c)
                conds["XC0PD2"] = (fm["c0"] + 2 * dd, tab + dyn_c, f0o, per_c)
            if "_STY" in (a.only or ""):
                for sty in ("breathy", "low_tension", "soft", "warm"):
                    fs = front(load48(FEM / A / f"{tk}_{sty}.wav").astype(np.float32))
                    js = dtwj(akey(fm, a.align), akey(fs, a.align))
                    conds[f"XCD_STY{sty}"] = (fs["c0"][js], tab + 0.7 * (fs["e3"][:, js] - TA3["T"] @ fs["p"][:, js]), f0o, per_c)
                    conds[f"ASELF_STY{sty}"] = (fs["c0"], fs["e3"], np.where(fs["ft"] > 0, fs["ft"], 0.0), fs["p4"])
            conds["C0TAB"] = (fm["c0"] - O_["T0"] @ fm["p"] + T_["T0"] @ fm["p"], tab + dyn_c, f0o, per_c)
            conds["C0EX"] = (fm["c0"] + c0e(fm["e3"], fa["e3"][:, j]), tab + dyn_c, f0o, per_c)
            if a.only:
                conds = {k_: v_ for k_, v_ in conds.items() if k_ in a.only.split(",")}
            pair = f"{M}_{A}_{tk}"
            for c, (cc, ee, ff, pp) in conds.items():
                sf.write(od / f"{pair}_{c}.wav", render(cc, ee, ff, pp), N.SR)
            print(pair, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
