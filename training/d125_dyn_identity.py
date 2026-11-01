"""D125(学習なし・診断専用): 目標らしく聞こえるには、単位内の動きがどこまで目標に似ている必要があるか。D124(T1)= 別の話者の動きは登録の写像では目標に近づかない(相関 0.22・床 0.58)。
D106 / D123 と同じ組(男声 M → 女声 TTS A・24 目標 × 5 文)で、A の neutral の読みの c0・教師 f0・F4 を固定し、包絡 = Σp_A T_A[u] + 動き の『動き』だけ替えて出力部(rvtab_f3 380k)で鳴らす:
  ASELF = A の F3 の包絡そのもの・TABA = 表だけ(動き 0)・DYNA = A 自身の動き(F3 − Σp T_A・= ASELF とほぼ同じ)・DYNS = A の別の話し方(soft)の同じ文の動き(DTW で並べる)・
  DYNB = 別の女声 B(同じ文)の動き(DTW)・DYNBC = DYNB を大域の色づけ C_A^½ C_B^−½・DYNM = 男声 M の動き(DTW)を製品と同じ伸縮 W_1.3 + 色づけ C_A^½ (W C_M Wᵀ)^−½ で・DYNM0 = 男声の動きを伸縮だけ。
表と共分散 = 各話者の別の 9 文(F3・製品と同じ推定器)。DTW は診断専用。D118 の物差し(女声 TTS の LDA)で測る。読み: DYNB ≈ ASELF → 女声一般の動きで足り律速は男声に固有の動き・DYNB ≪ → 目標に近い動きを作る仕組みが要る。
出力 = <out>/m{i}_{A}_t{k}_{cond}.wav → d118_identity_ceiling.py --dir <out> --tag d125。
    uv run python d125_dyn_identity.py --out_dir <scratchpad>/d125
"""
from __future__ import annotations

import argparse
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
STY = ["neutral", "breathy", "low_tension", "soft", "warm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rvoc_ckpt", default=str(ROOT / "results/rvtab_f3/snap/ema_380k.pt"))
    ap.add_argument("--n_male", type=int, default=24)
    ap.add_argument("--texts", default="0,1,2,3,4")
    a = ap.parse_args()
    import librosa
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    from d6_unit_cov import load48
    from d88_perturb_identity import msqrt, wmat
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    ld = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    sr_ = ld(a.rvoc_ckpt)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    s2 = ld(ROOT / "results/f2_2/last.pt"); f2 = LV.Lvl(s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev); f2.load_state_dict(s2["net"]); f2.eval()
    s3 = ld(ROOT / "results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    s4 = ld(ROOT / "results/f4_1/last.pt"); f4 = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev); f4.load_state_dict(s4["net"]); f4.eval()
    sc = ld(ROOT / "results/c1_1/last.pt"); c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev); c1.load_state_dict(sc["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    W = wmat(1.3)

    @torch.no_grad()
    def front(x: np.ndarray, need_f0: bool = False) -> dict:
        n = len(x) // N.HOP
        x = x[: n * N.HOP].astype(np.float32)
        m0 = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        r = {"n": n, "c0": pad1(C3.causal_avg(f2(m0)[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)), "e3": pad(10.0 * f3(m0)[0].cpu().numpy()[1:25, :n].astype(np.float64)),
             "p4": pad(f4(m0)[0].cpu().numpy()[:, :n].astype(np.float64)), "p": pad(p), "lm": pad(m0[0].cpu().numpy()[:, :n].astype(np.float64))}
        r["u"] = r["p"].argmax(0)
        if need_f0:
            ft, _ = H.teacher_f0(x, n)
            r["ft"] = pad1(ft[:n].astype(np.float64))
        return r

    def enroll(frs: list) -> dict:
        E = np.concatenate([f["e3"] for f in frs], 1); U = np.concatenate([f["u"] for f in frs]); P = np.concatenate([f["p"] for f in frs], 1)
        T = C0.table(E, U, K, Cb)
        R_ = E - T @ P
        R_ = R_ - R_.mean(1, keepdims=True)
        return {"T": T, "C": R_ @ R_.T / R_.shape[1]}

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def warp_to(src: dict, dst: dict) -> np.ndarray:
        an = dst["lm"] / (np.linalg.norm(dst["lm"], axis=0, keepdims=True) + 1e-9)
        bn = src["lm"] / (np.linalg.norm(src["lm"], axis=0, keepdims=True) + 1e-9)
        _, wp = librosa.sequence.dtw(C=1 - an.T @ bn)
        j = np.zeros(dst["n"], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        return j

    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)[: a.n_male]
    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STY))
    random.Random(80).shuffle(fems)
    others = fems[a.n_male:]
    for mi, M in enumerate(males):
        A = fems[mi]
        B = others[mi % len(others)]
        mf, tot = [], 0.0
        for f in sorted(MALE.glob(f"{M}/t*.wav")):
            if tot >= 22.0:
                break
            x = load48(f).astype(np.float32); mf.append(front(x)); tot += len(x) / 48000
        EM = enroll(mf)
        for k in [int(t) for t in a.texts.split(",")]:
            pre = f"{M}_{A}_t{k:02d}"
            if (od / f"{pre}_DYNM0.wav").exists():
                continue
            EA = enroll([front(load48(FEM / A / f"t{i:02d}_neutral.wav").astype(np.float32)) for i in range(10) if i != k])
            EB = enroll([front(load48(FEM / B / f"t{i:02d}_neutral.wav").astype(np.float32)) for i in range(10) if i != k])
            fa = front(load48(FEM / A / f"t{k:02d}_neutral.wav").astype(np.float32), need_f0=True)
            fs_ = front(load48(FEM / A / f"t{k:02d}_soft.wav").astype(np.float32))
            fb = front(load48(FEM / B / f"t{k:02d}_neutral.wav").astype(np.float32))
            fm = front(load48(MPAR / M / f"f{k:02d}.wav").astype(np.float32))
            f0a = np.where(fa["ft"] > 0, fa["ft"], 0.0)
            tab = EA["T"] @ fa["p"]
            dA = fa["e3"] - tab
            dS = (fs_["e3"] - EA["T"] @ fs_["p"])[:, warp_to(fs_, fa)]
            dB = (fb["e3"] - EB["T"] @ fb["p"])[:, warp_to(fb, fa)]
            dM = (fm["e3"] - EM["T"] @ fm["p"])[:, warp_to(fm, fa)]
            Mc = msqrt(EA["C"], 0.5) @ msqrt(W @ EM["C"] @ W.T, -0.5)
            conds = {"ASELF": fa["e3"], "TABA": tab, "DYNA": tab + dA, "DYNS": tab + dS, "DYNB": tab + dB,
                     "DYNBC": tab + msqrt(EA["C"], 0.5) @ msqrt(EB["C"], -0.5) @ dB, "DYNM": tab + Mc @ W @ dM, "DYNM0": tab + W @ dM}
            for c, ev in conds.items():
                sf.write(od / f"{pre}_{c}.wav", render(fa["c0"], ev, f0a, fa["p4"]), N.SR)
            print(pre, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
