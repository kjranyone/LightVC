"""D90(学習なし): 動きはどこまで元の話者を運ぶか(動きの生成器 G の検査 I・変換時の写像の選択)。女声 → 女声で測る(性別の差を外して話者の漏れだけを見る)。
評価の 30 目標の実女声: 内容と動き = 話者 A の発話(cen_A)・表 = 話者 B(B = A+1 の順送り)。包絡 = Σp T_B + P(F3_A − Σp T_A^{F3})(T_A^{F3} = A の登録の F3 の表 = 実行時と同じ推定器)。
f0 = A の F1 を A の中央値 → B の μ・c0 = A の F2・周期性 = B の 75 パーセンタイル・出力部 = --rvoc_ckpt。P:
  TAB = 0(表だけ)・RAW = 恒等・COLOR = C_B^½ C_A^−½・WCOLOR = 伸縮 α 1.15 + 色づけ・HALF = 0.5·恒等。
出力 2 組(同じ音声を、目標番号 = B と = A で並べる): <out>/asB/pNNNN_tBBB_<P>.wav・<out>/asA/pNNNN_tAAA_<P>.wav(+ cen)。
話者の中心(cen_TTT)は評価の発話でなく登録の参照の中央 8s(A の内容の発話と重ならない = 内容の一致で A に寄る偏りを避ける)。学習外 LDA(d4b)で asB の top-1 = B に聞こえる率(話者らしさ)・asA の top-1 = A に聞こえる率(元の話者の漏れ)。
    uv run python d90_ff_leak.py --s <scratchpad> --out_dir <scratchpad>/d90 --rvoc_ckpt ../results/rvtab_full/snap/ema_360k.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rvoc_ckpt", required=True)
    a = ap.parse_args()
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
    S = Path(a.s)
    dev = "cuda"
    od = Path(a.out_dir)
    for sub in ("asA", "asB"):
        (od / sub).mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(a.rvoc_ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    Wa = wmat(1.15)

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
        pad = lambda z: np.pad(z, ((0, 0), (0, max(0, n - z.shape[1]))), mode="edge")[:, :n]
        return {"x": x, "n": n, "f0": np.pad(f0f, (0, n - len(f0f))), "c0": np.pad(c0f, (0, n - len(c0f))), "e3": pad(e3), "p": pad(p), "u": pad(p).argmax(0)}

    def enroll(x: np.ndarray) -> dict:
        fr = front(x)
        n = fr["n"]
        f0, _ = H.teacher_f0(fr["x"], n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n][1:25].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n]
        v = f0 > 0
        T = C0.table(env, fr["u"], K, Cb)
        T3 = C0.table(fr["e3"], fr["u"], K, Cb)
        r = env - T[:, fr["u"]]
        rv = r[:, v] - r[:, v].mean(1, keepdims=True)
        r3 = fr["e3"] - T3[:, fr["u"]]
        v3 = fr["f0"] > 0
        r3v = r3[:, v3] - r3[:, v3].mean(1, keepdims=True)
        return {"T": T, "T3": T3, "mu": float(np.median(np.log(f0[v]))), "mu3": float(np.median(np.log(fr["f0"][v3]))), "p75": np.percentile(per[:, v], 75, axis=1),
                "cov": rv @ rv.T / rv.shape[1], "cov3": r3v @ r3v.T / r3v.shape[1]}

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    en = {t: enroll(load48(S / "d37/_work" / f"ref_{t:03d}.wav").astype(np.float32)) for t in range(30)}
    for A in range(30):
        B = (A + 1) % 30
        EA, EB = en[A], en[B]
        x, _ = sf.read(S / "d37" / f"cen_{A:03d}.wav", dtype="float32")
        fr = front(x)
        n = fr["n"]
        f0o = np.where(fr["f0"] > 0, fr["f0"] * np.exp(EB["mu"] - EA["mu3"]), 0.0)
        per_t = np.repeat(EB["p75"][:, None], n, 1)
        d = fr["e3"] - EA["T3"] @ fr["p"]
        tab = EB["T"] @ fr["p"]
        Mc = msqrt(EB["cov"], 0.5) @ msqrt(EA["cov3"], -0.5)
        Mw = msqrt(EB["cov"], 0.5) @ msqrt(Wa @ EA["cov3"] @ Wa.T, -0.5) @ Wa
        conds = {"TAB": tab, "RAW": tab + d, "COLOR": tab + Mc @ d, "WCOLOR": tab + Mw @ d, "HALF": tab + 0.5 * d}
        for c, ev in conds.items():
            y = render(fr["c0"], ev, f0o, per_t)
            sf.write(od / "asB" / f"p{A:04d}_t{B:03d}_{c}.wav", y, N.SR)
            sf.write(od / "asA" / f"p{A:04d}_t{A:03d}_{c}.wav", y, N.SR)
        sf.write(od / "asB" / f"p{A:04d}_t{B:03d}_SRC.wav", fr["x"], N.SR)
        sf.write(od / "asA" / f"p{A:04d}_t{A:03d}_SRC.wav", fr["x"], N.SR)
        print("A", A, "B", B, flush=True)
    for t in range(30):
        r = load48(S / "d37/_work" / f"ref_{t:03d}.wav").astype(np.float32)
        mid = max(0, len(r) // 2 - 4 * N.SR)
        for sub in ("asA", "asB"):
            sf.write(od / sub / f"cen_{t:03d}.wav", r[mid: mid + 8 * N.SR], N.SR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
