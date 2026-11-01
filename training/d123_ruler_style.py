"""D123 = 査読(10-11)の T3(学習なし・診断専用): 同一性の物差し(D118 の LDA)は声を読んでいるか、演技(話し方)を読んでいるか。
D118 の最大の損失は c0(男声の音量の配り方 +0.22)。構音 VC の標準の前提では音量・抑揚は操作者(演技)側の量なので、物差しが演技の一致を読んでいれば変換器の失格に帰属させてはいけない。
D106 と同じ組(男声 M → 女声 TTS A・24 目標 × 5 文)で、A の neutral の読みの特徴(F3 の包絡・教師 f0・F4)を固定し c0 だけを替えて出力部(rvtab_f3 380k)で鳴らす:
  ASELF = A の neutral の c0(基準)・C0S_<sty> = 同じ文の A の別の話し方(breathy / low_tension / soft / warm)の c0 を DTW で neutral の時間へ・C0M = 男声 M の同じ文の c0 を DTW で。
  RS_<sty> = A の別の話し方の実音声(加工なし)。DTW は診断専用。
判定(査読の線): RS と C0S がともに ≥ 0.9 で C0M だけ < 0.8 → c0 は目標固有の量(設計対象)・C0S < 0.85 → 物差しが演技を読む(関門を話し方に不変に較正し直すまで c0 の差を変換器に帰属させない)。
出力 = <out>/m{i}_{A}_t{k}_{cond}.wav → d118_identity_ceiling.py --dir <out> --tag d123 で測る。
    uv run python d123_ruler_style.py --out_dir <scratchpad>/d123
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"
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
    import f0est as FE
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import rvoc as R
    import train_c3 as C3
    import train_rvoc as TR
    from d6_unit_cov import load48
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    ld = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    sr_ = ld(a.rvoc_ckpt)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    s2 = ld(ROOT / "results/f2_2/last.pt"); f2 = LV.Lvl(s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev); f2.load_state_dict(s2["net"]); f2.eval()
    s3 = ld(ROOT / "results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    s4 = ld(ROOT / "results/f4_1/last.pt"); f4 = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev); f4.load_state_dict(s4["net"]); f4.eval()
    mfront = CC.MelFront().to(dev)

    @torch.no_grad()
    def front(x: np.ndarray) -> dict:
        n = len(x) // N.HOP
        x = x[: n * N.HOP].astype(np.float32)
        m0 = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
        c0 = C3.causal_avg(f2(m0)[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
        e3 = 10.0 * f3(m0)[0].cpu().numpy()[1:25, :n].astype(np.float64)
        p4 = f4(m0)[0].cpu().numpy()[:, :n].astype(np.float64)
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        ft, _ = H.teacher_f0(x, n)
        return {"n": n, "c0": pad1(c0), "e3": pad(e3), "p4": pad(p4), "lm": pad(m0[0].cpu().numpy()[:, :n].astype(np.float64)), "ft": pad1(ft[:n].astype(np.float64))}

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def warp_to(src: dict, dst: dict) -> np.ndarray:
        _, wp = librosa.sequence.dtw(X=dst["lm"], Y=src["lm"], metric="euclidean")
        j = np.zeros(dst["n"], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        return j

    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)[: a.n_male]
    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STY))
    random.Random(80).shuffle(fems)
    for mi, M in enumerate(males):
        A = fems[mi]
        for k in [int(t) for t in a.texts.split(",")]:
            pre = f"{M}_{A}_t{k:02d}"
            if (od / f"{pre}_C0M.wav").exists():
                continue
            fa = front(load48(FEM / A / f"t{k:02d}_neutral.wav").astype(np.float32))
            f0a = np.where(fa["ft"] > 0, fa["ft"], 0.0)
            sf.write(od / f"{pre}_ASELF.wav", render(fa["c0"], fa["e3"], f0a, fa["p4"]), N.SR)
            for sty in STY[1:]:
                fs_ = front(load48(FEM / A / f"t{k:02d}_{sty}.wav").astype(np.float32))
                sf.write(od / f"{pre}_C0S{sty}.wav", render(fs_["c0"][warp_to(fs_, fa)], fa["e3"], f0a, fa["p4"]), N.SR)
                shutil.copy(FEM / A / f"t{k:02d}_{sty}.wav", od / f"{pre}_RS{sty}.wav")
            fm = front(load48(MPAR / M / f"f{k:02d}.wav").astype(np.float32))
            sf.write(od / f"{pre}_C0M.wav", render(fm["c0"][warp_to(fm, fa)], fa["e3"], f0a, fa["p4"]), N.SR)
            print(pre, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
