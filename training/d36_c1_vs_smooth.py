"""D36(診断・学習なし): 男声 → 女声の柔らかい引きの明瞭さの損失を、(a) 表の過平滑 と (b) 男声の入力での C1 の認識の誤り に分ける。
同じ文の組(data/same_text_pairs・男声 TTS 3 × 女声 TTS --n_f 人 × 評価 20 文・D34 と同じ評価文・表は評価外の文の 20s)。出力部 = rvoc3hi(Python)。
  F_COPY = 女声の実の包絡(写し合成・女声の時間・c0・f0・周期性)・F_TAB = Σ p_f T_f(女声自身の事後 = 認識は正しく過平滑だけ)・F_HARD = T_f[u_f]・
  M_TAB = Σ p_m T_f(男声の事後・男声の時間・c0・f0 は女声の中央値へ移動・周期性は女声の平均 = 製品と同じ形)・M_M35 = 0.65·M_TAB + 0.35·男声の包絡。
CER の基準 = 男声の元の発話(SRC)の文字起こし(同じ文)。d23_cer.py で測る。D34 の解析のキャッシュ(包絡・事後)を使う。
    uv run python d36_c1_vs_smooth.py --cache <scratchpad>/d34_cache.pkl --out_dir <scratchpad>/d36
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_f", type=int, default=4)
    a = ap.parse_args()
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_rvoc as TR
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    A = pickle.loads(Path(a.cache).read_bytes())
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)

    @torch.no_grad()
    def render(c0, env24, f0, per):
        cc = np.concatenate([c0[None] - PA.C0_SIL, env24], 0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    root = ROOT / "data/same_text_pairs"
    males, fems = ["m226", "m227", "m232"], [f"f{c}" for c in "ABCDEFGH"][: a.n_f]
    names = sorted(f.name for f in (root / "fA").glob("*.wav"))
    cats = sorted({n.rsplit("_", 1)[0] for n in names})
    ev = [n for c in cats for n in sorted(x for x in names if x.startswith(c + "_"))[12:16]]
    en = [n for n in names if n not in ev]

    def table20(s):
        order = [en[i] for i in np.random.RandomState(0).permutation(len(en))]
        sel, tot = [], 0
        for n in order:
            if tot >= 20 * 200:
                break
            sel.append(n); tot += A[s][n]["env"].shape[1]
        return C0.table(np.concatenate([A[s][n]["env"][1:25] for n in sel], 1), np.concatenate([A[s][n]["u"] for n in sel]), K, Cb)

    def prosody(s, n):
        x = load48(root / s / n)
        T = A[s][n]["env"].shape[1]
        x = x[: T * N.HOP]
        f0, _ = H.teacher_f0(x, T)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()
        return x, f0[:T], per[:, :T]

    pi = 0
    for fi, f in enumerate(fems):
        Tf = table20(f)
        fp = {n: prosody(f, n) for n in ev}
        vo_all = np.concatenate([fp[n][1][fp[n][1] > 0] for n in ev])
        mu_f = float(np.median(np.log(vo_all)))
        per_f = np.concatenate([fp[n][2][:, fp[n][1] > 0] for n in ev], 1).mean(1)
        for m in males:
            for n in ev:
                af, am = A[f][n], A[m][n]
                xm, f0m, _ = prosody(m, n)
                _, f0f, perf = fp[n]
                envs_f = {"F_COPY": af["env"][1:25], "F_TAB": Tf @ af["p"], "F_HARD": Tf[:, af["u"]]}
                for c, ev_ in envs_f.items():
                    sf.write(od / f"p{pi:04d}_t{fi:03d}_{c}.wav", render(af["env"][0], ev_, f0f, perf), 48000)
                vo = f0m > 0
                lfs = np.log(np.maximum(f0m, 1.0))
                med = float(np.median(lfs[vo])) if vo.any() else mu_f
                f0o = np.where(vo, np.exp(lfs - med + mu_f), 0.0)
                perc = np.repeat(per_f[:, None], len(f0m), 1)
                mt = Tf @ am["p"]
                for c, ev_ in {"M_TAB": mt, "M_M35": 0.65 * mt + 0.35 * am["env"][1:25]}.items():
                    sf.write(od / f"p{pi:04d}_t{fi:03d}_{c}.wav", render(am["env"][0], ev_, f0o, perc), 48000)
                sf.write(od / f"p{pi:04d}_t{fi:03d}_SRC.wav", xm, 48000)
                pi += 1
        print("female", f, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
