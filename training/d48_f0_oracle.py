"""D48(診断・学習なし): 耳 warp_ab2(『f0 を見失っている・ガビガビ』)の帰属。伸縮つき話者差 γ 0.7(α 1.3・κ 30)を Python で製品と同じ式で作り、f0 だけを替える。
  PYF1 = F1(製品の f0・Rust の G7 の検算)・PYT = 教師 f0(f0hi・有声と高さが正しい)・PYTV = 高さは F1・有声の判定だけ教師(F1 の値が無いフレームは教師の値)。
他は全て製品と同じ: 包絡 = Σp T + γ W(F3 − Σp T_op)(柔らかい引き β 1・平滑なし)・c0 = F2(3 平均)・f0 の移動 = 走る中央値(最初の 2000 フレーム)→ 目標の中央値・周期性 = 目標の平均 × 有声。
元 = D37 の 60 文・表 = D37 の Rust 登録(目標 enroll_TTT・操作者 opS_PPPP)。出力 = d4b の形式。
    uv run python d48_f0_oracle.py --s <scratchpad> --out_dir <scratchpad>/d48
"""
from __future__ import annotations

import argparse
import bisect
import json
import shutil
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
    ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=1.3)
    a = ap.parse_args()
    import c1_content as CC
    import f0est as FE
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    import zsvc as Z
    S = Path(a.s)
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
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
    Dm = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)
    fb = N.mel_fb().numpy()
    fc = fb.argmax(1) * (N.SR / 2) / (fb.shape[1] - 1)
    B = Dm[1:25]
    I = np.zeros((N.N_MEL, N.N_MEL))
    src = np.clip(fc / a.alpha, fc[0], fc[-1])
    jj = np.clip(np.searchsorted(fc, src) - 1, 0, N.N_MEL - 2)
    ww = np.clip((src - fc[jj]) / np.maximum(fc[jj + 1] - fc[jj], 1e-9), 0, 1)
    I[np.arange(N.N_MEL), jj] = 1 - ww
    I[np.arange(N.N_MEL), jj + 1] += ww
    W = B @ I @ B.T
    rd = lambda d: (np.fromfile(d / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64), json.loads((d / "enroll.json").read_text()))

    def runmed(f0, mu):
        vs, out = [], np.zeros_like(f0)
        for t, x in enumerate(f0):
            if t < 2000 and x > 0:
                bisect.insort(vs, float(np.log(x)))
            n = len(vs)
            med = mu if n == 0 else (vs[n // 2] if n % 2 else 0.5 * (vs[n // 2 - 1] + vs[n // 2]))
            out[t] = np.exp(np.log(x) - med + mu) if x > 0 else 0.0
        return out

    @torch.no_grad()
    def render(c0n, env24n, f0, per):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per, 0, 1.5)[:, None] * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for f in sorted((S / "d37").glob("p*_SRC.wav")):
        k = f.name[: -len("_SRC.wav")]
        pi, ti = int(k[1:5]), int(k[7:10])
        T, jt = rd(S / "d37/_work" / f"enroll_{ti:03d}")
        Top, _ = rd(S / "d37/_work" / f"opS_{pi:04d}")
        x, _ = sf.read(f, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        xt = torch.from_numpy(x)[None].to(dev)
        with torch.no_grad():
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
            f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
            c0n = C3.causal_avg(f2(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
            e3 = f3(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[0].cpu().numpy()[:, :n].astype(np.float64)
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        f0t, _ = H.teacher_f0(x, n)
        f0t = f0t[:n].astype(np.float64)
        env = (T @ p) / 10.0 + a.gamma * (W @ (e3[1:25] - (Top @ p) / 10.0))
        per = np.asarray(jt["per_mean"], np.float64)
        f0tv = np.where(f0t > 0, np.where(f0f > 0, f0f, f0t), 0.0)
        for c, f0src in (("PYF1", f0f), ("PYT", f0t), ("PYTV", f0tv)):
            f0o = runmed(f0src, jt["mu"])
            sf.write(od / f"{k}_{c}.wav", render(c0n, env, f0o, per), N.SR)
        for c in ("SRC", "TAB"):
            shutil.copy(S / "d37" / f"{k}_{c}.wav", od / f"{k}_{c}.wav")
        shutil.copy(S / "d40" / f"{k}_G7.wav", od / f"{k}_G7.wav")
        print(k, flush=True)
    for c in sorted((S / "d37").glob("cen_*.wav")):
        shutil.copy(c, od / c.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
