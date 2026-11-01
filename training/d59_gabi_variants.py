"""D59(診断・学習なし): 耳『女声の高さで鳴らすとガビる』(ladder_ab2)の候補を、新しい男声 → 女声の変換(伸縮 γ 0.7・α 1.3・κ なし = D56 の L4FULL と同じ式)で 1 つずつ本物の分析に替える。
  BASE = 製品と同じ(F1 の f0・F2 の c0・F3 の包絡・周期性 = 目標の 75 パーセンタイルの定数・高さの基準 = 操作者の中央値)
  TF0 = f0 だけ元の男声の教師 f0(同じ高さの移動)・RC0 = c0 だけ元の男声の CheapTrick の c0・PDYN = 周期性だけ元の男声の時変の周期性(因果の窓 = 4 フレーム遅れ)を目標の 75 パーセンタイルの水準へ・ALL = 3 つとも。
元 = D56 の 60 文・表 = D56 の操作者の登録(op_PPPP)と D53 の目標の登録。出力 = d4b の形式。
    uv run python d59_gabi_variants.py --s <scratchpad> --out_dir <scratchpad>/d59
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--max", type=int, default=60)
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
    sm_w = TR.ckpt_env_smooth(sr_, None)
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
    src_ = np.clip(fc / 1.3, fc[0], fc[-1])
    jj = np.clip(np.searchsorted(fc, src_) - 1, 0, N.N_MEL - 2)
    ww = np.clip((src_ - fc[jj]) / np.maximum(fc[jj + 1] - fc[jj], 1e-9), 0, 1)
    I[np.arange(N.N_MEL), jj] = 1 - ww
    I[np.arange(N.N_MEL), jj + 1] += ww
    W = B @ I @ B.T
    LAG = int(round(PA.A / N.HOP))
    rd = lambda d: (np.fromfile(d / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64), json.loads((d / "enroll.json").read_text()))

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for f in sorted((S / "d56").glob("p*_SRC.wav"))[: a.max]:
        k = f.name[: -len("_SRC.wav")]
        pi, ti = int(k[1:5]), int(k[7:10])
        T, jt = rd(S / "d53_en" / f"en_{ti:03d}")
        Top, jo = rd(S / "d56/_work" / f"op_{pi:04d}")
        x, _ = sf.read(f, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        xt = torch.from_numpy(x)[None].to(dev)
        with torch.no_grad():
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
            f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
            c0f = C3.causal_avg(f2(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
            e3 = f3(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[0].cpu().numpy()[:, :n].astype(np.float64)
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        f0t, _ = H.teacher_f0(x, n)
        f0t = f0t[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0t), ((0, 0), (1, 1)), mode="edge")
        ct = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n]
        c0r = (ct[0] - PA.C0_SIL) / 10
        per_s = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0t.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        per_c = np.concatenate([np.zeros((4, LAG)), per_s], 1)[:, :n]
        vs = f0t > 0
        p75 = np.asarray(jt["per_p75"], np.float64)
        pdyn = per_c - (per_c[:, vs].mean(1, keepdims=True) if vs.any() else 0) + p75[:, None]
        pconst = np.repeat(p75[:, None], n, 1)
        env = (T @ p) / 10.0 + 0.7 * (W @ (e3[1:25] - (Top @ p) / 10.0))
        sh = np.exp(jt["mu"] - jo["mu"])
        f1o, tfo = f0f * sh, f0t * sh
        conds = {"BASE": (c0f, f1o, pconst), "TF0": (c0f, tfo, pconst), "RC0": (c0r, f1o, pconst), "PDYN": (c0f, f1o, pdyn), "ALL": (c0r, tfo, pdyn)}
        for c, (c0n, f0o, pt) in conds.items():
            sf.write(od / f"{k}_{c}.wav", render(c0n, env, f0o, pt), N.SR)
        shutil.copy(f, od / f.name)
        shutil.copy(S / "d56" / f"{k}_L4FULL.wav", od / f"{k}_L4FULL.wav")
        print(k, flush=True)
    for c in sorted((S / "d56").glob("cen_*.wav")):
        shutil.copy(c, od / c.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
