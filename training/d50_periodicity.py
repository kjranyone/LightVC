"""D50(診断・学習なし): 周期性の入れ方(耳 warp_ab2『ガビガビ』= 倍音間コントラストの低下の主因の 1 つ: 女声の写し合成で 定数化 3.17 → 2.96)。
周期性 per = pae.periodicity(中心窓 2048)。因果の近似 = 窓を左寄せ(終わりがフレームの時刻)= 中心窓の値を A/HOP ≈ 4 フレーム遅らせたもの(先読み 0)。
(a) 女声の写し合成(目標の実音声 30 本・本物の包絡・c0・教師 f0): ORA = 実の per・CAUS = 因果の per(4 遅れ)・PC = 有声の平均の定数(製品)・P75 / P90 = 有声の 75 / 90 パーセンタイルの定数。
(b) 男声 → 女声の変換(D48 と同じ Python の製品の式・伸縮 γ 0.7・F1): CMEAN = 目標の平均の定数(製品)・CP75 = 目標の 75 パーセンタイルの定数・DYN = 因果の元の per − 元の有声平均 + 目標の平均(帯ごと・[0, 1.5])・DYN75 = DYN の水準を目標の 75 パーセンタイルに。
目標の per の統計 = 目標の登録の参照(D37 の ref_TTT.wav)を教師 f0 で解析。出力 = (a) <out>/copy・(b) <out>/conv(d4b の形式)。
    uv run python d50_periodicity.py --s <scratchpad> --out_dir <scratchpad>/d50
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
    od = Path(a.out_dir)
    (od / "copy").mkdir(parents=True, exist_ok=True)
    (od / "conv").mkdir(parents=True, exist_ok=True)
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

    def analyse(x, f0):
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()
        return env[:, : len(f0)].astype(np.float64), per[:, : len(f0)].astype(np.float64)

    def delay(per):
        return np.concatenate([np.repeat(per[:, :1] * 0, LAG, 1), per], 1)[:, : per.shape[1]]

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def runmed(f0, mu):
        vs, out = [], np.zeros_like(f0)
        for t, x in enumerate(f0):
            if t < 2000 and x > 0:
                bisect.insort(vs, float(np.log(x)))
            n = len(vs)
            med = mu if n == 0 else (vs[n // 2] if n % 2 else 0.5 * (vs[n // 2 - 1] + vs[n // 2]))
            out[t] = np.exp(np.log(x) - med + mu) if x > 0 else 0.0
        return out

    tstat = {}
    for ti in range(30):
        x, _ = sf.read(S / "d37/_work" / f"ref_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        _, per = analyse(x, f0)
        pv = per[:, f0 > 0]
        tstat[ti] = {"mean": pv.mean(1), "p75": np.percentile(pv, 75, axis=1), "p90": np.percentile(pv, 90, axis=1)}
    print("target per stats done", flush=True)
    for ti in range(30):
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        env, per = analyse(x, f0)
        pv = per[:, f0 > 0]
        rep = lambda v: np.repeat(v[:, None], n, 1)
        c0n = (env[0] - PA.C0_SIL) / 10
        k = f"p{ti:04d}_t{ti:03d}"
        for c, pt in {"ORA": per, "CAUS": delay(per), "PC": rep(pv.mean(1)), "P75": rep(np.percentile(pv, 75, axis=1)), "P90": rep(np.percentile(pv, 90, axis=1))}.items():
            sf.write(od / "copy" / f"{k}_{c}.wav", render(c0n, env[1:25] / 10, f0, pt), N.SR)
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / "copy" / f"{k}_SRC.wav")
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / "copy" / f"cen_{ti:03d}.wav")
    print("copy done", flush=True)
    for f in sorted((S / "d37").glob("p*_SRC.wav")):
        k = f.name[: -len("_SRC.wav")]
        pi, ti = int(k[1:5]), int(k[7:10])
        T = np.fromfile(S / "d37/_work" / f"enroll_{ti:03d}" / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)
        jt = json.loads((S / "d37/_work" / f"enroll_{ti:03d}" / "enroll.json").read_text())
        Top = np.fromfile(S / "d37/_work" / f"opS_{pi:04d}" / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)
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
        env = (T @ p) / 10.0 + 0.7 * (W @ (e3[1:25] - (Top @ p) / 10.0))
        f0o = runmed(f0f, jt["mu"])
        _, per_s = analyse(x, np.where(f0f > 0, f0f, 0.0))
        per_c = delay(per_s)
        vs = f0f > 0
        smean = per_c[:, vs].mean(1) if vs.any() else np.zeros(4)
        st = tstat[ti]
        rep = lambda v: np.repeat(v[:, None], n, 1)
        conds = {"CMEAN": rep(np.asarray(jt["per_mean"])), "CP75": rep(st["p75"]),
                 "DYN": per_c - smean[:, None] + st["mean"][:, None], "DYN75": per_c - smean[:, None] + st["p75"][:, None]}
        for c, pt in conds.items():
            sf.write(od / "conv" / f"{k}_{c}.wav", render(c0n, env, f0o, pt), N.SR)
        shutil.copy(f, od / "conv" / f.name)
        print(k, flush=True)
    for c in sorted((S / "d37").glob("cen_*.wav")):
        shutil.copy(c, od / "conv" / c.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
