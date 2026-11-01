"""D71(診断・学習なし・D68 の派生): 本人の動き res の位相だけを乱した代用(周波数ごとのパワーと 24 次元の相互スペクトルを保ち、時間の並びだけ壊す・全次元に同じ乱数位相)で合格するか。
  TAB = 表・DYN = 表 + res・SURR = 表 + 位相を乱した res(種 = 目標番号)。合格 → 目標の統計だけから確率的に動きを生成できる。
以下は D68 の説明。D68(診断・学習なし): 耳 tabenv_ab1(『表 + 本物の単位の中の動き』だけ合格)と rvtab_ab1(出力部を表の包絡で追加学習しても失格)の帰属。
合格した動き res = 本物の包絡 − 同じ発話の単位平均の事後混合(INS·p)を、時間方向の変調周波数で遅い成分(LP)と速い成分(HP)に分ける(ゼロ位相・診断専用)。
目標の実音声 30 本(D37 の cen)・f0 = 教師・c0 / 周期性 = 本物・出力部 = rvoc3hi 330k・包絡だけ替える:
  TAB = 表(Σp T)・SLOW = 表 + LP(res)・FAST = 表 + HP(res)・DYN = 表 + res(tabenv_ab1 で合格)。
res の変調周波数ごとのエネルギーの割合も印字する。出力 = d4b の形式(cen・pNNNN_tNNN_<cond>・SRC)。
    uv run python d68_dyn_split.py --s <scratchpad> --out_dir <scratchpad>/d68 --fc 8
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
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fc", type=float, default=8.0)
    ap.add_argument("--max", type=int, default=30)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    fr = N.SR / N.HOP
    sos = butter(4, a.fc, fs=fr, output="sos")
    edges = [0, 2, 4, 8, 16, 32, fr / 2]
    band_e = np.zeros(len(edges) - 1)

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for ti in range(a.max):
        T = np.fromfile(S / "d53_en" / f"en_{ti:03d}" / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        with torch.no_grad():
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        er = env[1:25]
        INS = C0.table(er, p.argmax(0), K, Cb)
        res = er - INS @ p
        lo = sosfiltfilt(sos, res, axis=1)
        hi = res - lo
        P = np.abs(np.fft.rfft(res - res.mean(1, keepdims=True), axis=1)) ** 2
        fq = np.fft.rfftfreq(n, 1 / fr)
        for b in range(len(edges) - 1):
            band_e[b] += P[:, (fq >= edges[b]) & (fq < edges[b + 1])].sum()
        c0n = (env[0] - PA.C0_SIL) / 10
        tab = T @ p
        Fr = np.fft.rfft(res - res.mean(1, keepdims=True), axis=1)
        ph = np.exp(2j * np.pi * np.random.default_rng(ti).random(Fr.shape[1]))
        ph[0] = 1
        if n % 2 == 0:
            ph[-1] = 1
        surr = np.fft.irfft(Fr * ph[None], n=n, axis=1) + res.mean(1, keepdims=True)
        conds = {"TAB": tab, "DYN": tab + res, "SURR": tab + surr}
        k = f"p{ti:04d}_t{ti:03d}"
        for c, ev in conds.items():
            sf.write(od / f"{k}_{c}.wav", render(c0n, ev / 10, f0, per), N.SR)
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / f"{k}_SRC.wav")
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / f"cen_{ti:03d}.wav")
        print("target", ti, flush=True)
    frac = band_e / band_e.sum()
    rep = {"fc_hz": a.fc, "res_mod_energy_frac": {f"{edges[b]:.0f}-{edges[b + 1]:.0f}Hz": round(float(frac[b]), 4) for b in range(len(frac))}}
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    (od / "d68.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
