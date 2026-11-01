"""D79(学習なし): 表の包絡を f0 と整合させる(フレームごとの定常な調波で合成 → CheapTrick で取り直す)と、写し合成の参照 PESQ(D77 で耳の合否を再現 AUC 0.996)が上がるか。
各フレーム t: 包絡 c_t(DCT24 → 対数 mel → 倍音の周波数で読む)と f0_t から定常な調波(長さ L・中心で CheapTrick)→ 同じ mel → DCT。無声フレームは元のまま。表と f0 だけを使う(本物の動きは使わない・先読みなし)。
評価の 30 目標の女声(D37 の cen)・f0 = 教師・c0 / 周期性 = 本物(D65 と同じ)・出力部 = rvoc3hi 330k:
  TAB = Σp T・TABRE = TAB を f0 で取り直し・DYN = 本物の包絡(合格の基準)・REAL_RE = 本物の包絡を取り直したもの(取り直し自体の損失の確認)。
参照 PESQ(16k wb・SRC と RMS を揃える)を印字。出力 = <out>/pNNNN_tNNN_<cond>.wav + SRC・results/conv_p0/d79.json。
    uv run python d79_reanalysis.py --s <scratchpad> --out_dir <scratchpad>/d79
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
L = 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max", type=int, default=30)
    a = ap.parse_args()
    import librosa
    import pyworld
    from pesq import pesq
    import c1_content as CC
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_rvoc as TR
    from quality_gate import align
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
    fb, dct = PA._np_consts()
    fbn = fb / np.maximum(fb.sum(1, keepdims=True), 1e-9)
    fc = (fbn * np.arange(fb.shape[1])[None]).sum(1) * (N.SR / 2) / (fb.shape[1] - 1)
    Dinv = np.linalg.pinv(dct[:25])
    tt = np.arange(L) / N.SR

    def reanalyse(c: np.ndarray, f0: np.ndarray) -> np.ndarray:
        out = c.copy()
        logmel = Dinv @ c
        for t in np.nonzero(f0 > 0)[0]:
            f = f0[t]
            k = np.arange(1, int((N.SR / 2 - 200) / f) + 1)
            amp = np.exp(np.interp(k * f, fc, logmel[:, t]))
            y = (amp[:, None] * np.sin(2 * np.pi * f * k[:, None] * tt[None])).sum(0)
            sp = pyworld.cheaptrick(y, np.array([f]), np.array([L / 2 / N.SR]), N.SR, fft_size=PA.NFFT)
            mel = np.log(np.maximum(np.sqrt(sp) @ fb.T, 1e-5))[0]
            out[:, t] = dct[:25] @ mel
        return out

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def score(x: np.ndarray, y: np.ndarray) -> float:
        _, y = align(x, y)
        y = np.pad(y[: len(x)], (0, max(0, len(x) - len(y))))
        y = y * np.sqrt((x ** 2).mean() / max((y ** 2).mean(), 1e-12))
        return float(pesq(16000, librosa.resample(x, orig_sr=48000, target_sr=16000), librosa.resample(y.astype(np.float32), orig_sr=48000, target_sr=16000), "wb"))

    res: dict = {}
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
        tabc = env.copy()
        tabc[1:25] = T @ p
        conds = {"TAB": tabc, "TABRE": reanalyse(tabc, f0), "DYN": env, "REAL_RE": reanalyse(env, f0)}
        key = f"p{ti:04d}_t{ti:03d}"
        row = {}
        for c, ev in conds.items():
            y = render((env[0] - PA.C0_SIL) / 10, ev[1:25] / 10, f0, per)
            sf.write(od / f"{key}_{c}.wav", y, N.SR)
            row[c] = round(score(x, y), 4)
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / f"{key}_SRC.wav")
        res[key] = row
        print(key, row, flush=True)
    summ = {c: round(float(np.mean([r[c] for r in res.values()])), 4) for c in ["TAB", "TABRE", "DYN", "REAL_RE"]}
    summ["TABRE_gt_TAB"] = round(float(np.mean([r["TABRE"] > r["TAB"] for r in res.values()])), 3)
    print(json.dumps(summ), flush=True)
    (ROOT / "results/conv_p0/d79.json").write_text(json.dumps({"summary": summ, "per": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
