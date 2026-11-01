"""D100(学習なし): f0 の先読みでオクターブ誤りがどれだけ減るか。F1 の音高の事後(328 ビン・20 cent)に、遷移 = 2 ビン幅のガウス(σ 10 ビン)+ 一様の跳び ε 0.02
(Rust の F0Est::set_track と同じ)の隠れマルコフで、L フレーム先まで見る固定遅れの平滑をかける(L = 0 は今の前向きフィルタ)。有声の判定は F1 のまま。
教師 f0(harvest)に対する log f0 の相関・オクターブ誤り(|Δ log2| > 0.5)を、実の男声(VCTK)・男声 TTS・実の女声で比べる。L フレーム = L × 5ms の遅れ(内容符号器の予備 10ms の内なら L ≤ 2)。
出力 = results/conv_p0/d100.json。
    uv run python d100_f0_lookahead.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
VM = ["p226", "p227", "p232", "p237", "p241", "p243", "p245", "p246", "p247", "p251", "p252", "p254"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--eps", type=float, default=0.02)
    a = ap.parse_args()
    import f0est as FE
    import f0hi as H
    import nvoc as N
    from d6_unit_cov import load48
    dev = "cuda"
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    fr_ = FE.Front().to(dev)
    nb = FE.N_BIN
    idx = np.arange(nb)
    G = np.exp(-0.5 * ((idx[:, None] - idx[None, :]) / a.sigma) ** 2)
    G /= G.sum(1, keepdims=True)
    A = (1 - a.eps) * G + a.eps / nb
    lhz = np.log(FE.bin_hz().numpy())

    def dec(pr: np.ndarray) -> np.ndarray:
        k = pr.argmax(0)
        out = np.zeros(pr.shape[1])
        for t in range(pr.shape[1]):
            ii = np.clip(np.arange(k[t] - 4, k[t] + 5), 0, nb - 1)
            w = pr[ii, t]
            out[t] = np.exp((w * lhz[ii]).sum() / max(w.sum(), 1e-12))
        return out

    def smooth(E: np.ndarray, L: int) -> np.ndarray:
        T = E.shape[1]
        al = np.zeros_like(E)
        a_ = E[:, 0] / E[:, 0].sum()
        al[:, 0] = a_
        for t in range(1, T):
            a_ = (A.T @ a_) * E[:, t]
            a_ /= max(a_.sum(), 1e-300)
            al[:, t] = a_
        if L == 0:
            return al
        post = np.zeros_like(E)
        for t in range(T):
            b = np.ones(nb)
            for s in range(min(T - 1, t + L), t, -1):
                b = A @ (E[:, s] * b)
                b /= max(b.sum(), 1e-300)
            q = al[:, t] * b
            post[:, t] = q / max(q.sum(), 1e-300)
        return post

    sets = {
        "male_real_vctk": [f for s in VM for f in sorted(glob.glob(str(ROOT / f"data/vctk/VCTK-Corpus/VCTK-Corpus/wav48/{s}/*.wav")))[10:12]],
        "male_tts_v4l": sorted(glob.glob(str(ROOT / "data/male_tts_v4l_par/m0*/f00.wav")))[:24],
        "female_real": sorted(glob.glob(str(Path(a.s) / "d37/cen_*.wav"))),
    }
    variants = ["raw", "L0", "L1", "L2", "L4"]
    rep = {}
    for nm, files in sets.items():
        acc = {v: [] for v in variants}
        for p in files:
            x = load48(p).astype(np.float32)[: 8 * 48000]
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            ft, _ = H.teacher_f0(x, n)
            ft = ft[:n].astype(np.float64)
            with torch.no_grad():
                v_, p_ = f1(fr_(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x)[None].to(dev)], 1), n))
                vo = (torch.sigmoid(v_[0]) > 0.5).cpu().numpy()[:n]
                pr = p_[0].softmax(0).cpu().numpy().astype(np.float64)[:, :n]
            m = min(len(vo), pr.shape[1], n)
            vo, pr, ftm = vo[:m], pr[:, :m], ft[:m]
            fs = {"raw": dec(pr)}
            for L in (0, 1, 2, 4):
                fs[f"L{L}"] = dec(smooth(pr + 1e-12, L))
            for v, f in fs.items():
                f = np.where(vo, f, 0.0)
                b = (ftm > 0) & (f > 0)
                if b.sum() < 20:
                    continue
                acc[v].append((np.corrcoef(np.log2(ftm[b]), np.log2(f[b]))[0, 1], np.mean(np.abs(np.log2(f[b] / ftm[b])) > 0.5)))
        rep[nm] = {v: {"corr": round(float(np.mean([z[0] for z in r])), 4), "oct_err": round(float(np.mean([z[1] for z in r])), 4), "n": len(r)} for v, r in acc.items()}
        print(nm, json.dumps(rep[nm]), flush=True)
    (ROOT / "results/conv_p0/d100.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
