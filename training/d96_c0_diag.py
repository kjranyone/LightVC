"""D96(学習なし): 音量の推定器 F2(c0)の何がずれているか。D95 で F2 の c0 に替えただけの写し合成が本物と機械で区別される(AUC 0.993・本物の分析 0.861)。
評価の 30 目標(D37 の cen)で、本物の c0(CheapTrick・[.25,.5,.25] 平滑)と F2 の c0(生 / 因果 3 平均)を比べる:
相関・RMSE・平均の偏り・時間のずれ(相互相関の最大の遅れ・フレーム)・変調スペクトル(c0 の時間軌道の 0〜2 / 2〜8 / 8〜32 / 32〜100Hz のエネルギー比 = なまり)。
出力 = results/conv_p0/d96.json。
    uv run python d96_c0_diag.py --s <scratchpad>
"""
from __future__ import annotations

import argparse
import json
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
    a = ap.parse_args()
    import c1_content as CC
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import train_c3 as C3
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    mfront = CC.MelFront().to(dev)
    fr = N.SR / N.HOP
    edges = [0, 2, 8, 32, fr / 2]
    acc: dict = {"raw": [], "avg3": []}
    for t in range(30):
        x, _ = sf.read(S / "d37" / f"cen_{t:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0[:n].astype(np.float64)), ((0, 0), (1, 1)), mode="edge")
        c0 = ((sm_w * e[0, :-2] + (1 - 2 * sm_w) * e[0, 1:-1] + sm_w * e[0, 2:])[:n] - PA.C0_SIL) / 10
        with torch.no_grad():
            m = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
            r = f2(m)[:, None]
            v = {"raw": r[0, 0].cpu().numpy()[:n], "avg3": C3.causal_avg(r, 3)[0, 0].cpu().numpy()[:n]}
        for k, y in v.items():
            y = np.pad(y, (0, n - len(y)), mode="edge").astype(np.float64)
            cc = [np.corrcoef(c0[max(0, L):n + min(0, L)], y[max(0, -L):n - max(0, L)])[0, 1] for L in range(-6, 7)]
            lag = int(np.argmax(cc)) - 6

            def band(z):
                P = np.abs(np.fft.rfft(z - z.mean())) ** 2
                fq = np.fft.rfftfreq(len(z), 1 / fr)
                return np.array([P[(fq >= edges[b]) & (fq < edges[b + 1])].sum() for b in range(len(edges) - 1)])
            bt, by = band(c0), band(y)
            acc[k].append({"corr": float(np.corrcoef(c0, y)[0, 1]), "rmse": float(np.sqrt(((c0 - y) ** 2).mean())), "bias": float((y - c0).mean()),
                           "best_lag": lag, "best_corr": float(max(cc)), "mod_ratio_db": (10 * np.log10(np.maximum(by, 1e-12) / np.maximum(bt, 1e-12))).tolist(),
                           "sd_ratio": float(y.std() / c0.std())})
    rep = {}
    for k, rs in acc.items():
        rep[k] = {"corr": round(float(np.mean([r["corr"] for r in rs])), 4), "rmse": round(float(np.mean([r["rmse"] for r in rs])), 4), "bias": round(float(np.mean([r["bias"] for r in rs])), 4),
                  "best_lag_median": float(np.median([r["best_lag"] for r in rs])), "best_corr": round(float(np.mean([r["best_corr"] for r in rs])), 4),
                  "sd_ratio": round(float(np.mean([r["sd_ratio"] for r in rs])), 3),
                  "mod_ratio_db_0_2_8_32_100": [round(float(np.mean([r["mod_ratio_db"][b] for r in rs])), 2) for b in range(4)]}
        print(k, json.dumps(rep[k]), flush=True)
    (ROOT / "results/conv_p0/d96.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
