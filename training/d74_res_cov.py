"""D74(診断・学習なし): 動きの統計(res の 24 次元の共分散)は話者ごとに違うか。違えば、写した動きに目標の共分散をかぶせる変換(res_out = C_tgt^½ C_src^−½ res_src)が「本人らしさ」を足しうる。
女声 TTS(同じ 10 文・neutral)の話者 --n 人: 各話者の 10 文を前半 5 / 後半 5 に分けて共分散を作り、Förstner 距離 d(C1, C2) = sqrt(Σ log² λ_i(C1⁻¹C2)) を
同じ話者の前半 vs 後半(話者の中のぶれ)と別の話者どうしで比べる。res = 本物の包絡 − 同じ発話の単位平均の事後混合(D68 と同じ)。出力 = results/conv_p0/d74.json。
    uv run python d74_res_cov.py --n 24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/female_tts_corpus"


def forstner(a: np.ndarray, b: np.ndarray) -> float:
    lam = eigh(b, a, eigvals_only=True)
    return float(np.sqrt((np.log(np.maximum(lam, 1e-12)) ** 2).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    dev = "cuda"
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]

    def res(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        with torch.no_grad():
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        er = env[1:25]
        r = er - C0.table(er, p.argmax(0), K, Cb) @ p
        return r - r.mean(1, keepdims=True)

    spks = sorted(d.name for d in CORPUS.iterdir() if d.is_dir() and all((d / f"t{i:02d}_neutral.wav").exists() for i in range(10)))
    random.Random(74).shuffle(spks)
    spks = spks[: a.n]
    halves = []
    for s in spks:
        rs = [res(load48(CORPUS / s / f"t{i:02d}_neutral.wav").astype(np.float32)) for i in range(10)]
        h = []
        for part in (rs[:5], rs[5:]):
            R = np.concatenate(part, 1)
            h.append(R @ R.T / R.shape[1] + 1e-6 * np.eye(24))
        halves.append(h)
        print(s, "within", round(forstner(h[0], h[1]), 3), "trace", round(float(np.trace(h[0])), 3), flush=True)
    within = [forstner(h[0], h[1]) for h in halves]
    between = [forstner(halves[i][0], halves[j][1]) for i in range(len(spks)) for j in range(len(spks)) if i != j]
    tr = [float(np.trace(h[0])) for h in halves]
    rep = {"n_spk": len(spks), "forstner_within_mean": round(float(np.mean(within)), 4), "forstner_between_mean": round(float(np.mean(between)), 4),
           "forstner_between_p10": round(float(np.percentile(between, 10)), 4), "forstner_within_p90": round(float(np.percentile(within, 90)), 4),
           "trace_mean": round(float(np.mean(tr)), 4), "trace_cv": round(float(np.std(tr) / np.mean(tr)), 4)}
    print(json.dumps(rep), flush=True)
    (ROOT / "results/conv_p0/d74.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
