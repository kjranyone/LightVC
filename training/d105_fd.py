"""D105(学習なし): 本物との分布の距離(連続量・D95 の AUC が 1.00 に張り付いて向きが見えないため)。
特徴 = D94/D95 と同じ WavLM-large 層 4/6/8 の平均 + 標準偏差(6144)→ 本物の女声のプール(評価外 300 人)で当てた PCA 64 次元 → フレシェ距離(FD)。
本物の基準 = プールの別の 150 人の FD(自然の水準)。比べる組 = --sets <dir>:<cond> ...。出力 = results/conv_p0/<tag>.json。
    uv run python d105_fd.py --s <scratchpad> --sets d103:TAB d103:C07 x385:C07 --tag d105
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def fd(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = np.cov(a.T), np.cov(b.T)
    cs = sqrtm(ca @ cb)
    if np.iscomplexobj(cs):
        cs = cs.real
    return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * cs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--tag", default="d105")
    ap.add_argument("--pool", default="real", choices=("real", "tts"), help="本物の基準: real = female-dataset・tts = 女声 TTS(neutral・D80 の目標と偽者を除く)")
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import WavLMModel
    S = Path(a.s)
    dev = "cuda"
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()

    def load16(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        return resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    excl = set((S / "excl.txt").read_text().split()) if (S / "excl.txt").exists() else set()
    base = ROOT / ("female-dataset" if a.pool == "real" else "data/female_tts_corpus")
    if a.pool == "tts":
        rows = json.loads((ROOT / "results/conv_p0/d80.json").read_text())["rows"]
        excl |= {r["pair"].split("_")[1] for r in rows}
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(105).shuffle(pool)
    R = []
    for spk in pool[:450]:
        ws = sorted((base / spk).glob("*.wav")) if a.pool == "real" else sorted((base / spk).glob("t0[5-9]_neutral.wav"))
        if ws:
            x = load16(ws[len(ws) // 2])
            if len(x) > 3 * 16000:
                R.append(emb(x))
    R = np.array(R)
    Rfit, Rref, Rnat = R[:300], R[300:450], None
    mu, sd = Rfit.mean(0), Rfit.std(0) + 1e-6
    Z = (Rfit - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P = Vt[:64].T
    proj = lambda X: ((X - mu) / sd) @ P
    rep = {"real_vs_real": round(fd(proj(Rfit[:150]), proj(Rref)), 2)}
    print("real_vs_real", rep["real_vs_real"], flush=True)
    for spec in a.sets:
        d, c = spec.split(":")
        X = np.array([emb(load16(f)) for f in sorted((S / d).glob(f"*_{c}.wav"))])
        rep[spec] = round(fd(proj(X), proj(Rref)), 2)
        print(spec, rep[spec], len(X), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
