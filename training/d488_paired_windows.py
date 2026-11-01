"""D488: 窓ごと(先頭 0.5 / 2 / 8 s)の同じ判別器の対の差。本物 = d286_window と同じ REAL_POOL 150 人(各窓で特徴を取り直す)・偽物 = 全ての版をまとめた群(同じ発話の名前)。
窓ごとに判別器を 1 つ学習し(偽物は目標でまとめた 5 分割・本物は番号で 5 分割)、版ごとの決定値の中央値と、版 − 基準の発話ごとの差の平均を目標でまとめたブートストラップで出す(v1⊥ の射影)。
    uv run python d488_paired_windows.py --dir <dir with variant subdirs> --ref P43 --out <json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
WINS = (0.5, 2.0, 8.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()
    z = np.load(ROOT / "results/conv_p0/v1perp_basis.npz")
    Bs, mu_, sc_ = z["B"], z["mean"], z["scale"]
    tr = lambda X: ((X - mu_) / sc_) - (((X - mu_) / sc_) @ Bs.T) @ Bs

    @torch.no_grad()
    def emb(p: Path) -> dict:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        out = {}
        for w in WINS:
            y = x[: int(w * 16000)]
            y = (y - y.mean()) / (y.std() + 1e-7)
            hs = net(torch.from_numpy(y)[None].to("cuda"), output_hidden_states=True).hidden_states
            out[w] = torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()
        return out

    base = ROOT / "female-dataset"
    real = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((base / r["id"]).glob("*.wav"))
            real.append(emb(ws[len(ws) // 2]))
    D = Path(a.dir)
    vars_ = sorted(d.name for d in D.iterdir() if d.is_dir())
    keys = sorted(set.intersection(*[{f.name[:10] for f in (D / v).glob("p*.wav")} for v in vars_]))
    E = {v: [emb(next((D / v).glob(f"{k}_*.wav"))) for k in keys] for v in vars_}
    tg = np.array([int(k[7:10]) for k in keys])
    us = sorted(set(tg.tolist()))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    rng = np.random.default_rng(488)
    rep: dict = {"n": len(keys), "vars": vars_}
    rf = np.arange(len(real)) % 5
    for w in WINS:
        Xr = tr(np.array([e[w] for e in real]))
        Xf = np.concatenate([tr(np.array([e[w] for e in E[v]])) for v in vars_])
        tf = np.tile(tg, len(vars_))
        dv = np.zeros(len(Xf))
        for k5 in range(5):
            tr_r, tr_f = rf != k5, tf % 5 != k5
            Xtr = np.concatenate([Xr[tr_r], Xf[tr_f]])
            ytr = np.r_[np.zeros(tr_r.sum()), np.ones(tr_f.sum())]
            s = StandardScaler().fit(Xtr)
            c = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit(s.transform(Xtr), ytr)
            dv[~tr_f] = c.decision_function(s.transform(Xf[~tr_f]))
        per = {v: dv[i * len(keys):(i + 1) * len(keys)] for i, v in enumerate(vars_)}
        out = {"median": {v: round(float(np.median(per[v])), 3) for v in vars_}, "diff": {}}
        for v in vars_:
            if v == a.ref:
                continue
            d = per[v] - per[a.ref]
            bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
            out["diff"][f"{v}-{a.ref}"] = [round(float(d.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        rep[str(w)] = out
        print(w, json.dumps(out), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
