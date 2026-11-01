"""D286: 偽物らしさの差(自己再構成 SR 対 別人の女声 → 女声 FF 対 男声 → 女声 PV・diag_ppgvoc1 の 5k)がどの窓の長さから現れるか。
v1 と同じ特徴(凍結 WavLM-large 層 4/6/8 の平均 + 標準偏差・標準化・LogisticRegression C 0.01・本物 = REAL_POOL 150 人の真ん中のファイル)を
先頭 0.5 / 2 / 8 秒で作り、群ごとの決定値の中央値を比べる(本物は話者・候補は目標で 5 分割)。0.5 秒で差が出る = 局所(短い区間の判別器で押せる)。
    uv run python d286_window.py --out ../results/conv_p0/d286.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
WINS = (0.5, 2.0, 8.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--perp", action="store_true", help="v1⊥: results/conv_p0/v1perp_basis.npz の標準化と INLP の基底で射影してから比べる(8 秒の特徴で作った基底を短い窓にも当てる = 近似)")
    ap.add_argument("--eq_dir", default="", help="群ごとの置き場 <dir>/<群>/*.wav(下のディレクトリを全て群にする・群の大きさをそろえた組・D290: 群ごとの C2ST の決定値は群の大きさに依存する)")
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

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
    G = {"SR": sorted(glob.glob(str(ROOT / "results/diag_ppgvoc1/ev_005000/h*_SR.wav"))),
         "FF": sorted(glob.glob(str(ROOT / "results/conv_p0/d284_devff/g000/PV/*.wav"))),
         "PV": sorted(glob.glob(str(ROOT / "results/diag_ppgvoc1/ev_005000/p*_PV.wav"))),
         "CP": sorted(glob.glob(str(ROOT / "results/diag_e2e1/eval/h*_CP.wav")))}
    if a.eq_dir:
        G = {d.name: sorted(glob.glob(str(d / "*.wav"))) for d in sorted(Path(a.eq_dir).iterdir()) if d.is_dir()}
    E = {k: [emb(Path(f)) for f in v] for k, v in G.items()}
    rep: dict = {k: {"n": len(v)} for k, v in G.items()}
    rf = np.arange(len(real)) % 5
    tr = lambda X: X
    if a.perp:
        z = np.load(ROOT / "results/conv_p0/v1perp_basis.npz")
        Bs, mu_, sc_ = z["B"], z["mean"], z["scale"]
        tr = lambda X: ((X - mu_) / sc_) - (((X - mu_) / sc_) @ Bs.T) @ Bs
    for w in WINS:
        R = tr(np.array([e[w] for e in real]))
        for k, files in G.items():
            X = tr(np.array([e[w] for e in E[k]]))
            tg = np.array([int(Path(f).name.split("_t")[1][:3]) for f in files])
            Xa, ya, fa = np.concatenate([R, X]), np.r_[np.zeros(len(R)), np.ones(len(X))], np.r_[rf, tg % 5]
            dv = np.zeros(len(ya))
            for f in range(5):
                te = fa == f
                sc = StandardScaler().fit(Xa[~te])
                dv[te] = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te]).decision_function(sc.transform(Xa[te]))
            rep[k][f"logit_{w}s"] = round(float(np.median(dv[len(R):])), 3)
        print(w, {k: rep[k][f"logit_{w}s"] for k in G}, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
