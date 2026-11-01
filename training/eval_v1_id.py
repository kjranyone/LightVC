"""目標関数 v1(本物の女声と区別できないこと)と本人らしさ(声域の中心の LDA / ECAPA・kNN-VC との順位差)を、名前 = glob の群ごとにまとめて測る。
v1 = 凍結 WavLM-large 層 4/6/8 の平均 + 標準偏差(8 秒)→ 標準化 → LogisticRegression(C=0.01)・本物 = D237 の REAL_POOL 150 人・5 分割(本物は話者・候補は目標で分ける)。
本人らしさ = d4b_lda_verifier と d146_identity_triangulate を、dev の cen_reg と群の wav(名前 = pNNNN_tTTT_<群>.wav に揃えて置く)で回す。KNN の群があれば順位差。
    uv run python eval_v1_id.py --s <dev の親> --work <作業ディレクトリ> --tag d257 --groups "S000=<glob>" "KNN=<glob>" --out ../results/conv_p0/d257.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_id", action="store_true")
    ap.add_argument("--set", default="dev", help="声域をそろえた中心の組(<s>/<set>/cen_reg・D390 で dev2 / dev3 にも広げる)")
    ap.add_argument("--limit", type=int, default=0, help="群ごとに先頭の n 本だけ(見当づけ用・0 = 全部)")
    a = ap.parse_args()
    S, W = Path(a.s), Path(a.work)
    G = {}
    for g in a.groups:
        name, pat = g.split("=", 1)
        G[name] = sorted(Path(p) for p in glob.glob(pat))
        if a.limit:
            G[name] = G[name][: a.limit]
        print(name, len(G[name]), flush=True)
    rep: dict = {}
    Rk, ok = None, set()
    if not a.no_id:
        W.mkdir(parents=True, exist_ok=True)
        for f in sorted((S / a.set / "cen_reg").glob("cen_*.wav")):
            if not (W / f.name).exists():
                os.symlink(f, W / f.name)
        for name, files in G.items():
            for f in files:
                key = "_".join(f.stem.split("_")[:2])
                dst = W / f"{key}_{name}.wav"
                if not dst.exists():
                    os.symlink(f.resolve(), dst)
        env = os.environ | {"OMP_NUM_THREADS": "2"}
        npz = W / f"{a.tag}_lda.npz"
        subprocess.run(["uv", "run", "python", "d4b_lda_verifier.py", "--wav_dir", str(W), "--out", str(ROOT / f"results/conv_p0/{a.tag}_lda.json"), "--dump", str(npz)], cwd=ROOT / "training", env=env, check=True, capture_output=True)
        subprocess.run(["uv", "run", "python", "d146_identity_triangulate.py", "--dir", str(W), "--tag", f"{a.tag}_tri"], cwd=ROOT / "training", env=env, check=True, capture_output=True)
        import d181_verdict as V
        Rk = V.ranks(npz, ROOT / f"results/conv_p0/{a.tag}_tri.json")
        ok = {m["ti"] for m in json.loads((S / a.set / "cen_reg" / "meta.json").read_text()) if m["ok"]}
        print("identity measured", flush=True)
    from rr17c_verdict import ci
    import soundfile as sf
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    cache = ROOT / "results/conv_p0/real_pool_wavlm_l468.npy"
    if cache.exists():
        R = np.load(cache)
    else:
        base = ROOT / "female-dataset"
        R = []
        for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
            if r["grp"] == "REAL_POOL":
                ws = sorted((base / r["id"]).glob("*.wav"))
                R.append(emb(ws[len(ws) // 2]))
        R = np.array(R)
        np.save(cache, R)
    rf = np.arange(len(R)) % 5
    for name, files in G.items():
        X = np.array([emb(f) for f in files])
        tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
        Xa = np.concatenate([R, X])
        ya = np.r_[np.zeros(len(R)), np.ones(len(X))]
        fa = np.r_[rf, tg % 5]
        pr = np.zeros(len(ya))
        for k in range(5):
            te = fa == k
            sc = StandardScaler().fit(Xa[~te])
            clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
            pr[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
        rep[name] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "p_fake_median": round(float(np.median(pr[len(R):])), 4), "n": len(X)}
        if Rk is not None:
            for m in ("LDA", "ECAPA"):
                lv = [Rk[m][name][k] for k in Rk[m].get(name, {}) if int(k.split("_t")[1]) in ok]
                rep[name][f"{m}_mean_rank"] = round(float(np.mean(lv)), 2) if lv else None
                if name != "KNN" and "KNN" in Rk[m] and name in Rk[m]:
                    rows = [("x", int(k.split("_t")[1]), Rk[m][name][k] - Rk[m]["KNN"][k]) for k in sorted(set(Rk[m][name]) & set(Rk[m]["KNN"])) if int(k.split("_t")[1]) in ok]
                    rep[name][f"{m}_minus_KNN"] = ci(rows, np.random.default_rng(0))
        print(name, rep[name], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
