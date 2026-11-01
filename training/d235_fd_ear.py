"""D235(学習なし・results/conv_p0/d235_prereg.yaml): D105 の分布の距離(WavLM-large 層 4/6/8 の平均 + 標準偏差 → 本物の女声 300 人の PCA 64 → FD)が耳 cand_ab1 の順位(kNN-VC > G03S10ZH > RRQSDC19LE)と同じ向きに並ぶか。
dev・dev2・dev3 の 60 組ずつ・方式ごとの FD(目標のクラスタのブートストラップ)・発話ごとの白色化した d² と対の差・耳の 3 試行のクリップの d²。
    uv run python d235_fd_ear.py --s <dev の親> --out ../results/conv_p0/d235.json
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
from d105_fd import fd

ROOT = Path(__file__).resolve().parent.parent
SETS = ("dev", "dev2", "dev3")
FORMS = {"KNN": ("d135_real", "KNN"), "G03S10ZH": ("d114", "G03S10ZH"), "RRQSDC19LE": ("d139", "RRQSDC19LE"), "SRC": ("d56", "SRC")}
EAR = {"cand_01": ("dev3", "p0019_t009"), "cand_02": ("dev2", "p0058_t029"), "cand_03": ("dev", "p0055_t027")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    import soundfile as sf
    from scipy.signal import resample_poly
    from transformers import WavLMModel
    S = Path(a.s)
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to("cuda").eval().float()

    def load16(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        return resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to("cuda"), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    excl = set((S / "excl.txt").read_text().split())
    for f in list(S.glob("*/devset.json")) + [S / "eval_targets.json"]:
        if f.exists():
            j = json.loads(f.read_text())
            excl |= set(j["targets"] if isinstance(j, dict) else j)
    base = ROOT / "female-dataset"
    pool = sorted(d.name for d in base.iterdir() if d.is_dir() and d.name not in excl)
    random.Random(105).shuffle(pool)
    R = []
    for spk in pool:
        ws = sorted((base / spk).glob("*.wav"))
        if ws:
            x = load16(ws[len(ws) // 2])
            if len(x) > 3 * 16000:
                R.append(emb(x))
        if len(R) == 450:
            break
    R = np.array(R)
    Rfit, Rref = R[:300], R[300:]
    mu, sd = Rfit.mean(0), Rfit.std(0) + 1e-6
    Z = (Rfit - mu) / sd
    _, sv, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P = Vt[:64].T
    proj = lambda X: ((X - mu) / sd) @ P
    Pf = proj(Rfit)
    pm, ps = Pf.mean(0), Pf.std(0) + 1e-9
    d2 = lambda X: (((proj(X) - pm) / ps) ** 2).sum(1)
    ref_d2 = np.sort(d2(Rref))
    rep: dict = {"n_excl": len(excl), "real_vs_real": round(fd(Pf[:150], proj(Rref)), 1), "real_ref_d2_median": round(float(np.median(ref_d2)), 1)}
    E: dict = {}
    for D in SETS:
        for nm, (d, c) in FORMS.items():
            for f in sorted((S / D / d).glob(f"*_{c}.wav")):
                k = f.name[: -len(c) - 5]
                E[(nm, D, k)] = emb(load16(f))
        for f in sorted((S / D / "cen_reg").glob("cen_*.wav")):
            E[("CEN", D, f.stem)] = emb(load16(f))
        print(D, "done", len(E), flush=True)
    names = list(FORMS) + ["CEN"]
    tg = lambda D, k: f"{D}:{k.split('_t')[-1]}"
    rng = np.random.default_rng(0)
    Pref = proj(Rref)
    rep["fd"], rep["fd_sets"], rep["d2_median"] = {}, {}, {}
    for nm in names:
        ks = [(D, k) for (n, D, k) in E if n == nm]
        X = np.array([E[(nm, D, k)] for D, k in ks])
        cl = np.array([tg(D, k) for D, k in ks])
        us = sorted(set(cl))
        idx = {u: np.flatnonzero(cl == u) for u in us}
        PX = proj(X)
        bs = [fd(PX[np.concatenate([idx[u] for u in rng.choice(us, len(us))])], Pref) for _ in range(a.nboot)]
        rep["fd"][nm] = {"fd": round(fd(PX, Pref), 1), "ci95": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)], "n": len(ks)}
        rep["fd_sets"][nm] = {D: round(fd(PX[[i for i, (DD, _) in enumerate(ks) if DD == D]], Pref), 1) for D in SETS}
        rep["d2_median"][nm] = round(float(np.median(d2(X))), 1)
        print(nm, rep["fd"][nm], rep["fd_sets"][nm], "d2", rep["d2_median"][nm], flush=True)
    U = {nm: {(D, k): float(d2(E[(nm, D, k)][None])[0]) for (n, D, k) in E if n == nm} for nm in FORMS}
    rep["pair_d2"] = {}
    for x, y in (("RRQSDC19LE", "G03S10ZH"), ("G03S10ZH", "KNN"), ("RRQSDC19LE", "KNN")):
        ks = sorted(set(U[x]) & set(U[y]))
        v = np.array([U[x][q] - U[y][q] for q in ks])
        cl = np.array([tg(*q) for q in ks])
        us = sorted(set(cl))
        idx = {u: np.flatnonzero(cl == u) for u in us}
        bs = [np.mean(v[np.concatenate([idx[u] for u in rng.choice(us, len(us))])]) for _ in range(a.nboot)]
        rep["pair_d2"][f"{x}-{y}"] = {"mean": round(float(v.mean()), 1), "ci95": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)], "frac_pos": round(float((v > 0).mean()), 3)}
    for x, y in (("RRQSDC19LE", "G03S10ZH"), ("G03S10ZH", "KNN")):
        kx = [(D, k) for (n, D, k) in E if n == x]
        cl = np.array([tg(D, k) for D, k in kx])
        us = sorted(set(cl))
        PX = proj(np.array([E[(x, D, k)] for D, k in kx]))
        PY = proj(np.array([E[(y, D, k)] for D, k in kx]))
        idx = {u: np.flatnonzero(cl == u) for u in us}
        bs = []
        for _ in range(a.nboot):
            ii = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
            bs.append(fd(PX[ii], Pref) - fd(PY[ii], Pref))
        rep["pair_d2"][f"FD:{x}-{y}"] = {"diff": round(fd(PX, Pref) - fd(PY, Pref), 1), "ci95": [round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1)]}
    rep["ear_clips_d2"] = {t: {nm: round(U[nm][(D, k)], 1) for nm in ("KNN", "G03S10ZH", "RRQSDC19LE", "SRC")} | {"CEN": round(float(d2(E[("CEN", D, f"cen_{int(k.split('_t')[1]):03d}")][None])[0]), 1)} for t, (D, k) in EAR.items()}
    rep["ear_clips_d2_real_pct"] = {t: {nm: round(float(np.searchsorted(ref_d2, v) / len(ref_d2)), 3) for nm, v in row.items()} for t, row in rep["ear_clips_d2"].items()}
    F = {nm: rep["fd"][nm]["fd"] for nm in names}
    lo = lambda key: rep["pair_d2"][key]["ci95"][0]
    hi = lambda key: rep["pair_d2"][key]["ci95"][1]
    order = F["KNN"] < F["G03S10ZH"] < F["RRQSDC19LE"]
    sets_ok = sum(rep["fd_sets"]["KNN"][D] < rep["fd_sets"]["G03S10ZH"][D] < rep["fd_sets"]["RRQSDC19LE"][D] for D in SETS)
    far = all(F[nm] > 2 * F["CEN"] for nm in ("KNN", "G03S10ZH", "RRQSDC19LE"))
    cons = order and lo("FD:RRQSDC19LE-G03S10ZH") > 0 and lo("FD:G03S10ZH-KNN") > 0 and sets_ok >= 2 and far
    inc = hi("FD:RRQSDC19LE-G03S10ZH") < 0 or hi("FD:G03S10ZH-KNN") < 0
    rep["checks"] = {"order": order, "sets_ok": sets_ok, "far_from_real": far}
    rep["verdict"] = "EAR_CONSISTENT" if cons else "INCONSISTENT" if inc else "UNRESOLVED"
    print(json.dumps({k: rep[k] for k in ("real_vs_real", "pair_d2", "ear_clips_d2", "ear_clips_d2_real_pct", "checks", "verdict")}, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
