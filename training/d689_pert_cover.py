"""D689(腕 63 の検査 I の探針・学習前): 入力の側の摂動(train_rvoc の channel / sex)が、生成器の入力の条件(C2 の特徴・C1 の事後)を本物の男声の入力の側へどれだけ寄せるか。
発話ごとの C2 の平均・標準偏差 + 事後の平均(エントロピー)で、学習の女声(きれい)対 本物の男声の入力(外郎売・VCTK の男声)の線形の判別器を作り、摂動した女声の決定値の位置を測る。
    uv run python d689_pert_cover.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

import c1_content as CC
import nvoc as N
import train_ppgvoc as TP
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = ROOT / "results/diag_ppgvoc63"
SEC = 6.0


def load48(p: str) -> np.ndarray:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    x = x.mean(1)
    return (x if sr == N.SR else soxr.resample(x, sr, N.SR)).astype(np.float32)


@torch.no_grad()
def feats(x: np.ndarray, c1n, mfc, dev: str) -> np.ndarray:
    n = int(SEC * N.SR) // N.HOP * N.HOP
    x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    T = n // N.HOP
    a = torch.from_numpy(np.concatenate([np.zeros(N.WIN - N.HOP, np.float32), x]))[None].to(dev)
    z = TP.cvin_frames(a, T)[0].float().cpu().numpy()
    q = c1n(mfc(a))[..., -T:].softmax(1)[0].float().cpu().numpy()
    e = 10 * np.log10(np.convolve(x ** 2, np.ones(N.HOP) / N.HOP, "same")[:: N.HOP][:T] + 1e-10)
    v = e > e.max() - 35
    z, q = z[:, v], q[:, v]
    ent = -(q * np.log(q + 1e-9)).sum(0)
    return np.concatenate([z.mean(1), z.std(1), [ent.mean(), q.max(0).mean()]])


def main() -> int:
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    TP.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"])
    c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    rows, _, _ = TP.load_rows({h[0] for h in TP.e2e_held()}, True)
    rng = random.Random(689)
    pick = [r for r in rng.sample(rows, 600) if r[3] >= SEC][:240]
    tr, te = pick[:160], pick[160:]
    F = lambda x: feats(x, c1n, mfc, dev)
    clean_tr = np.stack([F(load48(r[0])) for r in tr])
    male = []
    for s, kd in (("d487_uiro", None), ("devvctkm", "ME")):
        for r in json.loads((H / s / "devset.json").read_text())["pairs"]:
            if kd is None or r["kind"] == kd:
                male.append((s, F(load48(str(H / s / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav")))))
    mi = list(range(len(male)))
    random.Random(1).shuffle(mi)
    m_tr = np.stack([male[i][1] for i in mi[: len(mi) // 2]])
    m_te = [male[i] for i in mi[len(mi) // 2:]]
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([clean_tr, m_tr])
    y = np.r_[np.zeros(len(clean_tr)), np.ones(len(m_tr))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X), y)
    dfn = lambda Z: clf.decision_function(sc.transform(np.atleast_2d(Z)))
    rep = {"n_clean_train": len(clean_tr), "n_male_train": len(m_tr)}
    arms = {"clean": [], "channel": [], "sex": [], "sex_shift_only": [], "both": []}
    for r in te:
        x = load48(r[0])
        arms["clean"].append(F(x))
        arms["channel"].append(F(TR.channel_apply(x, TR.channel_params(rng))))
        for _ in range(10):
            ps = TR.sex_params(rng)
            if ps[0] < 1.0:
                break
        arms["sex_shift_only"].append(F(TR.sex_apply(x, ps)))
        ps = TR.sex_params(rng)
        arms["sex"].append(F(TR.sex_apply(x, ps)))
        arms["both"].append(F(TR.sex_apply(TR.channel_apply(x, TR.channel_params(rng)), TR.sex_params(rng))))
    c_med = float(np.median(dfn(np.stack(arms["clean"]))))
    for s in ("d487_uiro", "devvctkm"):
        g = np.stack([z for s_, z in m_te if s_ == s])
        rep[f"male_test_{s}"] = round(float(np.median(dfn(g))), 3)
    m_med = float(np.median(dfn(np.stack([z for _, z in m_te]))))
    rep["clean_test"] = round(c_med, 3)
    rep["male_test"] = round(m_med, 3)
    for k, v in arms.items():
        d = float(np.median(dfn(np.stack(v))))
        rep[f"pert_{k}"] = {"median_decision": round(d, 3), "frac_of_gap": round((d - c_med) / (m_med - c_med), 3)}
    print(json.dumps(rep, indent=1, ensure_ascii=False), flush=True)
    (OUT / "d689_pert_cover.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
