"""D14: 単位の平均からの残差(包絡 − 表の引き)に、発話をまたいで一貫した話者性(構音の動き方)があるかの事前検査。
話者 --n_spk 人(評価話者以外の日本語女声)。各話者: 参照 20s で表を作り、別の発話を学習 3・評価 2 に分ける。
残差 r = c1..c24(CheapTrick・教師 f0)− T[:, u](C1 の硬い単位)。入力の種類:
  R_full = r の 0.5 s 窓・R_demean = 発話ごとの平均を引いた r(動きだけ)・R_shuf = R_demean の時間をシャッフル(動きを壊す)・M_only = 発話の平均の残差ベクトル(24 次元・録音の色)。
分類器(小さな 1D CNN・M_only は線形)で評価発話の話者を当てる(n_spk 択一・発話ごとに窓の対数尤度を平均)。
    uv run python d14_residual_speaker.py --n_spk 300 --out ../results/conv_p0/d14_residual.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent
WIN = 100


def analyse(path: str):
    import f0hi as H
    import nvoc as N
    import pae as PA
    try:
        x = load48(path)[: 10 * 48000]
        n = len(x) // N.HOP
        if n < 300:
            return None
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        E = PA.envelope(xa, f0)
        e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
        E = 0.25 * e[:, :-2] + 0.5 * e[:, 1:-1] + 0.25 * e[:, 2:]
        act = (E[0] - PA.C0_SIL) / 10 > 1.0
        return x, E[1:25, :n].astype(np.float32), act[:n]
    except Exception:
        return None


class CNN(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(24, 128, 5, padding=2), nn.LeakyReLU(0.1), nn.Conv1d(128, 128, 5, padding=4, dilation=2), nn.LeakyReLU(0.1),
                                 nn.Conv1d(128, 128, 5, padding=8, dilation=4), nn.LeakyReLU(0.1))
        self.out = nn.Linear(256, n_out)

    def forward(self, x):
        h = self.net(x)
        return self.out(torch.cat([h.mean(-1), h.std(-1)], 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_spk", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import nvoc as N
    import train_c1 as T1
    import train_rvoc as TR
    dev = "cuda"
    hs = TR.held_speakers()
    allspk = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in (hs | TTS_LEAK))
    rng = random.Random(3)
    rng.shuffle(allspk)
    plan = []
    for s in allspk:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        if tot >= 18.0 and len(rest) >= 5:
            plan.append((s, ref, rest[:5]))
        if len(plan) >= a.n_spk:
            break
    print("speakers", len(plan), flush=True)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]

    @torch.no_grad()
    def units(x):
        n = len(x) // N.HOP
        return c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].argmax(1)[0].cpu().numpy()

    data = {"tr": [], "te": []}
    with Pool(a.procs) as pool:
        for si, (s, ref, rest) in enumerate(plan):
            res = pool.map(analyse, [str(w) for w in ref] + [str(w) for w in rest])
            rr, ro = res[: len(ref)], res[len(ref):]
            rr = [r for r in rr if r is not None]
            if not rr:
                continue
            Er = np.concatenate([r[1] for r in rr], 1)
            ur = np.concatenate([np.pad(units(r[0]), (0, max(0, r[1].shape[1] - len(units(r[0])))), mode="edge")[: r[1].shape[1]] for r in rr])
            T = C0.table(Er, ur, K, Cb)
            ok = [r for r in ro if r is not None]
            for j, r in enumerate(ok):
                u = units(r[0])
                u = np.pad(u, (0, max(0, r[1].shape[1] - len(u))), mode="edge")[: r[1].shape[1]]
                res_ = (r[1] - T[:, u])[:, r[2]]
                if res_.shape[1] < WIN + 10:
                    continue
                data["tr" if j < 3 else "te"].append((si, res_))
            if si % 50 == 0:
                print("analysed", si, flush=True)
    labels = sorted({si for si, _ in data["tr"]} & {si for si, _ in data["te"]})
    lab = {si: i for i, si in enumerate(labels)}
    tr = [(lab[si], r) for si, r in data["tr"] if si in lab]
    te = [(lab[si], r) for si, r in data["te"] if si in lab]
    S = len(labels)
    print("classes", S, "train utts", len(tr), "test utts", len(te), flush=True)

    def variant(r, kind, rng_):
        if kind == "R_full":
            return r
        d = r - r.mean(1, keepdims=True)
        if kind == "R_demean":
            return d
        return d[:, rng_.permutation(d.shape[1])]

    rep: dict = {"n_spk": S, "chance": round(1 / S, 4), "n_train_utt": len(tr), "n_test_utt": len(te)}
    Xm = np.stack([r.mean(1) for _, r in tr]); ym = np.array([l for l, _ in tr])
    Xt = np.stack([r.mean(1) for _, r in te]); yt = np.array([l for l, _ in te])
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    clf = LDA(solver="eigen", shrinkage=0.5).fit(Xm, ym)
    rep["M_only"] = {"utt_top1": round(float((clf.predict(Xt) == yt).mean()), 4)}
    print("M_only", rep["M_only"], flush=True)
    for kind in ("R_full", "R_demean", "R_shuf"):
        rng_ = np.random.default_rng(0)
        trv = [(l, variant(r, kind, rng_)) for l, r in tr]
        tev = [(l, variant(r, kind, rng_)) for l, r in te]
        sd = float(np.concatenate([r for _, r in trv], 1).std())
        net = CNN(S).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)

        def windows(lst, n_per):
            X, Y = [], []
            for l, r in lst:
                for _ in range(n_per):
                    t0 = rng_.integers(0, r.shape[1] - WIN)
                    X.append(r[:, t0:t0 + WIN]); Y.append(l)
            return torch.from_numpy(np.stack(X) / sd).float(), torch.tensor(Y)
        for ep in range(a.epochs):
            X, Y = windows(trv, 20)
            perm = torch.randperm(len(X))
            net.train()
            for i in range(0, len(X), 256):
                b = perm[i:i + 256]
                loss = nn.functional.cross_entropy(net(X[b].to(dev)), Y[b].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        correct_w, correct_u = [], []
        with torch.no_grad():
            for l, r in tev:
                ts = list(range(0, r.shape[1] - WIN, WIN // 2))
                X = torch.from_numpy(np.stack([r[:, t:t + WIN] for t in ts]) / sd).float().to(dev)
                lp = net(X).log_softmax(-1)
                correct_w += (lp.argmax(-1).cpu().numpy() == l).tolist()
                correct_u.append(int(lp.sum(0).argmax().item() == l))
        rep[kind] = {"window_top1": round(float(np.mean(correct_w)), 4), "utt_top1": round(float(np.mean(correct_u)), 4)}
        print(kind, rep[kind], flush=True)
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
