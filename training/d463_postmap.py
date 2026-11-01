"""D463: 男声の操作者の C1 の事後を、内容を保ったまま女声の事後の分布へ写す小さな写像 f(フレーム・因果な 3 フレームの文脈・残差の MLP)が作れるか(推論の入口だけ・生成器はそのまま)。
D462: P43OP の男声の名残(VCTK の内容をそろえた対 0.83)の最大の運び手は事後の柔らかい形(硬くすると 0.60)。
データ = data/pair_tts_v4l(同じ文の男声 pm / 女声 pf・学習の操作者の側)。f の学習 = フレームの判別器 D(男 / 女)を騙す敵対 + 内容の保存(f の出力の argmax を元の事後の argmax に合わせる交差エントロピー)。整列は使わない。
評価(保留の番号の話者): (a) D434 と同じ発話の統計の男女の判別器(摂動なしの学習の話者で学習)で f(男声) の p(男)・AUC(女声 raw 対)・(b) argmax の一致(f 対 元)。
    uv run python d463_postmap.py --out ../results/conv_p0/d463
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/pair_tts_v4l"
K = 200


class PostMap(nn.Module):
    """対数事後 [B, K, T] → 対数事後(因果な 3 フレームの文脈・残差)。"""

    def __init__(self, k: int = K, h: int = 256):
        super().__init__()
        self.c1 = nn.Conv1d(k, h, 3)
        self.c2 = nn.Conv1d(h, h, 1)
        self.c3 = nn.Conv1d(h, k, 1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)

    def forward(self, lp: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.c1(F.pad(lp, (2, 0))))
        h = F.gelu(self.c2(h))
        return torch.log_softmax(lp + self.c3(h), 1)


def posts(dev: str) -> dict:
    import c1_content as CC
    import train_c1 as T1
    from d6_unit_cov import load48
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1.load_state_dict(st1["net"])
    c1.eval()
    mf = CC.MelFront().to(dev)
    out = {}
    with torch.no_grad():
        for s in sorted(D.glob("p[mf]*")):
            for w in sorted(s.glob("t*.wav")):
                x = load48(w)
                n = len(x) // 240
                lg = c1(mf(torch.from_numpy(T1.prime(x))[None].to(dev)))[0, :, -n:]
                out[(s.name, w.stem)] = torch.log_softmax(lg, 0).cpu()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ct", type=float, default=1.0)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(463)
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    P = posts(dev)
    num = lambda s: int(s[2:])
    test = {n for n in range(40) if n % 4 == 0}
    tr_m = [v for (s, t), v in P.items() if s.startswith("pm") and num(s) not in test]
    tr_f = [v for (s, t), v in P.items() if s.startswith("pf") and num(s) not in test]
    f = PostMap().to(dev)
    dsc = nn.Sequential(nn.Conv1d(K, 256, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(256, 256, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(256, 1, 1)).to(dev)
    of = torch.optim.AdamW(f.parameters(), 2e-4, betas=(0.5, 0.99))
    od = torch.optim.AdamW(dsc.parameters(), 2e-4, betas=(0.5, 0.99))
    rng = np.random.default_rng(463)
    L = 400

    def batch(pool: list, n: int = 16) -> torch.Tensor:
        xs = []
        for i in rng.integers(0, len(pool), n):
            v = pool[i]
            if v.shape[1] <= L:
                v = F.pad(v, (0, L + 1 - v.shape[1]), value=float(np.log(1 / K)))
            o = int(rng.integers(0, v.shape[1] - L))
            xs.append(v[:, o:o + L])
        return torch.stack(xs).to(dev)

    log = []
    for step in range(1, a.steps + 1):
        xm, xf = batch(tr_m), batch(tr_f)
        ym = f(xm)
        dl = F.softplus(-dsc(xf.exp())).mean() + F.softplus(dsc(ym.detach().exp())).mean()
        od.zero_grad()
        dl.backward()
        od.step()
        adv = F.softplus(-dsc(ym.exp())).mean()
        ct = F.nll_loss(ym, xm.argmax(1))
        loss = adv + a.ct * ct
        of.zero_grad()
        loss.backward()
        of.step()
        if step % 500 == 0:
            r = {"step": step, "dl": round(float(dl), 3), "adv": round(float(adv), 3), "ct": round(float(ct), 3)}
            print(json.dumps(r), flush=True)
            log.append(r)
    f.eval()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    ut = lambda lp: np.r_[lp.mean(1), lp.std(1)]
    Xtr = np.stack([ut(v.numpy()) for (s, t), v in P.items() if num(s) not in test])
    ytr = np.array([1 if s.startswith("pm") else 0 for (s, t), v in P.items() if num(s) not in test])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.05, max_iter=4000).fit(sc.transform(Xtr), ytr)
    te = [(s, t, v) for (s, t), v in P.items() if num(s) in test]
    rep: dict = {"log": log}
    with torch.no_grad():
        pm_raw = [ut(v.numpy()) for s, t, v in te if s.startswith("pm")]
        pf_raw = [ut(v.numpy()) for s, t, v in te if s.startswith("pf")]
        mapped = [f(v[None].to(dev))[0].cpu() for s, t, v in te if s.startswith("pm")]
        pm_map = [ut(v.numpy()) for v in mapped]
        keep = float(np.mean(np.concatenate([(m.argmax(0) == v.argmax(0)).numpy() for m, (s, t, v) in zip(mapped, [x for x in te if x[0].startswith("pm")])])))
        pf_mapped = [f(v[None].to(dev))[0].cpu() for s, t, v in te if s.startswith("pf")]
        keep_f = float(np.mean(np.concatenate([(m.argmax(0) == v.argmax(0)).numpy() for m, (s, t, v) in zip(pf_mapped, [x for x in te if x[0].startswith("pf")])])))
    pr = lambda X: clf.predict_proba(sc.transform(np.stack(X)))[:, 1]
    a_raw, a_map, a_f = pr(pm_raw), pr(pm_map), pr(pf_raw)
    rep["p_male_median"] = {"male_raw": round(float(np.median(a_raw)), 3), "male_mapped": round(float(np.median(a_map)), 3), "female_raw": round(float(np.median(a_f)), 3), "female_mapped": round(float(np.median(pr([ut(v.numpy()) for v in pf_mapped]))), 3)}
    rep["auc_vs_female_raw"] = {"male_raw": round(float(roc_auc_score(np.r_[np.ones(len(a_raw)), np.zeros(len(a_f))], np.r_[a_raw, a_f])), 4), "male_mapped": round(float(roc_auc_score(np.r_[np.ones(len(a_map)), np.zeros(len(a_f))], np.r_[a_map, a_f])), 4)}
    rep["argmax_keep"] = {"male": round(keep, 3), "female": round(keep_f, 3)}
    print(json.dumps({k: v for k, v in rep.items() if k != "log"}), flush=True)
    torch.save({"net": f.state_dict(), "k": K}, O / "postmap.pt")
    (O / "summary.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
