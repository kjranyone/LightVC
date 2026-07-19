from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, HOP, SR, N_MELS, DEV

W = 8
K = 4
CDIM = 32


class MLP(nn.Module):
    def __init__(self, i: int, o: int, h: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.1),
                                 nn.Linear(h, h), nn.LeakyReLU(0.1), nn.Linear(h, o))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data(feat_root: str, n: int) -> tuple:
    random.seed(0)
    files = list(Path(feat_root).rglob("*.pt"))
    random.shuffle(files)
    conts, seqs, mels = [], [], []
    got = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        if "f0" not in d:
            continue
        f0 = d["f0"].numpy()
        if (f0 > 1).sum() < 40:
            continue
        try:
            w, _ = librosa.load(d["path"], sr=SR, mono=True)
        except Exception:
            continue
        t = f0.shape[0]
        w = w[: t * HOP]
        if len(w) < t * HOP:
            w = np.pad(w, (0, t * HOP - len(w)))
        with torch.no_grad():
            mel = mel_of(torch.from_numpy(w).float().unsqueeze(0).to(DEV))[0].cpu().numpy()
        mel = mel[:, :t]
        if mel.shape[1] < t:
            continue
        cont = d["content"].float().numpy()
        c = F.interpolate(torch.from_numpy(cont).t().unsqueeze(0), size=t,
                          mode="linear", align_corners=False).squeeze(0).numpy()
        lf = np.log(np.clip(f0, 1, None))
        v = (f0 > 1).astype(np.float32)
        lf = (lf - lf[f0 > 1].mean()) * v
        eng = np.log(np.clip(d["energy"].numpy(), 1e-4, None))
        conts.append(c.T)
        seqs.append(np.stack([lf, v, eng], 0))
        mels.append(mel)
        got += 1
        if got >= n:
            break
    return conts, seqs, mels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    conts, seqs, mels = load_data(args.feat, args.n)
    print(f"loaded {len(conts)} utts")
    allc = np.concatenate(conts, 0)
    mu = allc.mean(0)
    U, S, Vt = np.linalg.svd(allc[:: max(1, len(allc) // 20000)] - mu, full_matrices=False)
    pca = Vt[:CDIM].T

    def redseq(i):
        cr = (conts[i] - mu) @ pca
        return np.concatenate([cr, seqs[i].T], 1).astype(np.float32)

    reds = [redseq(i) for i in range(len(conts))]
    rdim = reds[0].shape[1]

    ntr = int(len(reds) * 0.85)
    tr, te = list(range(ntr)), list(range(ntr, len(reds)))

    def sample(idxs, bs):
        pa, fu, ta = [], [], []
        for _ in range(bs):
            i = random.choice(idxs)
            r = reds[i]
            T = r.shape[0]
            if T < W + K + 2:
                continue
            t = random.randint(W, T - K - 1)
            pa.append(r[t - W:t + 1].reshape(-1))
            fu.append(r[t + 1:t + 1 + K].reshape(-1))
            ta.append(mels[i][:, t])
        return (torch.tensor(np.array(pa), device=DEV),
                torch.tensor(np.array(fu), device=DEV),
                torch.tensor(np.array(ta), device=DEV))

    pdim = rdim * (W + 1)
    fdim = rdim * K
    pred = MLP(pdim, fdim).to(DEV)
    g_causal = MLP(pdim, N_MELS).to(DEV)
    g_oracle = MLP(pdim + fdim, N_MELS).to(DEV)
    g_self = MLP(pdim + fdim, N_MELS).to(DEV)
    params = (list(pred.parameters()) + list(g_causal.parameters())
              + list(g_oracle.parameters()) + list(g_self.parameters()))
    opt = torch.optim.AdamW(params, 3e-4)

    for step in range(args.steps):
        pa, fu, ta = sample(tr, args.batch)
        fu_pred = pred(pa)
        pred_l = F.l1_loss(fu_pred, fu)
        l_c = F.l1_loss(g_causal(pa), ta)
        l_o = F.l1_loss(g_oracle(torch.cat([pa, fu], 1)), ta)
        l_s = F.l1_loss(g_self(torch.cat([pa, fu_pred.detach()], 1)), ta)
        loss = pred_l + l_c + l_o + l_s
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step} pred {pred_l.item():.3f} causal {l_c.item():.3f} "
                  f"oracle {l_o.item():.3f} self {l_s.item():.3f}", flush=True)

    vidx = CDIM + 1

    def eval_group(idxs, name, ftype):
        pa, fu, ta = [], [], []
        for i in idxs:
            r = reds[i]
            T = r.shape[0]
            for t in range(W, T - K - 1):
                vt = r[t, vidx] > 0.5
                vfut = r[t + 2, vidx] > 0.5 if t + 2 < T else vt
                vpast = r[t - 2, vidx] > 0.5
                if ftype == "release" and not (vt and not vfut):
                    continue
                if ftype == "onset" and not (vt and not vpast):
                    continue
                if ftype == "steady" and not (vt and vfut and vpast):
                    continue
                pa.append(r[t - W:t + 1].reshape(-1))
                fu.append(r[t + 1:t + 1 + K].reshape(-1))
                ta.append(mels[i][:, t])
        if len(ta) < 50:
            print(f"  {name:18} (n<50, skip)")
            return
        pa = torch.tensor(np.array(pa), device=DEV)
        fu = torch.tensor(np.array(fu), device=DEV)
        ta = torch.tensor(np.array(ta), device=DEV)
        with torch.no_grad():
            fp = pred(pa)
            var = ta.var().item()
            def r2(pm):
                return 1 - ((pm - ta) ** 2).mean().item() / var
            rc = r2(g_causal(pa))
            ro = r2(g_oracle(torch.cat([pa, fu], 1)))
            rs = r2(g_self(torch.cat([pa, fp], 1)))
        gap = ro - rc
        rec = (rs - rc) / gap * 100 if abs(gap) > 1e-4 else float("nan")
        print(f"  {name:18} causal {rc:.3f} | self {rs:.3f} | oracle {ro:.3f} "
              f"| 未来価値 {gap:+.3f} | 予測回収 {rec:.0f}%  (n={len(ta)})")

    print("\n=== frame 種別ごとの未来文脈の価値 (oracle-causal) ===")
    eval_group(te, "全体", "all")
    eval_group(te, "release(母音末/抜け)", "release")
    eval_group(te, "onset(立ち上がり)", "onset")
    eval_group(te, "steady(定常有声)", "steady")
    print("\n未来価値>0の種別があれば自己予測に的あり。全部≈0なら過去だけで十分(=単純causal)。")


if __name__ == "__main__":
    main()
