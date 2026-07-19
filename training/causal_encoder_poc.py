from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import N_MELS, DEV
from selfpred_poc import load_data, MLP

WM = 12
KA = 3
CDIM = 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    conts, seqs, mels = load_data(args.feat, args.n)
    print(f"loaded {len(conts)} utts")
    allc = np.concatenate(conts, 0)
    mu = allc.mean(0)
    sub = allc[:: max(1, len(allc) // 20000)] - mu
    _, _, Vt = np.linalg.svd(sub, full_matrices=False)
    pca = Vt[:CDIM].T
    cpca = [((c - mu) @ pca).astype(np.float32) for c in conts]

    ntr = int(len(conts) * 0.85)
    tr, te = list(range(ntr)), list(range(ntr, len(conts)))

    gen = MLP(CDIM, N_MELS).to(DEV)
    student = MLP(N_MELS * WM, CDIM).to(DEV)
    student_ant = MLP(N_MELS * WM, CDIM * (1 + KA)).to(DEV)
    opt = torch.optim.AdamW(list(gen.parameters()) + list(student.parameters())
                            + list(student_ant.parameters()), 3e-4)

    def sample(idxs, bs):
        mw, ct, cfut, mt = [], [], [], []
        for _ in range(bs):
            i = random.choice(idxs)
            T = cpca[i].shape[0]
            if T < WM + KA + 2:
                continue
            t = random.randint(WM - 1, T - KA - 1)
            mw.append(mels[i][:, t - WM + 1:t + 1].reshape(-1))
            ct.append(cpca[i][t])
            cfut.append(cpca[i][t:t + 1 + KA].reshape(-1))
            mt.append(mels[i][:, t])
        return (torch.tensor(np.array(mw), device=DEV),
                torch.tensor(np.array(ct), device=DEV),
                torch.tensor(np.array(cfut), device=DEV),
                torch.tensor(np.array(mt), device=DEV))

    for step in range(args.steps):
        mw, ct, cfut, mt = sample(tr, args.batch)
        gen_l = F.l1_loss(gen(ct), mt)
        stu_l = F.l1_loss(student(mw), ct)
        ant_l = F.l1_loss(student_ant(mw), cfut)
        loss = gen_l + stu_l + ant_l
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"step {step} gen {gen_l.item():.3f} student {stu_l.item():.3f} "
                  f"ant {ant_l.item():.3f}", flush=True)

    with torch.no_grad():
        mw, ct, cfut, mt = sample(te, 20000)
        var = mt.var().item()

        def melr2(c):
            return 1 - ((gen(c) - mt) ** 2).mean().item() / var

        def cmatch(c):
            return F.cosine_similarity(c, ct, dim=-1).mean().item()

        s = student(mw)
        sa = student_ant(mw)[:, :CDIM]
        print("\n=== causal content encoder: 生成質 (teacher=双方向 が上限) ===")
        print(f"  teacher(双方向ContentVec)     mel-R2 {melr2(ct):.3f}")
        print(f"  causal student(素)            mel-R2 {melr2(s):.3f}  content一致 {cmatch(s):.3f}")
        print(f"  causal student(anticipatory)  mel-R2 {melr2(sa):.3f}  content一致 {cmatch(sa):.3f}")
        gap0 = melr2(ct) - melr2(s)
        gap1 = melr2(ct) - melr2(sa)
        print(f"\n因果化コスト: 素 {gap0:+.3f} → anticipatory {gap1:+.3f} R2")
        if gap0 > 1e-4:
            print(f"anticipation が回収した割合: {(gap0 - gap1) / gap0 * 100:.0f}%")


if __name__ == "__main__":
    main()
