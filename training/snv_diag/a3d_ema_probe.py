"""a3d_resid_probe.py の較正: 同じ 60 声・同じ探り(フレームだけ / 時間の文脈つき)で、A2 の x が変換に成功した演奏の信号(SPARC の EMA 12・声源 3)から声を当てる率。
時間の文脈で読める声のうち、変換を壊さない量(話し方・リズム)の基準。"""
from __future__ import annotations
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from train_snv_a2 import FEAT, ROOT, VAL_VOICES, playing
from train_snv_a3 import load_corpus

DEV = "cuda"
N_A, N_POOL, TR_UTT = 60, 200, 7
rng = np.random.default_rng(0)
torch.manual_seed(0)
voices = load_corpus()
train = sorted(voices)[:-VAL_VOICES]
pool = sorted(rng.choice(train, N_POOL, replace=False).tolist())
A = pool[:N_A]
man = {v["voice"]: v["files"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]}
feat = {}
for n in A:
    us = []
    for f in man[n]:
        mp, sp, wp = FEAT / f"mel/e1/{n}/{f['t']}.npy", FEAT / f"sparc/e1/{n}/{f['t']}.npz", FEAT / f"wav44/e1/{n}/{f['t']}.npy"
        if mp.exists() and sp.exists() and wp.exists():
            us.append(playing(sp, np.load(mp, mmap_mode="r").shape[1]))
    assert len(us) == len(voices[n])
    feat[n] = [torch.from_numpy(u).to(DEV) for u in us]


def make(dim: int, kind: str) -> nn.Module:
    if kind == "frame":
        return nn.Sequential(nn.Conv1d(dim, 512, 1), nn.ReLU(), nn.Conv1d(512, 512, 1), nn.ReLU(), nn.Conv1d(512, N_A, 1))
    ls = [nn.Conv1d(dim, 256, 5, padding=2), nn.ReLU()]
    for d in (2, 4, 8, 16):
        ls += [nn.Conv1d(256, 256, 3, padding=d, dilation=d), nn.ReLU()]
    return nn.Sequential(*ls, nn.Conv1d(256, N_A, 1))


def run(cols: slice, kind: str, steps: int = 3000) -> dict:
    data = {n: [u[:, cols] for u in feat[n]] for n in A}
    allx = torch.cat([torch.cat(data[n]) for n in A])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    r = np.random.default_rng(1)
    net = make(allx.shape[1], kind).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(steps):
        xs, ys = [], []
        for _ in range(32):
            i = int(r.integers(N_A))
            x = data[A[i]][int(r.integers(TR_UTT))]
            s = int(r.integers(max(1, len(x) - 200)))
            xs.append(x[s:s + 200])
            ys.append(i)
        L = min(len(x) for x in xs)
        x = torch.stack([((v[:L] - mu) / sd).T for v in xs])
        lg = net(x)
        loss = F.cross_entropy(lg, torch.tensor(ys, device=DEV)[:, None].expand(-1, lg.shape[-1]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    utt, fr = [], []
    with torch.no_grad():
        for i, a in enumerate(A):
            for x in data[a][TR_UTT:]:
                lp = net(((x - mu) / sd).T[None])[0].log_softmax(0)
                utt.append(int(lp.mean(1).argmax()) == i)
                fr.append(float((lp.argmax(0) == i).float().mean()))
    return {"utt": round(float(np.mean(utt)), 3), "frame": round(float(np.mean(fr)), 3)}


print("chance", round(1 / N_A, 3), flush=True)
for name, cols in (("ema12", slice(0, 12)), ("src3", slice(12, 15)), ("ema12+src3", slice(0, 15))):
    for kind in ("frame", "ctx"):
        print(name, kind, run(cols, kind), flush=True)
