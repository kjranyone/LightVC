"""段 I3 の判別器(paper_snv §5 M5)が実データの声の符号を検出できるかの探り(学習の外・CPU): 声が演奏に残っている重み(snv_i2_m3e@10k)の型からのずれ r = μ_q − μ_w[τ] で、
train_snv_i.Critic と同じ形の判別器を学び、検証の発話でフレームごとの交差エントロピー(偶然 = log 声の数)と、フレームの対数確率を発話で足した 1 位の率を測る。
発話 = 学習の声の共有の内容の発話を声ごとに n 本(切り出し 800・冒頭 WARM_SELF を外す・構造 = 同じ声の別の普通の発話 3 本)。声ごとに 1 本を検証に回す。
    cd training && uv run --no-sync python snv_diag/i3_critic_probe.py results/snv_i2_m3e/step10k.pt [voices=120] [per=5] [steps=3000] [out=results/snv_i3/critic_probe.json]
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_snv_i as I
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = I.ROOT


def main() -> int:
    ck = [x for x in sys.argv[1:] if "=" not in x][0]
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    nv, per, steps = int(opt.get("voices", 120)), int(opt.get("per", 5)), int(opt.get("steps", 3000))
    torch.set_num_threads(8)
    torch.manual_seed(0)
    st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
    E = P11.EncS().eval()
    E.load_state_dict(st["Es"])
    tpl, logsig = st["tpl"].float(), st["logsig"].float()
    sig2 = (2 * logsig).exp()
    names = sorted(json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"], key=lambda v: v["voice"])
    tr, _ = T.split([v["voice"] for v in names])
    rng = random.Random(0)
    voices = sorted(rng.sample(tr, nv))
    c = I.Corpus(voices)

    @torch.no_grad()
    def resid(n: str, u: dict) -> torch.Tensor | None:
        rs = [x for x in c.u[n]["norm"] if x is not u][:I.REF_N]
        ry = torch.from_numpy(np.concatenate([x["y"][:, :I.REF_LEN] for x in rs], 1))[None]
        rv = torch.from_numpy(np.concatenate([x["v"][:I.REF_LEN] for x in rs]))[None]
        s = E.structure(P11.xin(ry, rv), torch.ones(rv.shape, dtype=torch.bool))
        n_all = u["y"].shape[1]
        a0 = rng.randint(0, max(0, n_all - T.CROP))
        y, v = u["y"][:, a0:a0 + T.CROP], u["v"][a0:a0 + T.CROP]
        mu = E(P11.xin(torch.from_numpy(y)[None], torch.from_numpy(v)[None]), s)[0][0][:, T.WARM_SELF:]
        tn = mu.shape[1] // I.POOL
        if tn < 8:
            return None
        cid = u["cid"]
        L = int(I_tlen[cid])
        mp = mu[:, :tn * I.POOL].reshape(mu.shape[0], tn, I.POOL).mean(-1)
        tp = tpl[gcid[cid], :L]
        lp = -0.5 * (((mp.T[:, None, :] - tp[None]) ** 2) / sig2).sum(-1)
        path = I.mas_free(lp[None], torch.tensor([tn]), torch.tensor([L]))[0]
        tau = path.repeat_interleave(I.POOL)
        return mu[:, :len(tau)] - tp[tau].T

    full = I.Corpus(tr)
    gkeys = {k: i for i, k in enumerate(full.cid)}
    gcid = {c.cid[k]: gkeys[k] for k in c.cid if k in gkeys}
    I_tlen = {c.cid[k]: int(full.tlen[gkeys[k]]) for k in c.cid if k in gkeys}
    data = []
    for si, n in enumerate(voices):
        us = [u for kind in ("norm", "nv") for u in c.u[n][kind] if u["cid"] >= 0 and u["cid"] in gcid]
        rng.shuffle(us)
        got = []
        for u in us:
            r = resid(n, u)
            if r is not None:
                got.append(r)
            if len(got) >= per:
                break
        for j, r in enumerate(got):
            data.append((si, r, j == 0))
    trn = [(s, r) for s, r, v in data if not v]
    val = [(s, r) for s, r, v in data if v]
    Q = I.Critic(len(voices))
    oq = torch.optim.Adam(Q.parameters(), lr=1e-3)
    for step in range(1, steps + 1):
        bs = rng.sample(trn, min(16, len(trn)))
        L = min(r.shape[1] for _, r in bs)
        x = torch.stack([r[:, :L] for _, r in bs])
        lab = torch.tensor([s for s, _ in bs])[:, None].expand(-1, L)
        loss = F_.cross_entropy(Q(x), lab)
        oq.zero_grad()
        loss.backward()
        oq.step()
    with torch.no_grad():
        ces, top = [], []
        for s, r in val:
            lg = Q(r[None])[0]
            ces.append(float(F_.cross_entropy(lg.T, torch.full((lg.shape[1],), s))))
            top.append(int(F_.log_softmax(lg, 0).sum(1).argmax()) == s)
    res = {"ckpt": ck, "voices": nv, "train_utts": len(trn), "val_utts": len(val), "steps": steps, "chance_ce": round(math.log(nv), 4),
           "frame_ce_val": round(float(np.mean(ces)), 4), "utt_top1_val": round(float(np.mean(top)), 4), "utt_chance": round(1 / nv, 4), "train_ce_last": round(float(loss), 4)}
    print(json.dumps(res), flush=True)
    (ROOT / opt.get("out", "results/snv_i3/critic_probe.json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
