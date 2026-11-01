"""A2 の x の注意の解析: 頭ごとのエントロピー・時刻と鍵の相互情報量・読んだ参照の EMA と今の EMA の相関(時刻を混ぜた対照つき)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import torch
from snv_a0_measure import DEV, ROOT
from train_snv_a2 import FEAT, G, playing

A1 = ROOT / "results/snv_a1"
d = ROOT / "results/diag_snv_a2_x"
ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
g = G(ck["mode"]).to(DEV).eval()
g.load_state_dict(ck["g"])
nz = np.load(d / "norm.npz"); mu, sd = nz["mu"], nz["sd"]
drows = json.loads((A1 / "diag/meta.json").read_text())
mi, hcond, hmarg, tv, corr_self, corr_shuf = [], [], [], [], [], []
rng = np.random.default_rng(0)
with torch.no_grad():
    for r in drows:
        v = r["target"]
        rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
        rp = (playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[1]) - mu) / sd
        S = g.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy(rp)[None].to(DEV), torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV))
        n_mel = np.load(FEAT / f"mel/a1diag/D0_{v}_{r['g']}.npy").shape[1]
        p = (playing(FEAT / f"sparc/a1diag/D0_{v}_{r['g']}.npz", n_mel) - mu) / sd
        _, att = g(torch.from_numpy(p)[None].to(DEV), S, return_attn=True)
        a = att[0].double().cpu().numpy()
        ent = lambda q: -(np.clip(q, 1e-12, None) * np.log(np.clip(q, 1e-12, None))).sum(-1)
        hc = ent(a).mean(-1)
        hm = ent(a.mean(1))
        hcond.append(hc); hmarg.append(hm); mi.append(hm - hc)
        ctx = np.einsum("htr,hrd->htd", a, S["v"][0].double().cpu().numpy())
        dev = ctx - ctx.mean(1, keepdims=True)
        tv.append((dev ** 2).sum() / (ctx ** 2).sum())
        pa = np.einsum("htr,rd->htd", a, rp).mean(0)
        ema = slice(0, 12)
        cs = [np.corrcoef(pa[:, j], p[:, j])[0, 1] for j in range(12)]
        perm = rng.permutation(n_mel)
        cf = [np.corrcoef(pa[perm, j], p[:, j])[0, 1] for j in range(12)]
        corr_self.append(np.nanmean(cs)); corr_shuf.append(np.nanmean(cf))
hc, hm, m = np.array(hcond), np.array(hmarg), np.array(mi)
print("per-head H(att|t) mean", hc.mean(0).round(2), "H(mean att)", hm.mean(0).round(2), "MI(t;k)", m.mean(0).round(2), "nats")
print("ctx time-varying energy fraction", float(np.mean(tv)))
print("EMA corr(attended ref playing, query playing)", float(np.mean(corr_self)), "shuffled", float(np.mean(corr_shuf)))
