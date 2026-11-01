"""diag_condgan1(results/diag_condgan1/prereg.yaml・current/cond_gan.md): 条件の空間の小さい GAN で男声 → 女声の条件を女声の分布へ写す診断。
データ = d273_data.py の書き出し(<cg>/fem・male・eval の .cond [T][31]・.front [T][224] = C1 の対数尤度 200 + 包絡 24)。
評価 = dev の男声 → 女声 30 組を G に通し Python の rvoc3hi で鳴らす(B0 = G なし・CG = G あり)→ eval_v1perp.py。
    uv run python d273_condgan.py --s <dev の親> --cg <dev の親>/condgan --out ../results/diag_condgan1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
R29 = list(range(0, 25)) + list(range(27, 31))
W = 200


def load(p: Path, k: str) -> tuple:
    c = np.fromfile(p / f"{k}.cond", "<f4").reshape(-1, 31)
    fr = np.fromfile(p / f"{k}.front", "<f4").reshape(-1, 224)
    n = min(len(c), len(fr))
    return c[:n], fr[:n, :200]


def summary(c: np.ndarray) -> np.ndarray:
    v = c[:, 26] > 0.5
    x = c[v][:, R29] if v.sum() > 10 else c[:, R29]
    lf = c[v, 25].mean() if v.sum() > 10 else 0.0
    return np.concatenate([x.mean(0), x.std(0), [lf]]).astype(np.float32)


class CConv(nn.Module):
    def __init__(self, cin: int, ch: int, cout: int, dils: tuple, causal: bool):
        super().__init__()
        self.causal = causal
        self.inp = nn.Conv1d(cin, ch, 1)
        self.layers = nn.ModuleList([nn.Conv1d(ch, ch, 5, dilation=d) for d in dils])
        self.out = nn.Conv1d(ch, cout, 1)
        self.dils = dils

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for d, l in zip(self.dils, self.layers):
            p = 4 * d
            hp = F.pad(h, (p, 0)) if self.causal else F.pad(h, (p // 2, p - p // 2))
            h = h + F.leaky_relu(l(hp), 0.2)
        return self.out(h)


class Disc(nn.Module):
    def __init__(self, cin: int, sdim: int, ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(cin, ch, 5, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2),
                                 nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2), nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.LeakyReLU(0.2))
        self.head = nn.Linear(ch, 1)
        self.proj = nn.Linear(sdim, ch)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.net(x).mean(-1)
        return self.head(h)[:, 0] + (self.proj(s) * h).sum(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--cg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--probe_steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    S, CG, O = Path(a.s), Path(a.cg), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    print(f"prereg: {O}/prereg.yaml", flush=True)
    dev = "cuda"
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    meta = json.loads((CG / "meta.json").read_text())
    fem: dict = {}
    for m in meta["fem"]:
        if (CG / "fem" / f"{m['key']}.cond").exists():
            c, lg = load(CG / "fem", m["key"])
            if len(c) > W + 10:
                fem.setdefault(m["spk"], []).append((c, lg))
    fem = {k: v for k, v in fem.items() if len(v) >= 2}
    male = []
    for m in meta["male"]:
        if m.get("src") != "vctk":
            continue
        if (CG / "male" / f"{m['key']}.cond").exists():
            c, lg = load(CG / "male", m["key"])
            if len(c) > W + 10:
                male.append((c, lg))
    fk = sorted(fem)
    print("female speakers", len(fk), "male clips", len(male), flush=True)
    summ = {k: [summary(c) for c, _ in fem[k]] for k in fk}
    allc = np.concatenate([c for k in fk for c, _ in fem[k]])
    mu, sd = allc.mean(0), allc.std(0) + 1e-4
    sall = np.stack([s for k in fk for s in summ[k]])
    smu, ssd = sall.mean(0), sall.std(0) + 1e-4
    nc = lambda c: (c - mu) / sd
    ns = lambda s: (s - smu) / ssd

    def crop(c, lg):
        o = random.randrange(0, len(c) - W)
        return c[o:o + W], lg[o:o + W]

    P = CConv(31, 128, 200, (1, 2, 4, 1, 2, 4), causal=False).to(dev)
    op = torch.optim.Adam(P.parameters(), 2e-4)
    for it in range(a.probe_steps):
        xb, yb = [], []
        for _ in range(a.bs):
            k = random.choice(fk); c, lg = random.choice(fem[k]); cc, ll = crop(c, lg)
            xb.append(nc(cc).T); yb.append(ll.T)
        x = torch.from_numpy(np.stack(xb)).float().to(dev); y = torch.from_numpy(np.stack(yb)).float().to(dev)
        loss = F.kl_div(F.log_softmax(P(x), 1), F.softmax(y, 1), reduction="batchmean") / W
        op.zero_grad(); loss.backward(); op.step()
        if (it + 1) % 500 == 0:
            print("probe", it + 1, round(float(loss), 4), flush=True)
    P.eval().requires_grad_(False)

    def prep_male(c, s):
        x = c.copy()
        v = x[:, 26] > 0.5
        if v.sum() > 5:
            x[v, 25] += s[-1] - x[v, 25].mean()
            for i, r in enumerate(range(27, 31)):
                x[v, r] = np.clip(x[v, r] + (s[25 + i] - x[v, r].mean()), 0, 1.5)
        return x

    G = CConv(31 + len(smu), 128, 29, (1, 2, 4, 8, 1, 2), causal=True).to(dev)
    D = Disc(30, len(smu)).to(dev)
    og = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.9))
    mu_t, sd_t = torch.from_numpy(mu).float().to(dev), torch.from_numpy(sd).float().to(dev)
    r29 = torch.tensor(R29, device=dev)

    def apply_g(xn: torch.Tensor, sn: torch.Tensor) -> tuple:
        x = xn * sd_t[None, :, None] + mu_t[None, :, None]
        inp = torch.cat([xn, sn[:, :, None].expand(-1, -1, xn.shape[-1])], 1)
        dlt = G(inp)
        wgt = torch.sigmoid(4 * (x[:, 0:1] - 1.5))
        vmask = (x[:, 26:27] > 0.5).float()
        m = torch.cat([wgt.expand(-1, 25, -1), (wgt * vmask).expand(-1, 4, -1)], 1)
        y = x.clone()
        y[:, r29] = x[:, r29] + dlt * m
        return y, dlt

    def dfeat(y: torch.Tensor) -> torch.Tensor:
        yn = (y - mu_t[None, :, None]) / sd_t[None, :, None]
        return torch.cat([yn[:, r29], yn[:, 26:27]], 1)

    log = open(O / "train.jsonl", "w")
    for it in range(a.steps):
        rx, rs, mx, ms, ml = [], [], [], [], []
        for _ in range(a.bs):
            k = random.choice(fk); j = random.randrange(len(fem[k])); c, _ = fem[k][j]
            rx.append(nc(crop(c, c)[0]).T); rs.append(ns(summ[k][1 - j]))
            c2, lg2 = random.choice(male); k2 = random.choice(fk); s2 = random.choice(summ[k2])
            cc, ll = crop(c2, lg2)
            mx.append(nc(prep_male(cc, s2)).T); ms.append(ns(s2)); ml.append(ll.T)
        rx = torch.from_numpy(np.stack(rx)).float().to(dev); rs = torch.from_numpy(np.stack(rs)).float().to(dev)
        mx = torch.from_numpy(np.stack(mx)).float().to(dev); ms = torch.from_numpy(np.stack(ms)).float().to(dev); ml = torch.from_numpy(np.stack(ml)).float().to(dev)
        y, dlt = apply_g(mx, ms)
        real = rx * sd_t[None, :, None] + mu_t[None, :, None]
        dl = ((D(dfeat(real), rs) - 1) ** 2).mean() + (D(dfeat(y.detach()), ms) ** 2).mean()
        od.zero_grad(); dl.backward(); od.step()
        adv = ((D(dfeat(y), ms) - 1) ** 2).mean()
        yn = (y - mu_t[None, :, None]) / sd_t[None, :, None]
        kl = F.kl_div(F.log_softmax(P(yn), 1), F.softmax(ml, 1), reduction="batchmean") / W
        gl = adv + 1.0 * kl + 0.1 * (dlt ** 2).mean()
        og.zero_grad(); gl.backward(); og.step()
        if (it + 1) % 250 == 0:
            r = {"step": it + 1, "dl": round(float(dl), 4), "adv": round(float(adv), 4), "kl": round(float(kl), 4), "d2": round(float((dlt ** 2).mean()), 4)}
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    torch.save({"G": G.state_dict(), "mu": mu, "sd": sd, "smu": smu, "ssd": ssd}, O / "last.pt")
    G.eval()
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    s_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(s_["ema"]); gen.eval().requires_grad_(False)

    @torch.no_grad()
    def render(c: np.ndarray) -> np.ndarray:
        cond = torch.from_numpy(c.T.astype(np.float32))[None].to(dev)
        f0 = np.where(c[:, 26] > 0.5, 200.0 * np.exp(c[:, 25]), 0.0).astype(np.float32)
        yy = gen(cond, TR.excitation(torch.from_numpy(f0)[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (yy / (np.abs(yy).max() + 1e-9) * 0.9).astype(np.float32)

    ev = O / "eval"
    ev.mkdir(exist_ok=True)
    for m in meta["eval"]:
        k, ti = m["key"], m["ti"]
        c, _ = load(CG / "eval", k)
        ref = np.fromfile(S / "dev/d271" / f"ref_{ti:03d}.f32", "<f4").reshape(-1, 31)
        s = summary(ref)
        with torch.no_grad():
            y, _ = apply_g(torch.from_numpy(nc(c).T[None]).float().to(dev), torch.from_numpy(ns(s)[None]).float().to(dev))
        y = y[0].cpu().numpy().T
        sf.write(ev / f"{k}_B0.wav", render(c), N.SR)
        sf.write(ev / f"{k}_CG.wav", render(y), N.SR)
        y.astype("<f4").tofile(ev / f"{k}_CG.cond")
    print("eval rendered", len(meta["eval"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
