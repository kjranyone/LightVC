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
from train_m1 import DEV

MOE = ["cute_high", "intimate_close", "young_bright"]
CDIM = 24
ADIM = 48
L = 128
CACHE = "/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/c76a325d-9c57-4dc0-bd41-4abd61a25a89/scratchpad/formant_cache.pt"


class EArt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 128, 5, 1, 2), nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, 5, 2, 2), nn.LeakyReLU(0.1),
            nn.Conv1d(128, 128, 5, 1, 2), nn.LeakyReLU(0.1))
        self.proj = nn.Linear(256, ADIM)
        self.sup = nn.Linear(ADIM, 4)

    def forward(self, ftraj):
        h = self.conv(ftraj)
        s = torch.cat([h.mean(-1), h.std(-1) + 1e-5], -1)
        code = self.proj(s)
        return code, self.sup(code)


class Rearti(nn.Module):
    def __init__(self, h: int = 256) -> None:
        super().__init__()
        self.pre = nn.Conv1d(CDIM, h, 5, 1, 2)
        self.c1 = nn.Conv1d(h, h, 5, 1, 2)
        self.c2 = nn.Conv1d(h, h, 5, 1, 2)
        self.film1 = nn.Linear(ADIM, 2 * h)
        self.film2 = nn.Linear(ADIM, 2 * h)
        self.out = nn.Conv1d(h, 3, 1)

    def _adain(self, x, film, s_art):
        g, b = film(s_art).chunk(2, -1)
        return F.instance_norm(x) * (1 + g).unsqueeze(-1) + b.unsqueeze(-1)

    def forward(self, cont, s_art):
        x = self.pre(cont)
        x = self._adain(F.leaky_relu(self.c1(x), 0.1), self.film1, s_art)
        x = self._adain(F.leaky_relu(self.c2(x), 0.1), self.film2, s_art)
        return self.out(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    raw = torch.load(CACHE, weights_only=False)
    print(f"loaded {len(raw)} utts from cache")

    allc = np.concatenate([r["content"].astype(np.float32) for r in raw], 0)
    mu = allc.mean(0)
    _, _, Vt = np.linalg.svd(allc[:: max(1, len(allc) // 15000)] - mu, full_matrices=False)
    pca = Vt[:CDIM].T.astype(np.float32)
    allf = np.concatenate([r["formant"] for r in raw], 0)
    fmu, fsd = allf.mean(0), allf.std(0) + 1e-6

    segs = []
    for r in raw:
        ft = ((r["formant"] - fmu) / fsd).astype(np.float32)
        T = len(ft)
        cont = r["content"].astype(np.float32)
        c = F.interpolate(torch.from_numpy(cont).t().unsqueeze(0), size=T,
                          mode="linear", align_corners=False).squeeze(0).numpy()
        cp = ((c.T - mu) @ pca).T.astype(np.float32)
        df = np.diff(ft, axis=0, prepend=ft[:1])
        ftraj = np.concatenate([ft, df], 1).T.astype(np.float32)
        for s in range(0, T - L, L):
            fn_s = ft[s:s + L].T
            sup = np.array([np.median(fn_s[0]),
                            np.percentile(fn_s[1], 90) - np.percentile(fn_s[1], 10),
                            fn_s[0].std() * fn_s[1].std(),
                            np.median(np.abs(df[s:s + L, 1]))], np.float32)
            segs.append((cp[:, s:s + L], fn_s.astype(np.float32),
                         ftraj[:, s:s + L], sup, r["style"]))
    random.seed(0)
    random.shuffle(segs)
    print(f"{len(segs)} segments")
    ntr = int(len(segs) * 0.85)
    tr, te = segs[:ntr], segs[ntr:]

    def stack(pool, key):
        return torch.tensor(np.stack([p[key] for p in pool]), device=DEV)
    Ctr, Ftr, Xtr, Str = stack(tr, 0), stack(tr, 1), stack(tr, 2), stack(tr, 3)

    ea, rr = EArt().to(DEV), Rearti().to(DEV)
    opt = torch.optim.AdamW(list(ea.parameters()) + list(rr.parameters()), 3e-4)
    for step in range(args.steps):
        i = torch.randint(0, len(tr), (args.batch,), device=DEV)
        code, ps = ea(Xtr[i])
        pred = rr(Ctr[i], code)
        loss = F.l1_loss(pred, Ftr[i]) + 0.3 * F.l1_loss(ps, Str[i])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)

    with torch.no_grad():
        def code_for(styles):
            xs = torch.tensor(np.stack([p[2] for p in tr if p[4] in styles]), device=DEV)
            return ea(xs)[0].mean(0, keepdim=True)
        s_neu = code_for(["neutral"])
        s_moe = code_for(MOE)
        f1 = {"neu": [], "moe": []}
        f2r = {"neu": [], "moe": []}
        for p in te:
            if p[4] != "neutral":
                continue
            c_t = torch.tensor(p[0], device=DEV).unsqueeze(0)
            for tag, sc in [("neu", s_neu), ("moe", s_moe)]:
                pr = rr(c_t, sc)[0].cpu().numpy()
                f1[tag].append(np.median(pr[0]))
                f2r[tag].append(np.percentile(pr[1], 90) - np.percentile(pr[1], 10))
    print("\n=== 再構音clone: neutral content に s_art 差替え (正規化formant) ===")
    print(f"  F1 median:  neutral-style {np.mean(f1['neu']):+.3f}  → moe-style {np.mean(f1['moe']):+.3f}  (期待 ↑)")
    print(f"  F2_range:   neutral-style {np.mean(f2r['neu']):.3f}  → moe-style {np.mean(f2r['moe']):.3f}  (期待 ↓)")
    d1 = np.mean(f1['moe']) - np.mean(f1['neu'])
    d2 = np.mean(f2r['moe']) - np.mean(f2r['neu'])
    print(f"\n  ΔF1 {d1:+.3f} (>0で萌え方向) | ΔF2_range {d2:+.3f} (<0で萌え方向)")
    print("  両方が萌え方向 = 構音スタイルcloneが機構として成立")


if __name__ == "__main__":
    main()
