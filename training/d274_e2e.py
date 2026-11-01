"""段 1 の基準 e2e_ppg(current/e2e_ppg.md・results/diag_e2e1/prereg.yaml): 内容(C1 の事後)・log f0・有声・目標の要約 → 出力部の条件(c0・包絡・周期性)の因果な変換器 E を、実女声の自己再構成(要約 = 同じ話者の別のクリップ)で学習する。
評価: 保留の実女声 30 人の自己再構成(SR)と同じクリップの写し合成(CP)・dev の男声 → 女声 60 組(E2E)を Python の rvoc3hi で鳴らす → eval_v1perp.py。
    uv run python d274_e2e.py --s <dev の親> --e2e <dev の親>/e2e --out ../results/diag_e2e1
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import torch.nn as nn
from d273_condgan import R29, CConv, Disc, summary


class DiscFM(Disc):
    def forward_fm(self, x: torch.Tensor, s: torch.Tensor) -> tuple:
        fs = []
        h = x
        for m in self.net:
            h = m(h)
            if isinstance(m, nn.LeakyReLU):
                fs.append(h)
        g = h.mean(-1)
        return self.head(g)[:, 0] + (self.proj(s) * g).sum(-1), fs

ROOT = Path(__file__).resolve().parent.parent
W = 200


def load(p: Path, k: str) -> tuple:
    c = np.fromfile(p / f"{k}.cond", "<f4").reshape(-1, 31)
    lg = np.fromfile(p / f"{k}.front", "<f4").reshape(-1, 224)[:, :200]
    n = min(len(c), len(lg))
    return c[:n], lg[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--e2e", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--adv", type=float, default=1.0)
    ap.add_argument("--fm", type=float, default=0.0, help="判別器の中間特徴の一致の重み(0 = なし)")
    ap.add_argument("--extra", nargs="*", default=[], help="追加の女声の書き出しの置き場(meta.json の fem と <dir>/fem/*.cond・.front)")
    a = ap.parse_args()
    S, E2, O = Path(a.s), Path(a.e2e), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    print(f"prereg: {O}/prereg.yaml", flush=True)
    dev = "cuda"
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    meta = json.loads((E2 / "meta.json").read_text())
    data: dict = {}
    for root in [E2] + [Path(x) for x in a.extra]:
        mm = json.loads((root / "meta.json").read_text())
        for m in mm["fem"]:
            if (root / "fem" / f"{m['key']}.cond").exists():
                c, lg = load(root / "fem", m["key"])
                if len(c) > W + 10:
                    data.setdefault(m["spk"], []).append((c, lg))
    spk = sorted(k for k, v in data.items() if len(v) >= 2)
    random.Random(274).shuffle(spk)
    held, train = spk[:30], spk[30:]
    print("speakers train", len(train), "held", len(held), flush=True)
    allc = np.concatenate([c for k in train for c, _ in data[k]])
    mu, sd = allc.mean(0), allc.std(0) + 1e-4
    summ = {k: [summary(c) for c, _ in data[k]] for k in spk}
    sall = np.stack([s for k in train for s in summ[k]])
    smu, ssd = sall.mean(0), sall.std(0) + 1e-4
    mu_t, sd_t = torch.from_numpy(mu).float().to(dev), torch.from_numpy(sd).float().to(dev)
    r29 = torch.tensor(R29, device=dev)

    def feats(c: np.ndarray, lg: np.ndarray) -> np.ndarray:
        lp = lg / a.tau
        lp = lp - lp.max(1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
        lf = c[:, 25:26]
        vo = c[:, 26:27]
        return np.concatenate([lp, lf / 0.5, vo], 1).astype(np.float32)

    E = CConv(200 + 2 + len(smu), 256, 29, (1, 2, 4, 8, 16, 1, 2, 4), causal=True).to(dev)
    D = DiscFM(30, len(smu)).to(dev)
    oe = torch.optim.Adam(E.parameters(), 2e-4, betas=(0.5, 0.9))
    od = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.9))

    def run_e(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return E(torch.cat([x, s[:, :, None].expand(-1, -1, x.shape[-1])], 1))

    def batch(keys: list) -> tuple:
        xb, sb, yb = [], [], []
        for _ in range(a.bs):
            k = random.choice(keys); j = random.randrange(len(data[k])); c, lg = data[k][j]
            o = random.randrange(0, len(c) - W)
            jj = random.choice([q for q in range(len(data[k])) if q != j])
            xb.append(feats(c[o:o + W], lg[o:o + W]).T); sb.append((summ[k][jj] - smu) / ssd); yb.append(((c[o:o + W] - mu) / sd).T)
        t = lambda z: torch.from_numpy(np.stack(z)).float().to(dev)
        return t(xb), t(sb), t(yb)

    log = open(O / "train.jsonl", "w")
    for it in range(a.steps):
        x, s, y = batch(train)
        pred = run_e(x, s)
        tgt = y[:, r29]
        vo = y[:, 26:27]
        fake = torch.cat([pred, vo], 1)
        real = torch.cat([tgt, vo], 1)
        dl = ((D(real, s) - 1) ** 2).mean() + (D(fake.detach(), s) ** 2).mean()
        od.zero_grad(); dl.backward(); od.step()
        l1 = (pred - tgt).abs().mean()
        lo, ff = D.forward_fm(fake, s)
        adv = ((lo - 1) ** 2).mean()
        loss = 10 * l1 + a.adv * adv
        if a.fm > 0:
            with torch.no_grad():
                _, fr = D.forward_fm(real, s)
            loss = loss + a.fm * sum(F.l1_loss(u, v) for u, v in zip(ff, fr))
        oe.zero_grad(); loss.backward(); oe.step()
        if (it + 1) % 500 == 0:
            with torch.no_grad():
                xh, sh, yh = batch(held)
                hl1 = float((run_e(xh, sh) - yh[:, r29]).abs().mean())
            r = {"step": it + 1, "l1": round(float(l1), 4), "held_l1": round(hl1, 4), "adv": round(float(adv), 4), "dl": round(float(dl), 4)}
            print(json.dumps(r), flush=True); log.write(json.dumps(r) + "\n"); log.flush()
    torch.save({"E": E.state_dict(), "mu": mu, "sd": sd, "smu": smu, "ssd": ssd, "tau": a.tau}, O / "last.pt")
    E.eval()
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

    @torch.no_grad()
    def convert(c: np.ndarray, lg: np.ndarray, s: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(feats(c, lg).T[None]).float().to(dev)
        sn = torch.from_numpy(((s - smu) / ssd)[None]).float().to(dev)
        p = run_e(x, sn)[0] * sd_t[r29][:, None] + mu_t[r29][:, None]
        y = c.copy()
        y[:, R29] = p.cpu().numpy().T
        v = y[:, 26] > 0.5
        y[~v, 27:31] = 0.0
        y[:, 27:31] = np.clip(y[:, 27:31], 0, 1.5)
        return y

    ev = O / "eval"
    ev.mkdir(exist_ok=True)
    for i, k in enumerate(held):
        (cA, lgA), (cB, _) = data[k][0], data[k][1]
        sf.write(ev / f"h{i:04d}_t{i:03d}_SR.wav", render(convert(cA, lgA, summary(cB))), N.SR)
        sf.write(ev / f"h{i:04d}_t{i:03d}_CP.wav", render(cA), N.SR)
    ed = E2 / "eval_dev"
    nm = 0
    for f in sorted(ed.glob("p00*.cond")):
        k = f.stem
        ti = int(k.split("_t")[1][:3])
        if not (ed / f"ref_{ti:03d}.cond").exists():
            continue
        c, lg = load(ed, k)
        ref = np.fromfile(ed / f"ref_{ti:03d}.cond", "<f4").reshape(-1, 31)
        sf.write(ev / f"{k}_E2E.wav", render(convert(c, lg, summary(ref))), N.SR)
        nm += 1
    print("eval rendered held", len(held), "mf", nm, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
