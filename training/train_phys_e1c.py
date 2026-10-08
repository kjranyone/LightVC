"""phys_e1c: 参照エンコーダ E(ゼロショットの物理パラメータ推定)。事前登録 results/phys_e1c/prereg.yaml。

phys_e1b からの変更(e2 で E が自然な話者差を読まないと判明したため):
  (1) 変換は広い格子 GRID_W(40Hz–23kHz)で計算し、E の入力と損失は内側 150–12000Hz だけ(格子端の貼り付けの痕跡を見せない)
  (2) 実話者どうしの統計一致の項: 話者 A(女声または JA TTS 男声)→ 女声 B について、E(B)−E(A) で A の統計用フレームを変換し、
      B の統計用フレームと平均包絡の形・フレーム間の標準偏差を合わせる(非並行・自然データどうしの差を読まないと下がらない)
損失 = 合成の往復(e1b と同じ)+ 実話者の統計一致。

    CUDA_VISIBLE_DEVICES=0 uv run python train_phys_e1c.py --steps 6000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
import physvc as PV
from physvc import N_PHI, RHO, TAU, V, W1, W3

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/artic_feat"
OUT = ROOT / "results/phys_e1c"
SRCS = ("real_female", "tts_base", "tts_emotional_live")
T_U = 800
R_MAX = 1200
S_MAX = 2400
GRIDW = PV.GRID_W
BW = PV.BAND_W
LEV_DB = 35.0
TH_LO = torch.tensor([-0.25, -0.08, -0.08, -0.08, -2.0, -0.3])
TH_HI = -TH_LO


def males() -> dict:
    out = {}
    for d in sorted((FEAT / "tts_male_ja").iterdir()):
        fs = sorted(d.glob("*.npz"))
        if len(fs) >= 5:
            out[f"tts_male_ja/{d.name}"] = fs
    return out


def speakers() -> tuple[dict, list, list]:
    from train_d1 import build_index
    spk = {}
    for s in SRCS:
        for d in sorted((FEAT / s).iterdir()):
            fs = sorted(d.glob("*.npz"))
            if len(fs) >= 5:
                spk[f"{s}/{d.name}"] = fs
    _, _, held = build_index(0)
    real = sorted(k for k in spk if k.startswith("real_female/"))
    tts = sorted(k for k in spk if not k.startswith("real_female/"))
    hs = {f"real_female/{h}" for h in held if f"real_female/{h}" in spk}
    rest = [k for k in real if k not in hs]
    hs |= set(rest[:: max(1, len(rest) // (100 - len(hs)))][: 100 - len(hs)])
    ht = set(tts[:: max(1, len(tts) // 20)][:20])
    ev = sorted(hs | ht)
    tr = [k for k in spk if k not in set(ev)]
    return spk, tr, ev


def speech(f: Path) -> np.ndarray:
    z = np.load(f)
    lev = z["lev"].astype(np.float32)
    return z["lar"][lev >= lev.max() - LEV_DB].astype(np.float32)


def crop_u(f: Path, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(f)
    lar, lev = z["lar"].astype(np.float32), z["lev"].astype(np.float32)
    m = lev >= lev.max() - LEV_DB
    if len(lar) > T_U:
        s = rng.randrange(0, len(lar) - T_U)
        lar, m = lar[s:s + T_U], m[s:s + T_U]
    out = np.zeros((T_U, 24), np.float32)
    mk = np.zeros(T_U, np.float32)
    out[:len(lar)], mk[:len(lar)] = lar, m
    return out, mk


def refs(fs: list, rng: random.Random, cap: int = R_MAX) -> np.ndarray:
    acc, n = [], 0
    for f in fs:
        s = speech(f)
        acc.append(s)
        n += len(s)
        if n >= cap:
            break
    r = np.concatenate(acc)[:cap]
    out = np.zeros((cap, 24), np.float32)
    out[:len(r)] = r
    return out, len(r)


class DS(torch.utils.data.Dataset):
    def __init__(self, spk: dict, keys: list, n: int, seed: int | None = None, srcs: dict | None = None):
        self.spk, self.keys, self.n, self.seed = spk, keys, n, seed
        self.srcs = srcs or {}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = random.Random(None if self.seed is None else self.seed * 100003 + i)
        k = self.keys[rng.randrange(len(self.keys))]
        fs = list(self.spk[k])
        rng.shuffle(fs)
        u, mu = crop_u(fs[0], rng)
        half = (len(fs) - 1) // 2
        ra, na = refs(fs[1:1 + half], rng)
        rb, nb = refs(fs[1 + half:], rng)
        g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
        th = TH_LO + (TH_HI - TH_LO) * torch.rand(N_PHI, generator=g)
        pool = {**{k: self.spk[k] for k in self.keys}, **self.srcs}
        ka = rng.choice(list(self.srcs)) if self.srcs and rng.random() < 0.3 else self.keys[rng.randrange(len(self.keys))]
        kb = self.keys[rng.randrange(len(self.keys))]
        fa, fb = list(pool[ka]), list(self.spk[kb])
        rng.shuffle(fa), rng.shuffle(fb)
        ha, hb = len(fa) // 2, len(fb) // 2
        xa, nxa = refs(fa[:ha], rng)
        sa, nsa = refs(fa[ha:], rng, S_MAX)
        xb, nxb = refs(fb[:hb], rng)
        sb, nsb = refs(fb[hb:], rng, S_MAX)
        nat = tuple(torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in (xa, nxa, sa, nsa, xb, nxb, sb, nsb))
        return (torch.from_numpy(u), torch.from_numpy(mu), torch.from_numpy(ra), na, torch.from_numpy(rb), nb, th) + nat


class Enc(nn.Module):
    def __init__(self, nf: int = int(BW.sum()), h: int = 256):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(nf, h), nn.LeakyReLU(0.1), nn.Linear(h, h), nn.LeakyReLU(0.1))
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.LeakyReLU(0.1), nn.Linear(h, N_PHI))

    def forward(self, E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.frame(E / 10.0)
        w = mask[..., None]
        n = w.sum(1).clamp(min=1)
        mu = (h * w).sum(1) / n
        sd = (((h - mu[:, None]) ** 2 * w).sum(1) / n).clamp(min=1e-6).sqrt()
        return self.head(torch.cat([mu, sd], -1))


def masked_mean(E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (E * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


def env(lar: torch.Tensor) -> torch.Tensor:
    return PV.env_grid(lar, GRIDW)


def einp(E: torch.Tensor) -> torch.Tensor:
    return PV.shape(E)[..., torch.as_tensor(BW, device=E.device)]


def lsd(A: torch.Tensor, B: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    band = torch.as_tensor(PV.band_of(A.shape[-1]), device=A.device)
    d = (PV.shape(A) - PV.shape(B))[..., band]
    per = d.pow(2).mean(-1).clamp(min=1e-8).sqrt()
    return (per * mask).sum(1) / mask.sum(1).clamp(min=1)


def mask_n(n: torch.Tensor, cap: int, dev) -> torch.Tensor:
    return (torch.arange(cap, device=dev)[None] < n.to(dev)[:, None]).float()


def nat_loss(b, enc: Enc | None, dev, zero: bool = False) -> torch.Tensor:
    xa, nxa, sa, nsa, xb, nxb, sb, nsb = (x.to(dev) for x in b[7:])
    mxa, msa, mxb, msb = mask_n(nxa, R_MAX, dev), mask_n(nsa, S_MAX, dev), mask_n(nxb, R_MAX, dev), mask_n(nsb, S_MAX, dev)
    Ea, Eb = env(sa), env(sb)
    if zero or enc is None:
        dphi = torch.zeros(sa.shape[0], N_PHI, device=dev)
    else:
        dphi = enc(einp(env(xb)), mxb) - enc(einp(env(xa)), mxa)
    Et = PV.transform_env(Ea, masked_mean(Ea, msa), dphi)
    band = torch.as_tensor(BW, device=dev)
    mt, mb_ = masked_mean(PV.shape(Et), msa)[:, band], masked_mean(PV.shape(Eb), msb)[:, band]
    st = (masked_mean((PV.shape(Et) - masked_mean(PV.shape(Et), msa)[:, None]) ** 2, msa)).clamp(min=1e-6).sqrt()[:, band]
    sb_ = (masked_mean((PV.shape(Eb) - masked_mean(PV.shape(Eb), msb)[:, None]) ** 2, msb)).clamp(min=1e-6).sqrt()[:, band]
    return (mt - mb_).pow(2).mean(-1).sqrt() + (st - sb_).pow(2).mean(-1).sqrt()


def step_batch(b, enc: Enc | None, dev, oracle: bool = False):
    u, mu, ra, na, rb, nb, th = (x.to(dev) if torch.is_tensor(x) else x for x in b[:7])
    ma = (torch.arange(R_MAX, device=dev)[None] < na.to(dev)[:, None]).float()
    mb = (torch.arange(R_MAX, device=dev)[None] < nb.to(dev)[:, None]).float()
    Eu, Ea, Eb = env(u), env(ra), env(rb)
    m_a = masked_mean(Ea, ma)
    Eu2 = PV.transform_env(Eu, m_a, th)
    Eb2 = PV.transform_env(Eb, m_a, th)
    m_b2 = masked_mean(Eb2, mb)
    if oracle:
        dphi = -th
    elif enc is None:
        dphi = torch.zeros_like(th)
    else:
        dphi = enc(einp(Ea), ma) - enc(einp(Eb2), mb)
    Ehat = PV.transform_env(Eu2, m_b2, dphi)
    return lsd(Ehat, Eu, mu), dphi, th


def evaluate(loader, enc, dev) -> dict:
    enc.eval()
    acc = {"none": [], "oracle": [], "E": [], "nat_zero": [], "nat_E": []}
    P, Tt = [], []
    with torch.no_grad():
        for b in loader:
            for k, kw in (("none", {"enc": None}), ("oracle", {"enc": None, "oracle": True}), ("E", {"enc": enc})):
                l, dphi, th = step_batch(b, kw.pop("enc"), dev, **kw)
                acc[k].append(l.cpu())
                if k == "E":
                    P.append(dphi.cpu()), Tt.append(th.cpu())
            acc["nat_zero"].append(nat_loss(b, None, dev, zero=True).cpu())
            acc["nat_E"].append(nat_loss(b, enc, dev).cpu())
    enc.train()
    out = {k: round(float(torch.cat(v).mean()), 4) for k, v in acc.items()}
    P, Tt = torch.cat(P), torch.cat(Tt)
    for nm, j in (("v", V), ("w1", W1), ("w3", W3), ("tau", TAU), ("rho", RHO)):
        y, yh = -Tt[:, j], P[:, j]
        out[f"r2_{nm}"] = round(float(1 - ((yh - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("prereg:", OUT / "prereg.yaml", flush=True)
    spk, tr, ev = speakers()
    print(f"speakers train {len(tr)} eval {len(ev)} (utts {sum(len(spk[k]) for k in tr)} / {sum(len(spk[k]) for k in ev)})", flush=True)
    import soundfile, librosa, csv
    row = next(r for r in csv.DictReader(open(ROOT / "data/kansei_vc/manifests/canonical_utterances.tsv"), delimiter="\t")
               if r["source_type"] == "real_female")
    wav = ROOT / row["wav_path"].replace("../", "", 1)
    x, s0 = soundfile.read(str(wav), dtype="float32")
    x = librosa.resample(x, orig_sr=s0, target_sr=D.SR).astype(np.float64) if s0 != D.SR else x.astype(np.float64)
    z = np.load(FEAT / "real_female" / wav.parent.name / (wav.stem + ".npz"))
    dd = float(np.abs(D.lar_frames(x, 24, la=480) - z["lar"].astype(np.float64)).max())
    print(f"eval_ref_check: max|ΔLAR| 再計算 vs artic_feat = {dd:.4f}", flush=True)
    assert dd <= 1e-2
    ml = males()
    print("male sources (JA TTS):", len(ml), flush=True)
    dl = torch.utils.data.DataLoader(DS(spk, tr, a.steps * a.bs, srcs=ml), batch_size=a.bs, num_workers=10, persistent_workers=True)
    el = torch.utils.data.DataLoader(DS(spk, ev, 512, seed=7), batch_size=64, num_workers=6)
    enc = Enc().to(dev)
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    r0 = evaluate(el, enc, dev)
    print("floor check (step 0):", json.dumps(r0), flush=True)
    log = {"step0": r0, "evals": []}
    t0 = time.time()
    for it, b in enumerate(dl, 1):
        l, _, _ = step_batch(b, enc, dev)
        ln = nat_loss(b, enc, dev)
        loss = l.mean() + ln.mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step()
        sch.step()
        if it % 250 == 0 or it == a.steps:
            r = evaluate(el, enc, dev)
            r.update(step=it, train_loss=round(float(loss), 4), min=round((time.time() - t0) / 60, 1))
            log["evals"].append(r)
            print(json.dumps(r), flush=True)
            if it == 1000 and r["E"] >= r["none"]:
                print("TRIPWIRE: step1000 で補正なしを下回らない → 停止", flush=True)
                break
            torch.save({"enc": enc.state_dict(), "step": it}, OUT / "last.pt")
        if it >= a.steps:
            break
    (OUT / "log.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
