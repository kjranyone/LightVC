"""S1-5b: 条件付き CFM 生成器（E/P/S 条件 → Y-S1 latent 100fps・1-step）。

条件: content 768(ContentVec feat・50fps→100fps 因果) + lf0 + en + speaker emb
目標: ys1 latent z [T,32]（per-dim 正規化・data/ys1_latent/abi.pt）
損失: CFM `E‖v(z_t,t|c) − (z1−z0)‖²` のみ。AR(1) noise（vc_fm §5 ABI）。

    CUDA_VISIBLE_DEVICES=0 uv run python train_cfmys.py --steps 40000 --tag s5_cfm
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_mel import causal_mel

sys.path.insert(0, str(Path(__file__).parent))
from train_vc_g import CausalBlock

ROOT = Path(__file__).resolve().parent.parent
LAT = ROOT / "data/ys1_latent"
F0FIX = ROOT / "data/female_real_f0fix"
COND_FPS = 50.0
LAT_FPS = 100.0
F0_FPS = 44100 / 512
MEL_FPS = 44100 / 256.0
MEL_SCALE = 8.0


class CFMYS(nn.Module):
    def __init__(self, dim: int = 384, layers: int = 8, zd: int = 32,
                 cin: int = 770, emb: int = 192, td: int = 64,
                 spk_in: bool = False):
        super().__init__()
        self.dim, self.layers, self.spk_in = dim, layers, spk_in
        self.emb = emb
        self.inp = nn.Conv1d(zd + cin + (emb if spk_in else 0), dim, 3)
        self.blocks = nn.ModuleList(
            [CausalBlock(dim, 3, 2 ** (i // 2)) for i in range(layers)])
        self.temb = nn.Sequential(nn.Linear(1, td), nn.GELU(), nn.Linear(td, td))
        self.tfilm = nn.ModuleList([nn.Linear(td, dim * 2) for _ in range(layers)])
        self.sfilm = nn.ModuleList([nn.Linear(emb, dim * 2) for _ in range(layers)])
        self.out = nn.Linear(dim, zd)

    @property
    def ctx(self) -> int:
        return 2 + sum((3 - 1) * (2 ** (i // 2)) for i in range(self.layers))

    def forward(self, z_t, cond, t, s=None):
        if self.spk_in:
            if s is None:
                s = cond.new_zeros(cond.shape[0], self.emb)
            s_in = s[:, :, None].expand(-1, -1, cond.shape[-1])
            h = self.inp(F.pad(torch.cat([cond, s_in, z_t], 1), (2, 0)))
        else:
            h = self.inp(F.pad(torch.cat([cond, z_t], 1), (2, 0)))
        te = self.temb(t[:, None])
        for b, tf, sf in zip(self.blocks, self.tfilm, self.sfilm):
            h = b(h)
            g, beta = tf(te)[:, :, None].expand(-1, -1, h.shape[-1]).chunk(2, 1)
            h = h * (1 + g) + beta
            if s is not None:
                g2, b2 = sf(s)[:, :, None].expand(-1, -1, h.shape[-1]).chunk(2, 1)
                h = h * (1 + g2) + b2
        return self.out(h.transpose(1, 2)).transpose(1, 2)

    def arch(self) -> dict:
        return {"arch": "cfmys", "dim": self.dim if hasattr(self, "dim") else 384,
                "L": len(self.blocks), "ctx": self.ctx, "spk_in": self.spk_in}


def ar_noise(T: int, rho: float, gen, dev, B: int, zd: int = 32):
    w = torch.randn(B, zd, T, generator=gen, device=dev)
    n = torch.empty_like(w)
    n[:, :, 0] = w[:, :, 0]
    for i in range(1, T):
        n[:, :, i] = rho * n[:, :, i - 1] + math.sqrt(1 - rho ** 2) * w[:, :, i]
    return n


def sample_k(net, T: int, rho: float, gen, dev, cond, s_, K: int) -> torch.Tensor:
    z = ar_noise(T, rho, gen, dev, 1)
    with torch.no_grad():
        for k in range(K):
            t = torch.full((1,), k / K, device=dev)
            z = z + net(z, cond, t, s_) / K
    return z.clamp(-8, 8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--crop", type=int, default=200)     # 2s @100fps
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interp", action="store_true",
                    help="直線補間z_t入力の本来のCFM学習(z0入力の平均回帰退化を避ける)")
    ap.add_argument("--spk-in", action="store_true",
                    help="speaker embeddingを入力concatへ追加(cross-speaker話者条件強化)")
    ap.add_argument("--mel-in", action="store_true",
                    help="causal mel80(/8スケール)を条件へ追加(包絡情報の明示供給)")
    ap.add_argument("--mel-med", type=int, default=0,
                    help="mel条件の周波数方向メディアン窓(倍音リップル除去・0=無効)")
    ap.add_argument("--aug-p", type=float, default=0.0,
                    help="f0shift_latent増強を引く確率(mel/content=元・lf0/target=シフト後)")
    ap.add_argument("--aug-dir", default=str(ROOT / "data/f0shift_latent"))
    ap.add_argument("--sample-k", type=int, default=8,
                    help="interp時の評価サンプリングEulerステップ数")
    ap.add_argument("--f0fix", action="store_true",
                    help="female_real_featの破損f0を data/female_real_f0fix で置換")
    ap.add_argument("--aux-w", type=float, default=0.0,
                    help="decode領域mel補助損失の重さ(0=無効)")
    ap.add_argument("--aux-every", type=int, default=1)
    ap.add_argument("--aux-bs", type=int, default=2)
    ap.add_argument("--aux-crop", type=int, default=128)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)

    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    mu, sd = abi["mu"].to(dev), abi["sd"].to(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)

    # 条件 feat(50fps) × latent(100fps) ペア索引
    feats, lats = [], {}
    for cname in ("female_real_feat", "female_tts_feat"):
        root = ROOT / "data" / cname
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                feats += sorted(spk.glob("*.pt"))
    for p in LAT.rglob("*.pt"):
        if p.name != "abi.pt":
            lats[p.stem] = p
    pairs = [f for f in feats if f.stem in lats and "female_real" in str(lats[f.stem])]
    if a.f0fix:
        missing = [f for f in pairs
                   if not (F0FIX / f.parent.name / f.name).exists()]
        if missing:
            sys.exit(f"--f0fix: {len(missing)} pairs lack overlay "
                     f"(e.g. {missing[0]}). run f0fix_real.py to completion")
    rng.shuffle(pairs)
    spk_all = sorted({f.parent.name for f in pairs})
    held = set(spk_all[-24:])
    tr = [f for f in pairs if f.parent.name not in held]
    ev = [f for f in pairs if f.parent.name in held][:24]
    print(f"  pairs {len(pairs)} (train {len(tr)} / eval {len(ev)})", flush=True)

    net = CFMYS(dim=a.dim, cin=850 if a.mel_in else 770,
                spk_in=a.spk_in).to(dev)

    codec = fb = win = None
    if a.aux_w > 0:
        from causal_codec import CausalCodec
        ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev)
        codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
        codec.load_state_dict(ckc.get("ema") or ckc["net"])
        codec.eval()
        for p in codec.parameters():
            p.requires_grad_(False)
        import librosa as _lb
        fb = torch.from_numpy(_lb.filters.mel(sr=48000, n_fft=1024,
                                              n_mels=80)).float().to(dev)
        win = torch.hann_window(1024, device=dev)

    def logmel(wav: torch.Tensor) -> torch.Tensor:
        xp = F.pad(wav, (1024 - 480, 0))
        S = torch.stft(xp, 1024, hop_length=480, win_length=1024, window=win,
                       center=False, return_complex=True)
        m = torch.einsum("fm,bmt->bft", fb, S.abs().square())
        return torch.log(torch.clamp(m, min=1e-5))

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {sum(p.numel() for p in net.parameters())/1e6:.2f}M ctx {net.ctx}",
          flush=True)

    cache: dict = {}

    def get(f: Path):
        if f not in cache:
            cap = 1000 if a.mel_in else 2500
            if len(cache) > cap:
                cache.pop(next(iter(cache)))
            try:
                d = torch.load(f, map_location="cpu", weights_only=False)
                if a.f0fix:
                    ff = F0FIX / f.parent.name / f.name
                    if ff.exists():
                        d = {**d, "f0": torch.load(ff, map_location="cpu",
                                                   weights_only=False)["f0"]}
                mel_full = None
                if a.mel_in:
                    wv, _ = librosa.load(d["path"], sr=44100, mono=True)
                    w = torch.from_numpy(wv) * 32768.0
                    mel_full = causal_mel(w, n_fft=1024, hop=256, num_mels=80,
                                          sr=44100)[0].half()
                zd_ = torch.load(lats[f.stem], map_location="cpu",
                                 weights_only=False)
                aug = None
                if a.aug_p > 0:
                    ap_ = Path(a.aug_dir) / f.parent.name / (f.stem + ".pt")
                    if ap_.exists():
                        ad = torch.load(ap_, map_location="cpu",
                                        weights_only=False)
                        aug = (ad["z"].float(), float(ad["st"]))
                cache[f] = (d, zd_["z"].float(), mel_full, aug)
            except Exception:
                cache[f] = None
        return cache[f]

    def cond_of(d, T: int, mel_full=None, st: float = 0.0) -> torch.Tensor:
        c_full = d["content"].T.float()                     # [768, Tc]
        t_src = c_full.shape[-1]
        idx = ((torch.arange(T, dtype=torch.float64) + 1.0)
               * COND_FPS / LAT_FPS - 1.0).floor().clamp(0, t_src - 1).long()
        c = c_full[:, idx]
        f0 = d["f0"].float()
        if st:
            f0 = torch.where(f0 > 0, f0 * 2.0 ** (st / 12.0), f0)
        en = d["energy"].float()
        i_f = ((torch.arange(T, dtype=torch.float64) + 1.0)
               * F0_FPS / LAT_FPS - 1.0).floor().clamp(0, f0.shape[0] - 1).long()
        lf0 = torch.log(f0[i_f].clamp(min=50.0) / 200.0)
        enl = torch.log(en[i_f].clamp(min=1e-4))
        parts = [c, lf0[None], enl[None]]
        if mel_full is not None:
            m = mel_full.float()
            if a.mel_med:
                k = a.mel_med
                pad = k // 2
                mp = np.pad(m.numpy(), ((pad, pad), (0, 0)), mode="edge")
                m = torch.from_numpy(np.median(
                    np.lib.stride_tricks.sliding_window_view(mp, k, axis=0),
                    axis=-1).copy()).float()
            idx_m = ((torch.arange(T, dtype=torch.float64) + 1.0)
                     * MEL_FPS / LAT_FPS - 1.0).floor().clamp(0, m.shape[-1] - 1).long()
            parts.append((m[:, idx_m] / MEL_SCALE).clamp(-6.0, 6.0))
        return torch.cat(parts, 0)

    def evaluate() -> float:
        net.eval()
        ge = torch.Generator(device=dev).manual_seed(999)
        tot, n = 0.0, 0
        with torch.no_grad():
            for f in ev:
                it = get(f)
                if it is None:
                    continue
                d, z, mel_full, _aug = it
                T = z.shape[0]
                if T <= 300:
                    continue
                cond = cond_of(d, T, mel_full).to(dev)[None]
                s_ = spk_emb.get(d.get("speaker"))
                s_ = s_[None].to(dev) if s_ is not None else None
                if a.interp:
                    zh = sample_k(net, T, a.rho, ge, dev, cond, s_, a.sample_k)
                else:
                    z0 = ar_noise(T, a.rho, ge, dev, 1)
                    v = net(z0, cond, torch.ones(1, device=dev), s_)
                    zh = (z0 + v).clamp(-8, 8)
                z_rec = zh * sd[:, None] + mu[:, None]        # [1,32,T]
                zz = (z_rec[0].transpose(0, 1) - z.to(dev)).abs().mean()
                tot += float(zz)
                n += 1
        net.train()
        return tot / max(n, 1)

    best = 1e9
    step = 0
    t0 = time.time()
    while step < a.steps:
        zs, conds, spks = [], [], []
        while len(zs) < a.batch:
            f = rng.choice(tr)
            it = get(f)
            if it is None:
                continue
            d, z, mel_full, aug = it
            st = 0.0
            if aug is not None and rng.random() < a.aug_p:
                z, st = aug
            T = z.shape[0]
            if T <= a.crop + net.ctx + 4:
                continue
            s0 = rng.randrange(net.ctx, T - a.crop)
            zn = ((z - mu.cpu()) / sd.cpu())
            zs.append(zn[s0: s0 + a.crop].transpose(0, 1))     # [32, crop]
            conds.append(cond_of(d, T, mel_full, st)[:, s0: s0 + a.crop])
            spks.append(spk_emb.get(d.get("speaker"), torch.zeros(192)))
        step += 1
        z1b = torch.stack(zs).to(dev)
        cb = torch.stack(conds).to(dev)
        sb = torch.stack(spks).to(dev)
        z0 = ar_noise(a.crop, a.rho, gen, dev, len(zs))
        tt = torch.rand(len(zs), device=dev)
        z_in = (1 - tt[:, None, None]) * z0 + tt[:, None, None] * z1b \
            if a.interp else z0
        v = net(z_in, cb, tt, sb)
        loss = F.mse_loss(v, z1b - z0)
        aux = torch.zeros((), device=dev)
        if codec is not None and step % a.aux_every == 0:
            ib = min(a.aux_bs, z1b.shape[0])
            s0f = (a.crop - a.aux_crop) // 2
            with torch.no_grad():
                z_gt = (z1b[:ib] * sd[:, None] + mu[:, None])[:, :, s0f:s0f + a.aux_crop]
                w_gt = codec.decode(z_gt).squeeze(1)
            zr = z0[:ib]
            for t in (0.0, 0.5):
                zr = zr + net(zr, cb[:ib],
                              torch.full((ib,), t, device=dev), sb[:ib]) * 0.5
            z_pd = (zr.clamp(-8, 8) * sd[:, None] + mu[:, None])[:, :, s0f:s0f + a.aux_crop]
            w_pd = codec.decode(z_pd).squeeze(1)
            n = min(w_pd.shape[-1], w_gt.shape[-1])
            aux = F.l1_loss(logmel(w_pd[..., :n]), logmel(w_gt[..., :n]))
            loss = loss + a.aux_w * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            evl = evaluate()
            print(f"  step {step:6d}  cfm {float(loss):.4f}  aux {float(aux):.4f}"
                  f"  eval-L1 {evl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": net.state_dict(), "args": net.arch(),
                        "step": step, "cli": vars(a),
                        "abi": {"mu": mu.cpu(), "sd": sd.cpu()}},
                       out_dir / f"{a.tag}_last.pt")
            if evl < best:
                best = evl
                torch.save({"net": net.state_dict(), "args": net.arch(),
                            "step": step, "cli": vars(a),
                            "abi": {"mu": mu.cpu(), "sd": sd.cpu()}},
                           out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best eval-L1 {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
