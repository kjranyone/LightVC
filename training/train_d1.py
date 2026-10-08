"""D1 G0/G1 trainer: 潜在AR-CFM(current/d1_latentar.md rev2・prereg: results/d1_g0/)。

2026-09-23修正: probeのGT参照を生スケールでdecode(bug7)・lossログとbest選択は
区間平均(旧=1ミニバッチlossでbatch運に支配)。best選択は学習lossであり品質ではない。

    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python train_d1.py --tag d1_g0 --overfit 10 --steps 8000
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from d1_model import D1AR, Z_DIM, COND_DIM, COND_DIM_M80, shift_right, sample_frame_ar
from train_cfmys import LAT, F0FIX, COND_FPS, LAT_FPS, F0_FPS

ROOT = Path(__file__).resolve().parent.parent


def build_index(overfit: int = 0, held: bool = False):
    feats, lats = [], {}
    for spk in sorted((ROOT / "data/female_real_feat").iterdir()):
        if spk.is_dir():
            feats += sorted(spk.glob("*.pt"))
    for p in (LAT / "female_real").rglob("*.pt"):
        lats[p.stem] = p
    pairs = [f for f in feats
             if f.stem in lats
             and (F0FIX / f.parent.name / f.name).exists()]
    spk_all = sorted({f.parent.name for f in pairs})
    rng = random.Random(0)
    rng.shuffle(pairs)
    if held:
        hset = set(spk_all[-24:])
        pairs = [f for f in pairs if f.parent.name not in hset]
    if overfit:
        keep = set(spk_all[:overfit])
        pairs = [f for f in pairs if f.parent.name in keep]
    return pairs, lats, spk_all[-24:]


def cond_of(d, T: int, mel_full=None) -> torch.Tensor:
    c_full = d["content"].T.float()
    idx = ((torch.arange(T, dtype=torch.float64) + 1.0)
           * COND_FPS / LAT_FPS - 1.0).floor().clamp(0, c_full.shape[-1] - 1).long()
    f0 = d["f0"].float()
    i_f = ((torch.arange(T, dtype=torch.float64) + 1.0)
           * F0_FPS / LAT_FPS - 1.0).floor().clamp(0, f0.shape[0] - 1).long()
    lf0 = torch.log(f0[i_f].clamp(min=50.0) / 200.0)
    en = d["energy"].float()
    enl = torch.log(en[i_f].clamp(min=1e-4))
    parts = [c_full[:, idx], lf0[None], enl[None]]
    if mel_full is not None:
        m = mel_full.float()
        idx_m = ((torch.arange(T, dtype=torch.float64) + 1.0)
                 * (44100.0 / 256) / LAT_FPS - 1.0).floor().clamp(
                     0, m.shape[-1] - 1).long()
        parts.append((m[:, idx_m] / 8.0).clamp(-6.0, 6.0))
    return torch.cat(parts, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=200)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--noise-p", type=float, default=0.3)
    ap.add_argument("--noise-sig-lo", type=float, default=0.1)
    ap.add_argument("--noise-sig-hi", type=float, default=0.5)
    ap.add_argument("--noise-rho", type=float, default=0.0,
                    help="履歴ノイズの時間相関AR(1)係数(0=白色・(A)腕=0.25)")
    ap.add_argument("--z0-rho", type=float, default=0.0,
                    help="z0の時間相関AR(1)係数(prereg G1-FAILフォールバック=0.25)")
    ap.add_argument("--mel80", action="store_true")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="G1: フルデータ(held末尾24話者を除外)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    pairs, lats, _ = build_index(a.overfit, held=a.full)
    rng.shuffle(pairs)
    print(f"  pairs {len(pairs)} (overfit={a.overfit or 'full'})", flush=True)
    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    mu, sd = abi["mu"].to(dev), abi["sd"].to(dev)

    net = D1AR(dim=a.dim, layers=a.layers,
               cond_dim=COND_DIM_M80 if a.mel80 else COND_DIM,
               no_history=a.no_history).to(dev)
    print(f"  params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                              pct_start=0.05)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}

    def get(f: Path):
        if f not in cache:
            if len(cache) > (400 if not a.mel80 else 150):
                cache.pop(next(iter(cache)))
            try:
                d = torch.load(f, map_location="cpu", weights_only=False)
                ff = F0FIX / f.parent.name / f.name
                d = {**d, "f0": torch.load(ff, map_location="cpu",
                                           weights_only=False)["f0"]}
                z = torch.load(lats[f.stem], map_location="cpu",
                               weights_only=False)["z"].float().T  # [32,T]
                mel_full = None
                if a.mel80:
                    import librosa
                    from causal_mel import causal_mel
                    wv, _ = librosa.load(d["path"], sr=44100, mono=True)
                    mel_full = causal_mel(torch.from_numpy(wv) * 32768.0,
                                          n_fft=1024, hop=256, num_mels=80,
                                          sr=44100)[0].half()
                cache[f] = (d, z, mel_full)
            except Exception:  # noqa: BLE001
                cache[f] = None
        return cache[f]

    from causal_codec import CausalCodec
    from eval_d4b_gates import band_metrics
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()

    def render_probe_utt(p: Path, seed: int = 0):
        d, z, mel_full = get(p)
        T = z.shape[1]
        cond = cond_of(d, T, mel_full)[None].to(dev)
        with torch.no_grad():
            zh = sample_frame_ar(net, cond, K=8, seed=seed, z0_rho=a.z0_rho)
            zg = (zh * sd[:, None] + mu[:, None]).clamp(-8, 8)
            y = codec.decode(zg[0][None].to(dev))[0, 0].cpu().numpy()
            yg = codec.decode(z[None].to(dev))[0, 0].cpu().numpy()
        return y, yg

    step = 0
    best = 1e9
    run_sum, run_n = 0.0, 0
    t0 = time.time()
    if a.resume:
        ck = out_dir / f"{a.tag}_last.pt"
        if ck.exists():
            st = torch.load(ck, map_location=dev, weights_only=False)
            net.load_state_dict(st["net"])
            step = st["step"]
            for _ in range(step):
                sch.step()
            print(f"  resumed at {step}", flush=True)
    while step < a.steps:
        zs, cs = [], []
        while len(zs) < a.batch:
            f = rng.choice(pairs)
            it = get(f)
            if it is None:
                continue
            d, z, mel_full = it
            T = z.shape[1]
            if T <= a.crop + 8:
                continue
            cond_full = cond_of(d, T, mel_full)
            s0 = rng.randrange(4, T - a.crop)
            zs.append(((z[:, s0:s0 + a.crop] - mu[:, None].cpu())
                       / sd[:, None].cpu()).clamp(-8, 8))
            cs.append(cond_full[:, s0:s0 + a.crop])
        step += 1
        z1 = torch.stack(zs).to(dev)                       # [B,32,T] normalized
        cb = torch.stack(cs).to(dev)
        B = a.batch
        e = torch.randn(B, Z_DIM, a.crop, device=dev)
        if a.z0_rho > 0:
            z0 = e.clone()
            for j in range(1, a.crop):
                z0[:, :, j] = (a.z0_rho * z0[:, :, j - 1]
                               + math.sqrt(1 - a.z0_rho ** 2) * e[:, :, j])
        else:
            z0 = e
        tt = torch.rand(B, device=dev)
        z_interp = (1 - tt[:, None, None]) * z0 + tt[:, None, None] * z1
        z_hist = shift_right(z1)
        if a.noise_p > 0:
            noisy = (torch.rand(B, device=dev) < a.noise_p)
            sig = (torch.empty(B, Z_DIM, 1, device=dev)
                   .uniform_(a.noise_sig_lo, a.noise_sig_hi)
                   * z1.std(dim=2, keepdim=True).clamp(min=0.05))
            if a.noise_rho > 0:
                e = torch.randn_like(z_hist)
                n = e.clone()
                for j in range(1, z_hist.shape[2]):
                    n[:, :, j] = (a.noise_rho * n[:, :, j - 1]
                                  + math.sqrt(1 - a.noise_rho ** 2) * e[:, :, j])
            else:
                n = torch.randn_like(z_hist)
            z_hist = z_hist + torch.where(
                noisy[:, None, None], sig * n, torch.zeros_like(z_hist))
        v = net(z_hist, cb, z_interp, tt)
        loss = F.mse_loss(v, z1 - z0)
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        run_sum += float(loss.detach())
        run_n += 1
        if step % a.every == 0 or step == a.steps:
            lmean = run_sum / max(run_n, 1)
            run_sum, run_n = 0.0, 0
            p = pairs[0]
            y, yg = render_probe_utt(p)
            soundfile.write(out_dir / "probe.wav", np.clip(y, -1, 1), 48000)
            ratio = band_metrics(y)["hi_mid"] / max(band_metrics(yg)["hi_mid"], 1e-4)
            print(f"  step {step:6d}  loss {lmean:.4f}  "
                  f"hi_mid比 {ratio:.2f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": net.state_dict(), "step": step, "cli": vars(a),
                        "abi": {"mu": mu.cpu(), "sd": sd.cpu()}},
                       out_dir / f"{a.tag}_last.pt")
            if lmean < best:
                best = lmean
                torch.save({"net": net.state_dict(), "step": step, "cli": vars(a),
                        "abi": {"mu": mu.cpu(), "sd": sd.cpu()}},
                           out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best loss {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
