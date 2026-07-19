"""E1 overfit-one gate for NSF-LTV v1.5 (current/vocoder.md §6 E1).

Trainability check: can LtvFrameNet fit ONE fixed segment by SGD through the
LTV renderer? Modes:
  env    — phase (a): lambda_env only (L1 to hpv-paw oracle envelopes + d).
  spec   — phase (b): MRSTFT + mel only (the real objective, no teacher).
  anneal — env warm-up cosine-annealed to 0 over 30% of steps + spec throughout.

Gate thresholds are CALIBRATED against the E0 oracle render of the SAME
segment (perfect-prediction floor) — no guessed thresholds. Direct comparison
row: NSF-HN3 overfit died at mel-L1 0.26 / mrs 2.44 / amp half / contrast 0.82.

Low-level citizenship (vocoder.md §3.5): --lc weights the mel loss per frame
by inverse loudness so quiet segments (breath) are first-class.

Usage: cd training && uv run python e1_overfit_ltv.py --mode anneal
Outputs: results/e1_overfit/ (gt/oracle/net wavs) + results/e1_overfit_ltv.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from e0_oracle_ltv import (FRAME_MS, active_frames, hpv_calibration,
                           hpv_paw_envelopes, oracle_d, sharp_contrast,
                           subsample_env)
from ltv_frame_net import LtvFrameNet, smooth_reg
from ltv_render import HOP, SR, HarmonicSource, MinPhaseFIR, ltv_ola, pitch_sync_mod

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_MELS = 128
NB = 1025
EPS = 1e-8

_MELW = {}


def mel_of(y: torch.Tensor) -> torch.Tensor:
    key = str(y.device)
    if key not in _MELW:
        fb = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=N_MELS)
        _MELW[key] = (torch.tensor(fb, dtype=torch.float32, device=y.device),
                      torch.hann_window(2048, device=y.device))
    fb, win = _MELW[key]
    s = torch.stft(y, 2048, HOP, window=win, return_complex=True).abs()
    return torch.log(fb @ s + 1e-5)


def mrstft(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for nfft, hop in [(512, 128), (1024, 256), (2048, 512)]:
        w = torch.hann_window(nfft, device=a.device)
        A = torch.stft(a, nfft, hop, window=w, return_complex=True).abs() + 1e-7
        B = torch.stft(b, nfft, hop, window=w, return_complex=True).abs() + 1e-7
        loss = loss + F.l1_loss(torch.log(A), torch.log(B)) + F.l1_loss(A, B)
    return loss


def mel_l1(a: torch.Tensor, b: torch.Tensor,
           lc: bool = False) -> torch.Tensor:
    ma, mb = mel_of(a), mel_of(b)
    if not lc:
        return F.l1_loss(ma, mb)
    w = 1.0 / (mb.exp().mean(1, keepdim=True).sqrt() + 1e-2)
    w = w / w.mean()
    return ((ma - mb).abs() * w).mean()


def metrics(y: np.ndarray, gt: np.ndarray, f0: np.ndarray,
            act: np.ndarray) -> dict:
    n = min(len(y), len(gt))
    y, gt = y[:n], gt[:n]
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
    gtt = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
    c_y, c_g = sharp_contrast(y, act), sharp_contrast(gt, act)
    return {
        "mel_l1": round(float(mel_l1(yt, gtt)), 4),
        "mrs": round(float(mrstft(yt, gtt)), 3),
        "amp_ratio": round(float(np.sqrt((y ** 2).mean() / ((gt ** 2).mean() + EPS))), 3),
        "contrast_ratio": round(c_y / (c_g + EPS), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/female_real_feat/209c94d37412922a")
    ap.add_argument("--pick", default="00038900")
    ap.add_argument("--seg", type=int, default=128)
    ap.add_argument("--off", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mode", default="anneal", choices=["env", "spec", "anneal"])
    ap.add_argument("--lc", action="store_true", default=True)
    ap.add_argument("--no-lc", dest="lc", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results/e1_overfit"))
    ap.add_argument("--json", default=str(ROOT / "results/e1_overfit_ltv.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import glob
    f = [p for p in sorted(glob.glob(str(Path(args.feat) / "*.pt")))
         if args.pick in p][0]
    d0 = torch.load(f, weights_only=False)
    f0_all = d0["f0"].float().numpy().astype(np.float64)
    energy = d0["energy"].float()
    y_all, _ = librosa.load(d0["path"], sr=SR, mono=True)
    tmel = len(f0_all)
    y_all = y_all[:tmel * HOP]
    off = min(args.off, max(0, tmel - args.seg))
    seg = min(args.seg, tmel - off)

    hv_all, hn_all = hpv_paw_envelopes(y_all, f0_all)
    ap_w = pw.d4c(y_all.astype(np.float64), f0_all,
                  np.arange(tmel, dtype=np.float64) * FRAME_MS / 1000.0, SR,
                  fft_size=4096)
    c_h, c_n = hpv_calibration()
    dh = c_h + 0.5 * np.log(np.maximum(f0_all, 1.0) / 200.0)
    hv_o = subsample_env(hv_all - dh[:, None], NB)[off:off + seg]
    hn_o = subsample_env(hn_all - c_n - 0.6, NB)[off:off + seg]
    d_o = oracle_d(ap_w, f0_all)[off:off + seg]

    f0s = f0_all[off:off + seg]
    gt = y_all[off * HOP:(off + seg) * HOP].astype(np.float32)
    act = active_frames(gt)
    gt_t = torch.tensor(gt, dtype=torch.float32, device=DEV).unsqueeze(0)
    f0_t = torch.tensor(f0s, dtype=torch.float32, device=DEV).unsqueeze(0)
    hv_ot = torch.tensor(hv_o, dtype=torch.float32, device=DEV).unsqueeze(0)
    hn_ot = torch.tensor(hn_o, dtype=torch.float32, device=DEV).unsqueeze(0)
    d_ot = torch.tensor(d_o, dtype=torch.float32, device=DEV).unsqueeze(0)

    harm = HarmonicSource(causal=True, jitter=0.003).to(DEV)
    fir_v = MinPhaseFIR(NB, 1024).to(DEV)
    fir_n = MinPhaseFIR(NB, 256).to(DEV)
    phi0 = torch.tensor(math.pi, device=DEV)
    torch.manual_seed(0)
    with torch.no_grad():
        e_h, phase = harm(f0_t)
        e_h, phase = e_h[:, :seg * HOP], phase[:, :seg * HOP]

    def render(hv, hn, dd, noise=None):
        if noise is None:
            noise = torch.randn_like(e_h)
        m = pitch_sync_mod(phase, dd, phi0, HOP, causal=True, p=4)
        return (ltv_ola(e_h, fir_v(hv), HOP) + ltv_ola(noise * m, fir_n(hn), HOP))

    with torch.no_grad():
        torch.manual_seed(1)
        y_or = render(hv_ot, hn_ot, d_ot)[0].cpu().numpy()
        g = np.sqrt((gt ** 2).mean() / ((y_or ** 2).mean() + EPS))
        y_or *= g
    floor = metrics(y_or, gt, f0s, act)
    sf.write(out / "gt.wav", gt, SR)
    sf.write(out / "oracle.wav", np.clip(y_or, -1, 1).astype(np.float32), SR)
    print(f"E0 oracle floor (same segment): {floor}", flush=True)

    mel_c = mel_of(gt_t)
    if mel_c.shape[-1] != seg:
        mel_c = F.interpolate(mel_c, size=seg, mode="linear", align_corners=False)
    logf0 = (torch.log(f0_t.clamp(min=1.0)) / 7.0).unsqueeze(1)
    eng = (torch.log(energy[off:off + seg].clamp(min=1e-4)) * 0.2)
    eng = eng.view(1, 1, -1).to(DEV)
    cond = torch.cat([mel_c, logf0, eng], 1)

    net = LtvFrameNet(cond_dim=N_MELS + 2, nb=NB,
                      hv_bias=hv_ot.mean(dim=(0, 1)).cpu(),
                      hn_bias=hn_ot.mean(dim=(0, 1)).cpu()).to(DEV)
    print(f"LtvFrameNet params {sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"mode={args.mode} lc={args.lc} dev={DEV}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.8, 0.99))

    t0 = time.time()
    for it in range(args.steps + 1):
        o = net(cond)
        if args.mode == "env":
            lam = 1.0
        elif args.mode == "spec":
            lam = 0.0
        else:
            prog = min(1.0, it / (0.3 * args.steps))
            lam = 0.5 * (1.0 + math.cos(math.pi * prog))
        loss = 0.0
        log = {}
        if lam > 0.0:
            l_env = (F.l1_loss(o["h_v"], hv_ot) + F.l1_loss(o["h_n"], hn_ot)
                     + F.l1_loss(o["d"], d_ot))
            loss = loss + lam * l_env
            log["env"] = float(l_env)
        if args.mode != "env":
            y_hat = render(o["h_v"], o["h_n"], o["d"])
            ml = mel_l1(y_hat, gt_t, lc=args.lc)
            ms = mrstft(y_hat, gt_t)
            loss = loss + 45.0 * ml + 2.0 * ms
            log["mel"] = float(ml)
            log["mrs"] = float(ms)
        loss = loss + 1e-3 * (smooth_reg(o["h_v"].transpose(1, 2))
                              + smooth_reg(o["h_n"].transpose(1, 2)))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        if it % 100 == 0:
            msg = " ".join(f"{k} {v:.4f}" for k, v in log.items())
            print(f"it {it:5d} lam {lam:.2f} {msg} ({time.time()-t0:.0f}s)", flush=True)

    with torch.no_grad():
        o = net(cond)
        torch.manual_seed(1)
        y_net = render(o["h_v"], o["h_n"], o["d"])[0].cpu().numpy()
        g = np.sqrt((gt ** 2).mean() / ((y_net ** 2).mean() + EPS))
        y_net *= g
    m_net = metrics(y_net, gt, f0s, act)
    sf.write(out / f"net_{args.mode}.wav", np.clip(y_net, -1, 1).astype(np.float32), SR)

    gate = {
        "mel_l1": m_net["mel_l1"] <= max(0.1, floor["mel_l1"] * 1.15),
        "mrs": m_net["mrs"] <= max(1.0, floor["mrs"] * 1.15),
        "amp_ratio": 0.95 <= m_net["amp_ratio"] <= 1.05,
        "contrast": m_net["contrast_ratio"] >= 0.95 * floor["contrast_ratio"],
    }
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "mode": args.mode,
        "steps": args.steps, "lc": args.lc, "segment": f"{args.pick} off{off} seg{seg}",
        "floor_oracle": floor, "net": m_net, "gate": gate,
        "pass": all(gate.values()),
        "hn3_reference": {"mel_l1": 0.26, "mrs": 2.44, "amp_ratio": 0.5,
                          "contrast_ratio": 0.82},
    }
    prev = json.loads(Path(args.json).read_text()) if Path(args.json).exists() else []
    prev.append(report)
    Path(args.json).write_text(json.dumps(prev, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print(f"-> {args.json}", flush=True)


if __name__ == "__main__":
    main()
