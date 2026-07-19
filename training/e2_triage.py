"""E2 ear-driven triage set (Step 0 — no fixes, attribution only).

Same held-out utterances, same noise seed, same gain matching across ALL arms:
  gt        — original
  oracle    — E0 render from cached oracle envelopes+d (no net) = physics ceiling
  net       — e2 checkpoint prediction
  oraclehop — oracle at filter hop 256 (sub=2)
  nethop    — net prediction at filter hop 256

Outputs results/e2_triage/{utt}_{role}.wav + results/e2_triage.json.
Judge = human ear only. One AB pair per round thereafter.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from e0_oracle_ltv import upsample_frames
from e1_overfit_ltv import mel_of
from ltv_frame_net import LtvFrameNet
from ltv_render import HOP, SR, HarmonicSource, MinPhaseFIR, ltv_ola, pitch_sync_mod

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/e2_ltv_cache/af1ad5575a3fa383"
OUT = ROOT / "results/e2_triage"
SEED = 1
NB = 1025
EPS = 1e-8


def render(f0: np.ndarray, hv: np.ndarray, hn: np.ndarray, d: np.ndarray,
           gt: np.ndarray, sub: int = 1) -> np.ndarray:
    T = len(f0)
    if sub > 1:
        vm = upsample_frames((f0 > 1.0).astype(np.float64), sub) > 0.5
        f0h = f0.copy()
        if (f0 > 1.0).any():
            f0h[f0h <= 1.0] = np.interp(np.where(f0 <= 1.0)[0],
                                        np.where(f0 > 1.0)[0], f0[f0 > 1.0])
        f0u = upsample_frames(f0h, sub)
        f0u[~vm] = 0.0
        f0, hv, hn, d = f0u, upsample_frames(hv, sub), upsample_frames(hn, sub), \
            upsample_frames(d, sub)
        T = T * sub
    hop = HOP // sub
    harm = HarmonicSource(hop=hop, causal=True, jitter=0.003)
    fv = MinPhaseFIR(NB, 1024)
    fn = MinPhaseFIR(NB, 256)
    phi0 = torch.tensor(math.pi)
    f0t = torch.tensor(f0, dtype=torch.float32).unsqueeze(0)
    hvt = torch.tensor(hv, dtype=torch.float32).unsqueeze(0)
    hnt = torch.tensor(hn, dtype=torch.float32).unsqueeze(0)
    dt = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        torch.manual_seed(SEED)
        e_h, ph = harm(f0t)
        n = T * hop
        e_h, ph = e_h[:, :n], ph[:, :n]
        torch.manual_seed(SEED)
        noise = torch.randn_like(e_h)
        m = pitch_sync_mod(ph, dt, phi0, hop, causal=True, p=4)
        y = (ltv_ola(e_h, fv(hvt), hop) + ltv_ola(noise * m, fn(hnt), hop))[0].numpy()
    g = np.sqrt((gt ** 2).mean() / ((y ** 2).mean() + EPS))
    y = y * g
    peak = np.abs(y).max()
    if peak > 0.95:
        y = y * (0.95 / peak)
    return y.astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/"
                "2ed9836e-e8de-4c00-b91a-ee2ae45093f0/scratchpad/e2_snap.pt")
    ck = torch.load(snap, map_location="cpu", weights_only=False)
    net = LtvFrameNet(cond_dim=130, nb=NB)
    net.load_state_dict(ck["net"])
    net.eval()
    report = {"date": time.strftime("%Y-%m-%d %H:%M"), "ckpt_step": ck["step"],
              "seed": SEED, "utts": {}}
    for npz in sorted(CACHE.glob("*.npz"))[-3:]:
        z = np.load(npz)
        T = min(len(z["f0"]), int(8.0 * SR) // HOP)
        w = ROOT / "female-dataset/af1ad5575a3fa383" / (npz.stem + ".wav")
        x, _ = librosa.load(str(w), sr=SR, mono=True)
        gt = x[:T * HOP].astype(np.float32)
        f0 = z["f0"][:T].astype(np.float64)
        hv_o = z["hv"][:T].astype(np.float32)
        hn_o = z["hn"][:T].astype(np.float32)
        d_o = z["d"][:T].astype(np.float64)

        xt = torch.tensor(gt).unsqueeze(0)
        f0t = torch.tensor(f0, dtype=torch.float32).unsqueeze(0)
        mel = mel_of(xt)
        if mel.shape[-1] != T:
            mel = F.interpolate(mel, size=T, mode="linear", align_corners=False)
        logf0 = (torch.log(f0t.clamp(min=1.0)) / 7.0).unsqueeze(1)
        rms = xt.reshape(1, T, HOP).pow(2).mean(-1).sqrt()
        eng = (torch.log(rms.clamp(min=1e-4)) * 0.2).unsqueeze(1)
        with torch.no_grad():
            o = net(torch.cat([mel, logf0, eng], 1))
        hv_n = o["h_v"][0].numpy()
        hn_n = o["h_n"][0].numpy()
        d_n = o["d"][0].numpy().astype(np.float64)

        uid = npz.stem
        sf.write(OUT / f"{uid}_gt.wav", gt, SR)
        sf.write(OUT / f"{uid}_oracle.wav", render(f0, hv_o, hn_o, d_o, gt), SR)
        sf.write(OUT / f"{uid}_net.wav", render(f0, hv_n, hn_n, d_n, gt), SR)
        sf.write(OUT / f"{uid}_oraclehop.wav",
                 render(f0, hv_o, hn_o, d_o, gt, sub=2), SR)
        sf.write(OUT / f"{uid}_nethop.wav",
                 render(f0, hv_n, hn_n, d_n, gt, sub=2), SR)
        report["utts"][uid] = {"T": T}
        print("done", uid, flush=True)
    (ROOT / "results/e2_triage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("-> results/e2_triage/", flush=True)


if __name__ == "__main__":
    main()
