"""G2 の帯域の切り分け(学習なし): DCT24 の包絡の差し替えを 5kHz 未満だけ / 5kHz 以上だけに限ると、話者はどれだけ運ばれるか。

物理の管(平面波)は約 5〜7kHz までしか有効でなく、artic_g2 の PHYS は倍音 ≤ 5kHz にだけ当てはめている。一方 D24V は全帯域を差し替える。
話者の手がかりが高域にあるなら、PHYS の劣後は帯域の差で説明できる。同じフレーム(両側とも有声)・同じ採点(女声 45 人中の目標の順位)。

    uv run python artic_g2_band.py --ladder <scratchpad>/r4_spk --n 20 --out ../results/artic_inv/g2_band.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
FCUT = 5000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lad = Path(a.ladder)
    sys.path.insert(0, str(lad))
    import gen as G
    import artic_fit as AF
    J = json.loads((lad / "jobs.json").read_text())
    mf = [j for j in J["jobs"] if j[0] == "MF"][: a.n]
    outdir = ROOT / "results/artic_inv/g2_wav"
    lo = (G.mc < FCUT)[:, None]
    for _, src, tgt, _ in mf:
        xr, _ = sf.read(lad / "wav" / f"MF__{src}__{tgt}__R0.wav")
        xt, _ = sf.read(lad / "wav" / f"MF__{src}__{tgt}__T0.wav")
        xr, xt = xr.astype(np.float64), xt.astype(np.float64)
        Xb = G.stft(xr)
        Eb, Eo = G.env(G.logmel(Xb), 24), G.env(G.logmel(G.stft(xt)), 24)
        pos = G.align_map(xr, xt, Xb.shape[1], Eo.shape[1])
        fb_, _ = AF.harvest(xr)
        fo_, _ = AF.harvest(xt)
        import nvoc as N
        jb = np.clip(np.round(np.arange(Xb.shape[1]) * N.HOP / AF.HOP).astype(int), 0, len(fb_) - 1)
        jo = np.clip(np.round(pos * N.HOP / AF.HOP).astype(int), 0, len(fo_) - 1)
        both = (fb_[jb] > 0) & (fo_[jo] > 0)
        gd = G.interp_frames(Eo, pos) - Eb
        for cond, m in (("D24V_LO", lo), ("D24V_HI", ~lo)):
            g = np.zeros_like(gd)
            g[:, both] = (gd * m)[:, both]
            y = G.istft(Xb * np.exp(G.to_lin(np.clip(g, -6, 6))), len(xr))
            sf.write(outdir / f"MF__{src}__{tgt}__{cond}.wav", G.norm(y).astype(np.float32), 48000)
        print(src, tgt, "both-voiced", round(float(both.mean()), 2), flush=True)
    from a2_dsp_vc import ecapa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    from train_dec2 import load48
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    def emb_w(x48):
        y = librosa.resample(x48.astype(np.float64), orig_sr=48000, target_sr=16000).astype(np.float32)
        with torch.no_grad():
            v = wm(**fe(y, sampling_rate=16000, return_tensors="pt")).embeddings[0].numpy().astype(np.float64)
        return v / np.linalg.norm(v)
    emb_e = ecapa()
    fems = J["fems"]

    def utt(s, u):
        return VC / f"wav48/{s}/{s}_{u:03d}.wav"
    rep: dict = {"n": len(mf), "fcut_hz": FCUT}
    for name, f_ in (("ecapa", emb_e), ("wavlm", emb_w)):
        C = {}
        for s in fems:
            c = np.mean([f_(load48(str(utt(s, u))).astype(np.float64)[:8 * 48000]) for u in [u for u in range(60, 200) if utt(s, u).exists()][:3]], 0)
            C[s] = c / np.linalg.norm(c)
        res = {}
        for cond in ("D24V_LO", "D24V_HI", "D24V", "PHYS", "PHYS_K16LS"):
            rk = []
            for _, src, tgt, _ in mf:
                p = outdir / f"MF__{src}__{tgt}__{cond}.wav"
                if not p.exists():
                    continue
                x, _ = sf.read(p)
                v = f_(x.astype(np.float64)[: 8 * 48000])
                sims = {gg: float(v @ C[gg]) for gg in fems}
                rk.append(1 + sum(1 for gg in fems if gg != tgt and sims[gg] > sims[tgt]))
            rk = np.array(rk)
            res[cond] = {"n": len(rk), "top1": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "median_rank": float(np.median(rk))}
            print(name, cond, res[cond], flush=True)
        rep[name] = res
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
