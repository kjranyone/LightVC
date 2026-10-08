"""d4bゲート判定: G1コーラス不在・G2 free-run品質・G3 lf0掃引。

    CUDA_VISIBLE_DEVICES=0 uv run python eval_d4b_gates.py [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import HOP
from train_d4b import D4Cat, sample_from, mulaw_decode
from train_d4a import get_utt_cond, held_paths, ROOT
from causal_codec import CausalCodec
from diag_cfm_audit import LAT, decode_f0

OUT = ROOT / "results/d4b_wavenet"
LF0_IDX = 80


def shift_lf0_cond(cond: torch.Tensor, semitones: float) -> torch.Tensor:
    c = cond.clone()
    hz = 200.0 * torch.exp(c[0, LF0_IDX])
    hz_s = torch.where(hz > 50.5, hz * 2.0 ** (semitones / 12.0), hz)
    c[0, LF0_IDX] = torch.log(hz_s.clamp(min=50.0) / 200.0)
    return c


def band_metrics(y: np.ndarray) -> dict:
    w44 = librosa.resample(y.astype(np.float64), orig_sr=48000, target_sr=44100)
    S = np.abs(librosa.stft(w44, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=44100, n_fft=2048)
    tot = S.sum() + 1e-12
    return {"hi_mid": round(float(S[(f > 2000) & (f < 6000)].sum() / tot), 4),
            "hi": round(float(S[(f > 6000) & (f < 12000)].sum() / tot), 4)}


def codec_ceiling_metrics(wav48: np.ndarray, codec) -> dict:
    n = len(wav48) // HOP * HOP
    with torch.no_grad():
        z = codec.encode(torch.from_numpy(wav48[:n])[None, None].cuda())
        y = codec.decode(z)[0, 0].cpu().numpy()
    return band_metrics(y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dir", default="d4b_wavenet",
                    help="results/下の腕ディレクトリ(出力先兼ckpt既定)")
    ap.add_argument("--out", default="gates.json")
    ap.add_argument("--temps", default="0.9,0.5,0.3",
                    help="free-run品質は温度に強く依存(2026-09-21実測)。全温度で記録")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    temps = [float(t) for t in a.temps.split(",")]
    out_dir = ROOT / "results" / a.dir
    if a.ckpt is None:
        a.ckpt = str(out_dir / f"{a.dir}_last.pt")
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = D4Cat(excitation=ck["net"]["inp.weight"].shape[1] == 3).to(dev).eval()
    m.load_state_dict(ck["net"])
    print(f"  ckpt step {ck.get('step')}", flush=True)

    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()

    res = {"ckpt_step": ck.get("step"), "temps": temps}
    G1, G2 = {}, {}
    for i, p in enumerate(held_paths()[:4]):
        wv, cond = get_utt_cond(p)
        n = min(wv.shape[0], 192000)
        ct = cond[None, :, : n // HOP].to(dev)
        ceil = codec_ceiling_metrics(wv[:n].numpy(), codec)
        src_f0 = decode_f0(wv[:n].numpy())["f0_median"]
        for tp in temps:
            y = sample_from(m, ct, n, dev, tp)[0].cpu().numpy()
            soundfile.write(out_dir / f"gate_held{i}_t{tp}.wav", np.clip(y, -1, 1), 48000)
            fr = decode_f0(np.clip(y, -1, 1))
            bm = band_metrics(y)
            G2[f"held{i}_t{tp}"] = {"f0": fr["f0_median"], "src_f0": src_f0,
                                    "voiced": fr["voiced_ratio"]}
            G1[f"held{i}_t{tp}"] = {"fr_hi_mid": bm["hi_mid"],
                                    "ceil_hi_mid": ceil["hi_mid"],
                                    "ratio": round(bm["hi_mid"] / max(ceil["hi_mid"], 1e-4), 2)}
            print(f"  held{i} t{tp}: G2 {G2[f'held{i}_t{tp}']}  "
                  f"G1 {G1[f'held{i}_t{tp}']}", flush=True)
    res["G2_freerun"] = G2
    res["G1_chorus"] = G1

    G3 = {}
    p = held_paths()[0]
    wv, cond = get_utt_cond(p)
    n = min(wv.shape[0], 96000)
    base = None
    tp = temps[0]
    for st in (0.0, 7.0, 12.0):
        c = shift_lf0_cond(cond[None, :, : n // HOP].to(dev), st)
        y = sample_from(m, c, n, dev, tp)[0].cpu().numpy()
        soundfile.write(out_dir / f"gate_sweep_st{int(st)}.wav", np.clip(y, -1, 1), 48000)
        mm = decode_f0(np.clip(y, -1, 1))
        G3[f"st{int(st)}"] = mm
        if st == 0.0:
            base = mm["f0_median"]
    if base:
        for st in (7, 12):
            if G3[f"st{st}"]["f0_median"] > 0:
                G3[f"sweep{st}_st"] = round(
                    12 * np.log2(G3[f"st{st}"]["f0_median"] / base), 2)
    res["G3_lf0_sweep"] = G3
    print("  G3:", G3, flush=True)
    (out_dir / a.out).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
