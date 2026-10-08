"""天井の盲検: 元音声 vs Y-S1 c32 往復 vs BigVGAN v2 44k(mel再構成・参照専用)。

目的: Y-S1 codec 往復が耳で失格(コーラス・ざらつき・声質・こもり: 2026-09-24)。既知良好な神経ボコーダの
コピー合成が同じ発話で通るかを測り、失格が「Y-S1の設計/学習」か「要求水準が神経再合成一般を超える」かを切り分ける。
BigVGANは非因果・評価専用(推論経路に入れない・MIT)。発話は nrft_ab と同じ有声3発話(6s)。
出力: results/earbattery/ceil_ab/<trial>/{A,B,C}.wav・鍵 _key_聴取後に開く.json(元音声にRMS整合→共通減衰)。

    CUDA_VISIBLE_DEVICES=0 uv run python render_ceiling_ab.py
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import librosa
import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from train_d1 import build_index
from render_d1_ab import norm_trial, pick_voiced

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/earbattery/ceil_ab"
SNAPS = Path.home() / ".cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots"


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    snap = sorted(SNAPS.iterdir())[-1]
    h = AttrDict(json.loads((snap / "config.json").read_text()))
    voc = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    voc.load_state_dict(torch.load(snap / "bigvgan_generator.pt", map_location="cpu")["generator"])
    voc.remove_weight_norm()
    voc = voc.eval().to(dev)
    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev, weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ck["ema"])
    codec.eval()

    pairs, lats, held_spk = build_index(0)
    OUT.mkdir(parents=True, exist_ok=True)
    key: dict = {}
    for f in pick_voiced(pairs, lats, held_spk, 3):
        d = torch.load(f, map_location="cpu", weights_only=False)
        n48 = 600 * HOP_LENGTH
        x48, _ = librosa.load(d["path"], sr=SAMPLE_RATE, mono=True)
        x48 = x48[:n48].astype(np.float64)
        x44, _ = librosa.load(d["path"], sr=44100, mono=True)
        x44 = x44[:int(round(len(x48) * 44100 / SAMPLE_RATE))]
        with torch.no_grad():
            y_ys1 = codec.decode(codec.encode(torch.from_numpy(x48).float()[None, None].to(dev)))
            y_ys1 = y_ys1[0, 0].cpu().numpy().astype(np.float64)
            mel = get_mel_spectrogram(torch.from_numpy(x44).float()[None].to(dev), voc.h)
            y_bv44 = voc(mel).squeeze().cpu().numpy().astype(np.float64)
        y_bv = librosa.resample(y_bv44, orig_sr=44100, target_sr=SAMPLE_RATE)
        n = min(len(x48), len(y_ys1), len(y_bv))
        clips = {"source": x48[:n], "ys1_c32": y_ys1[:n], "bigvgan_v2": y_bv[:n]}
        normed = norm_trial(clips, clips["source"])
        names = list(normed)
        trial = f"ceil_{f.stem}"
        random.Random(trial).shuffle(names)
        td = OUT / trial
        td.mkdir(parents=True, exist_ok=True)
        key[trial] = {"utt": f.stem, "map": {}}
        for i, nm in enumerate(names):
            soundfile.write(td / f"{'ABC'[i]}.wav", normed[nm], SAMPLE_RATE)
            key[trial]["map"]["ABC"[i]] = nm
        print(trial, "ok", flush=True)
    (OUT / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
