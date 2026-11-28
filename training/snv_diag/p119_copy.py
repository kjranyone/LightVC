"""段 P119(学習なし): 描画器ごとの写しの、元の音との対数メルの差と UTMOS(元の音の UTMOS と並べる)。合成の声 3 声と開発の組の男声 4 人。
    cd training && SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_ESM=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p119_copy.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
ARMS = {"A": "", "C": "results/snv_p114_m0/snap/ema_10k.pt", "S": "results/snv_p97_s240/snap/ema_30k.pt", "Y": "results/snv_p97_y240/snap/ema_30k.pt"}


def lm(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    d = []
    for nf in (512, 1024, 2048):
        A = librosa.feature.melspectrogram(y=a[:n], sr=48000, n_fft=nf, hop_length=nf // 4, n_mels=80)
        B = librosa.feature.melspectrogram(y=b[:n], sr=48000, n_fft=nf, hop_length=nf // 4, n_mels=80)
        d.append(np.abs(np.log(A + 1e-7) - np.log(B + 1e-7)).mean())
    return float(np.mean(d))


def main() -> None:
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(EV.DEV).eval()
    mos = lambda y: float(utmos(torch.from_numpy(librosa.resample(y, orig_sr=EV.SR, target_sr=16000)).float()[None].to(EV.DEV), 16000).item())
    src = {f"syn_{v}": librosa.load(ROOT / f"data/snv_e1/{v}/t05.wav", sr=EV.SR, mono=True)[0].astype(np.float32) for v in ("ef100", "ef032", "ef300")}
    src |= {f"real{i + 1}": np.fromfile(PB.IN / sp / "in.f32", np.float32) for i, sp in enumerate(PB.SPS)}
    us = {k: PB.utt(x) for k, x in src.items()}
    res: dict = {"orig": {k: {"utmos": round(mos(u["wav"]), 3)} for k, u in us.items()}}
    for arm, ck in ARMS.items():
        os.environ.pop("SNV_RCK", None)
        if ck:
            os.environ["SNV_RCK"] = ck
        gen = EV.load_gen()
        res[arm] = {}
        for k, u in us.items():
            f0 = u["f0"].astype(np.float32)
            vv = (f0 > 0).astype(np.float32)
            lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
            cond = torch.from_numpy(np.concatenate([u["y"][:25], lf[None], vv[None], np.clip(u["y"][25:29], 0, 1.5) * vv[None]], 0))[None].to(EV.DEV)
            cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
            w = np.asarray(EV.draw(gen, cond, torch.from_numpy(f0)[None].to(EV.DEV), None), np.float32)
            res[arm][k] = {"logmel": round(lm(u["wav"], w), 3), "utmos": round(mos(w), 3)}
    for g in ("syn", "real"):
        ks = [k for k in us if k.startswith(g)]
        print(g, "元の音 UTMOS", round(float(np.mean([res["orig"][k]["utmos"] for k in ks])), 3), "|", " | ".join(f"{a}: 差 {np.mean([res[a][k]['logmel'] for k in ks]):.3f}・UTMOS {np.mean([res[a][k]['utmos'] for k in ks]):.3f}" for a in ARMS))
    json.dump(res, open(ROOT / "results/snv_p119/copy.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
