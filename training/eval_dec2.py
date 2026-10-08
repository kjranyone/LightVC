"""decoder v2 対照腕(A=diag_dec2_aa / B=diag_dec2_a2)の評価と盲検素材(prereg: results/diag_dec2_*/prereg.yaml)。

参照=元音声。すべて同じ c32 encoder 潜在を各 decoder で decode(GT decode の比較)。B の f0 は f0fix(harvest)を
因果整列したもの(製品は因果f0=要別検証)。
1) held24(各held話者の先頭発話・≤8s): logmel/mrstft 対元音声・|Δf0|[半音]・有声率差
2) 盲検: 有声率の高い train 話者3発話 + held 話者2発話 × {source, bigvgan_v2, c32, dec_aa, dec_a2}
   → results/earbattery/dec2_ab/(元音声にRMS整合→共通減衰・鍵 _key_聴取後に開く.json)

    CUDA_VISIBLE_DEVICES=0 uv run python eval_dec2.py [--which ema]
"""
from __future__ import annotations

import argparse
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
from decoder_aa import DecoderAA
from decoder_a2 import DecoderA2
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
from train_d1 import build_index
from train_cfmys import F0FIX
from train_dec2 import f0_latent, load48
from diag_cfm_audit import decode_f0
from render_d1_ab import norm_trial, pick_voiced

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "results/earbattery/dec2_ab"
SNAPS = Path.home() / ".cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["ema", "dec"], default="ema")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev, weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ck["ema"])
    codec.eval()
    decs = {"c32": lambda z, f0: codec.decoder(z)}
    ca = torch.load(ROOT / "results/diag_dec2_aa/diag_dec2_aa_last.pt", map_location=dev, weights_only=False)
    dA = DecoderAA(taps=8).to(dev)
    dA.load_state_dict(ca[a.which])
    dA.eval()
    decs["dec_aa"] = lambda z, f0: dA(z)
    cb = torch.load(ROOT / "results/diag_dec2_a2/diag_dec2_a2_last.pt", map_location=dev, weights_only=False)
    dB = DecoderA2(channels=12).to(dev)
    dB.load_state_dict(cb[a.which])
    dB.eval()
    decs["dec_a2"] = lambda z, f0: dB(z, f0)
    steps = {"dec_aa": int(ca["step"]), "dec_a2": int(cb["step"])}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]

    def metrics(y, x, f0x):
        n = min(y.shape[-1], x.shape[-1])
        y, x = y[..., :n].reshape(1, 1, -1), x[..., :n].reshape(1, 1, -1)
        fy = decode_f0(np.clip(y.cpu().numpy().reshape(-1).astype(np.float64), -1, 1))
        return {"logmel": float(logmel_l1(y, x, mels)), "mrstft": float(mrstft(y, x)),
                "f0_abs_st": (abs(float(12 * np.log2(fy["f0_median"] / f0x["f0_median"])))
                              if fy["f0_median"] > 0 and f0x["f0_median"] > 0 else float("nan")),
                "d_voiced": fy["voiced_ratio"] - f0x["voiced_ratio"]}

    pairs, lats, held_spk = build_index(0)
    hset = set(held_spk)

    def item(f, max_fr):
        d = torch.load(f, map_location="cpu", weights_only=False)
        x = load48(d["path"])
        T = min(len(x) // HOP_LENGTH, max_fr)
        f0r = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu", weights_only=False)["f0"].numpy()
        return (torch.from_numpy(x[:T * HOP_LENGTH]).to(dev), torch.from_numpy(f0_latent(f0r, T)).to(dev)[None], d)

    rows = {k: [] for k in decs}
    held_first = []
    for s in held_spk:
        c = sorted(p for p in pairs if p.parent.name == s)
        if c:
            held_first.append(c[0])
    with torch.no_grad():
        for f in held_first:
            x, f0, _ = item(f, 800)
            f0x = decode_f0(np.clip(x.cpu().numpy().astype(np.float64), -1, 1))
            z = codec.encode(x[None, None])
            for k, fn in decs.items():
                rows[k].append(metrics(fn(z, f0), x, f0x))
    rep = {"which": a.which, "steps": steps, "held_n": len(held_first), "held": {}}
    for k, v in rows.items():
        rep["held"][k] = {m: {"mean": round(float(np.nanmean([r[m] for r in v])), 4),
                              "median": round(float(np.nanmedian([r[m] for r in v])), 4)} for m in v[0]}
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    (ROOT / "results/ys1_dec2/eval_step2.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))

    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    snap = sorted(SNAPS.iterdir())[-1]
    voc = bigvgan.BigVGAN(AttrDict(json.loads((snap / "config.json").read_text())), use_cuda_kernel=False)
    voc.load_state_dict(torch.load(snap / "bigvgan_generator.pt", map_location="cpu")["generator"])
    voc.remove_weight_norm()
    voc = voc.eval().to(dev)
    AB.mkdir(parents=True, exist_ok=True)
    key: dict = {}
    utts = pick_voiced(pairs, lats, held_spk, 3) + held_first[:2]
    for f in utts:
        x, f0, d = item(f, 600)
        with torch.no_grad():
            z = codec.encode(x[None, None])
            clips = {"source": x.cpu().numpy().astype(np.float64)}
            for k, fn in decs.items():
                clips[k] = fn(z, f0)[0, 0].cpu().numpy().astype(np.float64)
            x44, _ = librosa.load(d["path"], sr=44100, mono=True)
            x44 = x44[:int(round(x.shape[-1] * 44100 / SAMPLE_RATE))]
            mel = get_mel_spectrogram(torch.from_numpy(x44).float()[None].to(dev), voc.h)
            clips["bigvgan_v2"] = librosa.resample(voc(mel).squeeze().cpu().numpy().astype(np.float64),
                                                   orig_sr=44100, target_sr=SAMPLE_RATE)
        n = min(len(v) for v in clips.values())
        clips = {k: v[:n] for k, v in clips.items()}
        normed = norm_trial(clips, clips["source"])
        names = list(normed)
        trial = f"dec2_{f.stem}"
        random.Random(trial).shuffle(names)
        td = AB / trial
        td.mkdir(parents=True, exist_ok=True)
        key[trial] = {"utt": f.stem, "held_speaker": f.parent.name in hset, "map": {}}
        for i, nm in enumerate(names):
            L = "ABCDE"[i]
            soundfile.write(td / f"{L}.wav", normed[nm], SAMPLE_RATE)
            key[trial]["map"][L] = nm
        print(trial, "ok", flush=True)
    (AB / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
