"""D4v2ゲート判定(prereg: results/d4v2/prereg.yaml)。

G_speaker: fecf→ab97変換(CFG w=1.5)のECAPA cos_vs_target ≥0.40(3seed×2発話平均)。
G_quality: 変換音のband指標がcodec天井比2倍以内。
G_f0: 同話者再構成のlf0掃引 ±3st追従。

    CUDA_VISIBLE_DEVICES=0 uv run python eval_d4v2.py [--ckpt ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
from speechbrain.inference.speaker import EncoderClassifier

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import HOP
from train_d4v2 import get_utt_cond_v2, load_cent, sample_cfg
from train_d4a import held_paths, ROOT
from eval_d4b_gates import shift_lf0_cond, band_metrics, codec_ceiling_metrics
from diag_cfm_audit import decode_f0
from causal_codec import CausalCodec

OUT = ROOT / "results/d4v2_spknorm"
SRC_SPK = "fecf5112354be881"
TGT_SPK = "ab97e212acbb6d6b"


def ecapa():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa", run_opts={"device": "cuda"})


def embed(m, path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=16000, mono=True)
    es = []
    for i in range(0, len(y), 20 * 16000):
        seg = y[i:i + 20 * 16000]
        if len(seg) < 1600:
            continue
        with torch.no_grad():
            e = m.encode_batch(
                torch.from_numpy(seg).float().unsqueeze(0).cuda()
            ).squeeze().cpu().numpy()
        es.append(e)
    v = np.mean(es, 0)
    return v / (np.linalg.norm(v) + 1e-6)


def spk_ref(m, spk: str, k: int = 8) -> np.ndarray:
    wavs = sorted((ROOT / "female-dataset" / spk).glob("*.wav"))[:k]
    return np.mean([embed(m, str(w)) for w in wavs], 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUT / "d4v2_spknorm_best.pt"))
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--cfg-w", type=float, default=1.5)
    a = ap.parse_args()
    dev = "cuda"
    m = torch.load(a.ckpt, map_location=dev, weights_only=False)
    from train_d4b import D4Cat
    net = D4Cat().to(dev).eval()
    net.load_state_dict(m["net"])
    print(f"  ckpt step {m.get('step')}", flush=True)

    cent = load_cent()
    srcs = sorted((ROOT / "female-dataset" / SRC_SPK).glob("*.wav"))
    srcs = [p for p in srcs
            if (ROOT / "data/female_real_feat" / SRC_SPK / (p.stem + ".pt")).exists()
            and (ROOT / "data/female_real_f0fix" / SRC_SPK / (p.stem + ".pt")).exists()
            ][:2]
    print(f"  src utts {len(srcs)}", flush=True)

    ec = ecapa()
    tgt_ref = spk_ref(ec, TGT_SPK)
    src_ref = spk_ref(ec, SRC_SPK)

    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()

    res = {"ckpt_step": m.get("step"), "cfg_w": a.cfg_w, "temp": a.temp}
    Gs, Gq = {}, {}
    for i, p in enumerate(srcs):
        it = get_utt_cond_v2(p, cent, spk_override=TGT_SPK)
        wv, cond = it
        n = min(wv.shape[0], 144000)
        cs_t, cs_s = [], []
        for sd in (0, 1, 2):
            ct = cond[None, :, : n // HOP].to(dev)
            y = sample_cfg(net, ct, n, dev, a.temp, a.cfg_w, seed=sd)[0].cpu().numpy()
            f = OUT / f"conv{i}_s{sd}.wav"
            soundfile.write(f, np.clip(y, -1, 1), 48000)
            e = embed(ec, str(f))
            cs_t.append(float(e @ tgt_ref))
            cs_s.append(float(e @ src_ref))
        bm = band_metrics(y)
        ceil = codec_ceiling_metrics(wv[:n].numpy(), codec)
        Gs[f"utt{i}"] = {"cos_tgt": round(float(np.mean(cs_t)), 4),
                         "cos_tgt_seeds": [round(c, 4) for c in cs_t],
                         "cos_src_leak": round(float(np.mean(cs_s)), 4)}
        Gq[f"utt{i}"] = {"fr_hi_mid": bm["hi_mid"], "ceil_hi_mid": ceil["hi_mid"],
                         "ratio": round(bm["hi_mid"] / max(ceil["hi_mid"], 1e-4), 2)}
        print(f"  utt{i}: {Gs[f'utt{i}']}  Gq {Gq[f'utt{i}']}", flush=True)
    res["refs"] = {"ceiling": "tgt gt vs own ref未計測(既存spk_asr_probe参照)",
                   "floor_cross": round(float(src_ref @ tgt_ref), 4)}
    res["G_speaker"] = Gs
    res["G_quality"] = Gq
    mt = float(np.mean([v["cos_tgt"] for v in Gs.values()]))
    mq = max(v["ratio"] for v in Gq.values())
    res["verdict"] = {"G_speaker": "PASS" if mt >= 0.40 else "FAIL",
                      "G_quality": "PASS" if mq <= 2.0 else "FAIL",
                      "cos_tgt_mean": round(mt, 4)}

    G3 = {}
    hp = held_paths()[0]
    it = get_utt_cond_v2(hp, cent)
    wv, cond = it
    n = min(wv.shape[0], 96000)
    base = None
    for st in (0.0, 3.0, -3.0):
        c = shift_lf0_cond(cond[None, :, : n // HOP].to(dev), st)
        y = sample_cfg(net, c, n, dev, a.temp, a.cfg_w)[0].cpu().numpy()
        soundfile.write(OUT / f"sweep_st{int(st)}.wav", np.clip(y, -1, 1), 48000)
        mm = decode_f0(np.clip(y, -1, 1))
        G3[f"st{int(st)}"] = mm
        if st == 0.0:
            base = mm["f0_median"]
    if base:
        for st in (3, -3):
            if G3[f"st{st}"]["f0_median"] > 0:
                G3[f"sweep{st}_st"] = round(
                    12 * np.log2(G3[f"st{st}"]["f0_median"] / base), 2)
    res["G_f0_sweep"] = G3
    print("  G3:", G3, flush=True)
    (OUT / "gates.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["verdict"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
