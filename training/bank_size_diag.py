"""
Diagnostic: oracle retrieval on the SAME bank as the learned model
(first 512 frames from 20 utts) vs learned attention.

If oracle also drops → bank size is the bottleneck
If oracle stays ~0.33 → model is the bottleneck
"""
import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000
FRAME_PERIOD = 5.0
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24
MC_DIM = 25
VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")


def analyze_wav(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)
    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)
    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
    return {"f0": f0.astype(np.float32), "mc": mc.astype(np.float32), "ap": ap}


def synth(f0, mc, ap):
    mc64 = np.ascontiguousarray(mc, dtype=np.float64)
    sp = sptk.mc2sp(mc64, ALPHA, FFTL)
    ap64 = np.ascontiguousarray(ap, dtype=np.float64)
    f064 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f064, sp, ap64, SR, frame_period=FRAME_PERIOD).astype(np.float32)


def shift_f0(f0, tgt_mean):
    voiced = f0[f0 > 0]
    if len(voiced) == 0: return f0.astype(np.float64)
    src_mean = float(np.exp(np.mean(np.log(voiced))))
    return np.where(f0 > 0, f0 * tgt_mean / src_mean, 0).astype(np.float64)


def find_pairs(n=20):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    used = set()
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb or sa in used or sb in used: continue
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                used.add(sa); used.add(sb)
                if len(pairs) >= n: return pairs
    return pairs


def build_bank_like_model(tgt_spk, n_utts=20, max_frames=512):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    all_mc = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
    bank = np.concatenate(all_mc, axis=0).astype(np.float32)
    return bank[:max_frames]


def build_bank_like_oracle(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    all_mc = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
    return np.concatenate(all_mc, axis=0).astype(np.float32)


def compute_speaker_mean(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    all_mc = []
    for f in files:
        d = np.load(f)
        all_mc.append(d["mc"])
    return np.concatenate(all_mc, axis=0).mean(axis=0)


def retrieve_weighted(query_keys, bank_keys, bank_values, k=5, temperature=1.0):
    tree = cKDTree(bank_keys)
    dist, idx = tree.query(query_keys, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    weights = np.exp(-dist / (temperature + 1e-10))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    return np.einsum('nk,nkd->nd', weights, bank_values[idx])


def main():
    DEVICE = torch.device("cuda")
    print("=== Bank Size Diagnostic ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(20)
    print(f"Pairs: {len(pairs)}\n")

    configs = {
        "oracle_full": [],
        "oracle_512_modelbank": [],
        "oracle_2048_modelbank": [],
    }

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])
        mc_s = feat_s["mc"]; f0_s = feat_s["f0"]; ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]; f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        src_mean = compute_speaker_mean(p["src"])
        tgt_mean = compute_speaker_mean(p["tgt"])
        src_norm = mc_s - src_mean

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        # oracle full (exclude target text, 10 utts)
        bank_full = build_bank_like_oracle(p["tgt"], p["text"], n_utts=10)
        bank_full_norm = bank_full - tgt_mean
        mc_pred = retrieve_weighted(src_norm, bank_full_norm, bank_full, k=5, temperature=1.0)
        wav_out = synth(f0_shifted[:T], mc_pred[:T].astype(np.float32), ap_s[:T])

        # oracle 512 (same bank as model: first 512 frames from 20 utts)
        bank_512 = build_bank_like_model(p["tgt"], n_utts=20, max_frames=512)
        bank_512_norm = bank_512 - tgt_mean
        mc_pred_512 = retrieve_weighted(src_norm, bank_512_norm, bank_512, k=5, temperature=1.0)
        wav_512 = synth(f0_shifted[:T], mc_pred_512[:T].astype(np.float32), ap_s[:T])

        # oracle 2048 (same bank source but more frames)
        bank_2048 = build_bank_like_model(p["tgt"], n_utts=20, max_frames=2048)
        bank_2048_norm = bank_2048 - tgt_mean
        mc_pred_2048 = retrieve_weighted(src_norm, bank_2048_norm, bank_2048, k=5, temperature=1.0)
        wav_2048 = synth(f0_shifted[:T], mc_pred_2048[:T].astype(np.float32), ap_s[:T])

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)
            s_full = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
            s_512 = F.cosine_similarity(e_tgt, emb(wav_512), dim=-1).item()
            s_2048 = F.cosine_similarity(e_tgt, emb(wav_2048), dim=-1).item()

        configs["oracle_full"].append(s_full)
        configs["oracle_512_modelbank"].append(s_512)
        configs["oracle_2048_modelbank"].append(s_2048)
        print(f"  [{idx+1}/{len(pairs)}] {p['src']}→{p['tgt']}: "
              f"full={s_full:.3f} 512={s_512:.3f} 2048={s_2048:.3f}", flush=True)

    print(f"\n=== Diagnostic Results ===")
    print(f"{'config':<30} {'mean':>8} {'std':>8}")
    print("-" * 50)
    for name in ["oracle_full", "oracle_2048_modelbank", "oracle_512_modelbank"]:
        arr = np.array(configs[name])
        print(f"{name:<30} {arr.mean():>8.4f} {arr.std():>8.4f}")
    print(f"\n学習済みattention:    0.153")
    print(f"→ bank sizeが原因か、モデルが原因かが分かる")


if __name__ == "__main__":
    main()
