"""
Source-filter oracle follow-up: F0 shift + DTW alignment impact.

O1b: F0-shifted source + target real envelope (no DTW)
O1c: source F0 + DTW-aligned target real envelope
O1d: F0-shifted source + DTW-aligned target real envelope
O3b: per-register full replacement (not transport) with F0 shift

Tests how much F0 and alignment contribute to the O1 ceiling.
"""
import sys, os, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa
from fastdtw import fastdtw

DEVICE = torch.device("cuda")
SR = 16000
FRAME_PERIOD = 5.0
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24

VCTK_WAV = Path("../data/vctk_200")

from oracle_sf import (
    wav_to_features, features_to_wav_mc, compute_cross_secs,
    load_secs_model, find_same_text_pairs, compute_register,
    build_speaker_profiles
)


def dtw_align_mc(mc_src, mc_tgt):
    dist, path = fastdtw(mc_src, mc_tgt, radius=50)
    src_to_tgt = {}
    tgt_to_src = {}
    for s_idx, t_idx in path:
        src_to_tgt.setdefault(s_idx, []).append(t_idx)
        tgt_to_src.setdefault(t_idx, []).append(s_idx)
    src_map = np.zeros(len(mc_src), dtype=int)
    for s_idx in range(len(mc_src)):
        if s_idx in src_to_tgt:
            src_map[s_idx] = src_to_tgt[s_idx][0]
        else:
            src_map[s_idx] = min(src_map[s_idx-1] if s_idx > 0 else 0, len(mc_tgt)-1)
    return src_map


def shift_f0(f0_src, f0_tgt_ref):
    voiced = f0_src[f0_src > 0]
    if len(voiced) == 0:
        return f0_src.copy()
    mean_src = np.exp(np.mean(np.log(voiced)))
    tgt_voiced = f0_tgt_ref[f0_tgt_ref > 0]
    if len(tgt_voiced) == 0:
        return f0_src.copy()
    mean_tgt = np.exp(np.mean(np.log(tgt_voiced)))

    ratio = mean_tgt / mean_src
    f0_shifted = np.where(f0_src > 0, f0_src * ratio, 0.0)
    return f0_shifted


def o1b_f0shift_target_env(model, pairs):
    """Shifted source F0 + target real envelope."""
    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_s != SR: wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        if sr_t != SR: wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)

        feat_src = wav_to_features(wav_src, SR)
        feat_tgt = wav_to_features(wav_tgt, SR)

        T = min(len(feat_src["f0"]), len(feat_tgt["mc"]))
        f0_shifted = shift_f0(feat_src["f0"][:T], feat_tgt["f0"][:T])

        synth_feat = {
            "f0": f0_shifted,
            "mc": feat_tgt["mc"][:T],
            "codeap": feat_tgt["codeap"][:T],
        }
        wav_syn = features_to_wav_mc(synth_feat, SR)
        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def o1c_dtw_target_env(model, pairs):
    """Source F0 + DTW-aligned target envelope."""
    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_s != SR: wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        if sr_t != SR: wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)

        feat_src = wav_to_features(wav_src, SR)
        feat_tgt = wav_to_features(wav_tgt, SR)

        src_map = dtw_align_mc(feat_src["mc"], feat_tgt["mc"])
        T = len(feat_src["f0"])
        mc_aligned = feat_tgt["mc"][src_map[:T]]
        codeap_aligned = feat_tgt["codeap"][src_map[:T]]

        synth_feat = {
            "f0": feat_src["f0"][:T],
            "mc": mc_aligned,
            "codeap": codeap_aligned,
        }
        wav_syn = features_to_wav_mc(synth_feat, SR)
        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def o1d_f0shift_dtw_env(model, pairs):
    """Shifted F0 + DTW-aligned target envelope."""
    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_s != SR: wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        if sr_t != SR: wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)

        feat_src = wav_to_features(wav_src, SR)
        feat_tgt = wav_to_features(wav_tgt, SR)

        src_map = dtw_align_mc(feat_src["mc"], feat_tgt["mc"])
        T = len(feat_src["f0"])
        mc_aligned = feat_tgt["mc"][src_map[:T]]
        codeap_aligned = feat_tgt["codeap"][src_map[:T]]
        f0_shifted = shift_f0(feat_src["f0"][:T], feat_tgt["f0"][:T])

        synth_feat = {
            "f0": f0_shifted,
            "mc": mc_aligned,
            "codeap": codeap_aligned,
        }
        wav_syn = features_to_wav_mc(synth_feat, SR)
        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def o3b_replace_f0shift(model, pairs, profiles):
    """Per-register replacement: target profile mean + F0 shift (no DTW)."""
    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_s != SR: wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        if sr_t != SR: wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)

        feat_src = wav_to_features(wav_src, SR)
        feat_tgt = wav_to_features(wav_tgt, SR)
        tgt_spk = p["tgt_spk"]
        prof_t = profiles.get(tgt_spk)
        if prof_t is None:
            continue

        reg_src = compute_register(feat_src)
        T = len(reg_src)
        f0_shifted = shift_f0(feat_src["f0"][:T], feat_tgt["f0"][:T])

        mc_out = np.zeros((T, MC_ORDER + 1), dtype=np.float32)
        codeap_out = np.zeros((T, feat_src["codeap"].shape[1]), dtype=np.float32)
        for t in range(T):
            r = reg_src[t]
            mc_out[t] = prof_t["mean_mc"].get(r, prof_t["global_mean_mc"])
            codeap_out[t] = prof_t["mean_codeap"].get(r, prof_t["global_mean_codeap"])

        synth_feat = {"f0": f0_shifted, "mc": mc_out, "codeap": codeap_out}
        wav_syn = features_to_wav_mc(synth_feat, SR)
        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def main():
    print("=== Source-Filter Oracle Follow-up: F0 + DTW ===\n")
    print("Installing fastdtw...")
    os.system("uv pip install fastdtw 2>/dev/null")

    secs_model = load_secs_model()
    pairs = find_same_text_pairs(20)
    print(f"Found {len(pairs)} pairs")

    all_results = {}

    print("\n--- O1b: F0-shifted source + target real envelope ---")
    t0 = time.time()
    r = o1b_f0shift_target_env(secs_model, pairs)
    tgt = np.array([x["tgt"] for x in r]); src = np.array([x["src"] for x in r])
    print(f"  SECS(tgt): {tgt.mean():.4f} ± {tgt.std():.4f}")
    print(f"  SECS(src): {src.mean():.4f} ± {src.std():.4f}  ({time.time()-t0:.1f}s)")
    all_results["O1b_f0shift"] = {"tgt": float(tgt.mean()), "src": float(src.mean())}

    print("\n--- O1c: Source F0 + DTW-aligned target envelope ---")
    t0 = time.time()
    r = o1c_dtw_target_env(secs_model, pairs)
    tgt = np.array([x["tgt"] for x in r]); src = np.array([x["src"] for x in r])
    print(f"  SECS(tgt): {tgt.mean():.4f} ± {tgt.std():.4f}")
    print(f"  SECS(src): {src.mean():.4f} ± {src.std():.4f}  ({time.time()-t0:.1f}s)")
    all_results["O1c_dtw"] = {"tgt": float(tgt.mean()), "src": float(src.mean())}

    print("\n--- O1d: F0-shifted + DTW-aligned target envelope ---")
    t0 = time.time()
    r = o1d_f0shift_dtw_env(secs_model, pairs)
    tgt = np.array([x["tgt"] for x in r]); src = np.array([x["src"] for x in r])
    print(f"  SECS(tgt): {tgt.mean():.4f} ± {tgt.std():.4f}")
    print(f"  SECS(src): {src.mean():.4f} ± {src.std():.4f}  ({time.time()-t0:.1f}s)")
    all_results["O1d_f0shift_dtw"] = {"tgt": float(tgt.mean()), "src": float(src.mean())}

    print("\n--- Building profiles for O3b ---")
    profiles = build_speaker_profiles(n_speakers=40)

    print("\n--- O3b: Per-register replacement + F0 shift ---")
    t0 = time.time()
    r = o3b_replace_f0shift(secs_model, pairs, profiles)
    tgt = np.array([x["tgt"] for x in r]); src = np.array([x["src"] for x in r])
    print(f"  SECS(tgt): {tgt.mean():.4f} ± {tgt.std():.4f}")
    print(f"  SECS(src): {src.mean():.4f} ± {src.std():.4f}  ({time.time()-t0:.1f}s)")
    all_results["O3b_replace_f0shift"] = {"tgt": float(tgt.mean()), "src": float(src.mean())}

    print("\n=== Summary (SECS to target) ===")
    print(f"  O1  (src F0 + tgt env, no align):  0.4304 (baseline)")
    print(f"  O1b (F0 shift + tgt env):          {all_results['O1b_f0shift']['tgt']:.4f}")
    print(f"  O1c (src F0 + DTW env):            {all_results['O1c_dtw']['tgt']:.4f}")
    print(f"  O1d (F0 shift + DTW env):          {all_results['O1d_f0shift_dtw']['tgt']:.4f}")
    print(f"  O3b (reg replace + F0 shift):      {all_results['O3b_replace_f0shift']['tgt']:.4f}")

    with open("results/oracle_sf_followup.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to results/oracle_sf_followup.json")


if __name__ == "__main__":
    main()
