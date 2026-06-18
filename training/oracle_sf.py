"""
Causal Source-Filter VC — Oracle Upper Bound Tests

Tests if the source-filter decomposition can carry speaker identity.

O1: source excitation + target real envelope → SECS
    (envelope alone carries speaker identity?)
O2: source F0 + target per-register mean envelope → SECS
    (mean envelope per pitch bin enough?)
O3: affine transport envelope (source→target) → SECS
    (optimal transport of mel-cepstrum)
O4: self analysis-synthesis → SECS
    (WORLD analysis-synthesis upper bound)
O5: frame size sweep (5ms, 10ms) → quality impact

Same-text pairs from VCTK: source speaker A + target speaker B, same text.
"""
import sys, os, glob, json, argparse, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa

DEVICE = torch.device("cuda")
SR = 16000
FRAME_PERIOD = 5.0  # ms (200 Hz)
FFTL = 2048
ALPHA = 0.410
MC_ORDER = 24

VCTK_WAV = Path("../data/vctk_200")


def wav_to_features(wav, sr=SR):
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)

    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)

    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
    codeap = world.code_aperiodicity(ap, SR)

    vuv = (f0 > 0).astype(np.float32)
    f0_hz = f0.astype(np.float32)

    energy = np.sqrt(np.sum(sp ** 2, axis=1)).astype(np.float32)
    energy_db = 10 * np.log10(energy + 1e-10)

    return {
        "f0": f0_hz,
        "vuv": vuv,
        "mc": mc.astype(np.float32),
        "codeap": codeap.astype(np.float32),
        "energy": energy_db,
        "sp": sp,
        "ap": ap,
    }


def features_to_wav(feat, sr=SR):
    sp = feat["sp"] if "sp" in feat else sptk.mc2sp(feat["mc"], ALPHA, FFTL)
    ap = feat["ap"] if "ap" in feat else world.decode_aperiodicity(feat["codeap"], SR, FFTL)
    f0 = feat["f0"].astype(np.float64)
    wav = world.synthesize(f0, sp, ap, sr, frame_period=FRAME_PERIOD)
    return wav.astype(np.float32)


def features_to_wav_mc(feat, sr=SR):
    mc = np.ascontiguousarray(feat["mc"], dtype=np.float64)
    sp = sptk.mc2sp(mc, ALPHA, FFTL)
    if "ap" in feat and feat["ap"] is not None:
        ap = np.ascontiguousarray(feat["ap"], dtype=np.float64)
    else:
        codeap = np.ascontiguousarray(feat["codeap"], dtype=np.float64)
        ap = world.decode_aperiodicity(codeap, SR, FFTL)
    f0 = np.ascontiguousarray(feat["f0"], dtype=np.float64)
    wav = world.synthesize(f0, sp, ap, sr, frame_period=FRAME_PERIOD)
    return wav.astype(np.float32)


def load_secs_model():
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    return model


def compute_secs(model, wav_ref, wav_syn, sr=SR):
    if sr != 16000:
        wav_ref = librosa.resample(wav_ref.astype(np.float32), orig_sr=sr, target_sr=16000)
        wav_syn = librosa.resample(wav_syn.astype(np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        e_ref = model.encode_batch(
            torch.from_numpy(wav_ref.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ).squeeze(0)
        e_syn = model.encode_batch(
            torch.from_numpy(wav_syn.astype(np.float32)).unsqueeze(0).to(DEVICE)
        ).squeeze(0)
    return F.cosine_similarity(e_ref, e_syn, dim=-1).item()


def compute_cross_secs(model, wav_src, wav_tgt, wav_syn, sr=SR):
    if sr != 16000:
        wav_src = librosa.resample(wav_src.astype(np.float32), orig_sr=sr, target_sr=16000)
        wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=16000)
        wav_syn = librosa.resample(wav_syn.astype(np.float32), orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        e_src = model.encode_batch(torch.from_numpy(wav_src).unsqueeze(0).to(DEVICE)).squeeze(0)
        e_tgt = model.encode_batch(torch.from_numpy(wav_tgt).unsqueeze(0).to(DEVICE)).squeeze(0)
        e_syn = model.encode_batch(torch.from_numpy(wav_syn).unsqueeze(0).to(DEVICE)).squeeze(0)
    return {
        "tgt": F.cosine_similarity(e_tgt, e_syn, dim=-1).item(),
        "src": F.cosine_similarity(e_src, e_syn, dim=-1).item(),
    }


def find_same_text_pairs(n_pairs=20):
    text_groups = defaultdict(list)
    for spk_dir in sorted(VCTK_WAV.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav_path in spk_dir.glob("*.wav"):
            stem = wav_path.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                text_id = parts[1]
                text_groups[text_id].append((spk, wav_path))

    pairs = []
    speakers_used = set()
    for text_id, utts in sorted(text_groups.items()):
        if len(utts) < 2:
            continue
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                spk_a, wav_a = utts[i]
                spk_b, wav_b = utts[j]
                if spk_a == spk_b:
                    continue
                if spk_a in speakers_used and spk_b in speakers_used:
                    continue
                pairs.append({
                    "text_id": text_id,
                    "src_spk": spk_a, "src_wav": str(wav_a),
                    "tgt_spk": spk_b, "tgt_wav": str(wav_b),
                })
                speakers_used.add(spk_a)
                speakers_used.add(spk_b)
                if len(pairs) >= n_pairs:
                    return pairs
    return pairs


def pitch_bin(f0_hz, n_bins=12, fmin=60, fmax=500):
    semitone = 12 * np.log2(np.clip(f0_hz, fmin, fmax) / fmin)
    bins = np.clip((semitone / (12 * np.log2(fmax / fmin)) * n_bins).astype(int), 0, n_bins - 1)
    return bins


def compute_register(feat, n_pitch_bins=8, n_energy_bins=3):
    f0 = feat["f0"]
    vuv = feat["vuv"]
    energy = feat["energy"]

    pb = pitch_bin(np.where(f0 > 0, f0, 200), n_pitch_bins)
    e_min, e_max = np.percentile(energy, 5), np.percentile(energy, 95)
    eb = np.clip(((energy - e_min) / (e_max - e_min + 1e-6) * n_energy_bins).astype(int), 0, n_energy_bins - 1)

    register = pb * (n_energy_bins) * 2 + eb * 2 + vuv.astype(int)
    return register


def o4_self_recon(model, pairs):
    """WORLD analysis-synthesis self-reconstruction."""
    results = []
    for p in pairs:
        wav, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        feat = wav_to_features(wav, SR)
        wav_recon = features_to_wav(feat, SR)

        secs = compute_secs(model, wav, wav_recon, SR)
        results.append(secs)
    return np.array(results)


def o1_source_excitation_target_envelope(model, pairs):
    """source F0/VUV + target spectral envelope + target aperiodicity."""
    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_s != SR:
            wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        if sr_t != SR:
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)

        feat_src = wav_to_features(wav_src, SR)
        feat_tgt = wav_to_features(wav_tgt, SR)

        T = min(len(feat_src["f0"]), len(feat_tgt["mc"]))
        synth_feat = {
            "f0": feat_src["f0"][:T],
            "vuv": feat_src["vuv"][:T],
            "mc": feat_tgt["mc"][:T],
            "codeap": feat_tgt["codeap"][:T],
        }
        wav_syn = features_to_wav_mc(synth_feat, SR)

        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def o2_mean_envelope(model, pairs, speaker_profiles=None):
    """source F0 + target per-register mean mel-cepstrum."""
    if speaker_profiles is None:
        speaker_profiles = build_speaker_profiles()

    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        if sr_s != SR:
            wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        feat_src = wav_to_features(wav_src, SR)

        tgt_spk = p["tgt_spk"]
        profile = speaker_profiles.get(tgt_spk)

        if profile is None:
            continue

        reg_src = compute_register(feat_src)
        T = len(reg_src)

        mc_out = np.zeros((T, MC_ORDER + 1), dtype=np.float32)
        codeap_out = np.zeros((T, feat_src["codeap"].shape[1]), dtype=np.float32)

        for t in range(T):
            r = reg_src[t]
            if r in profile["mean_mc"]:
                mc_out[t] = profile["mean_mc"][r]
                codeap_out[t] = profile["mean_codeap"][r]
            else:
                mc_out[t] = profile["global_mean_mc"]
                codeap_out[t] = profile["global_mean_codeap"]

        synth_feat = {
            "f0": feat_src["f0"][:T],
            "mc": mc_out,
            "codeap": codeap_out,
        }
        wav_syn = features_to_wav_mc(synth_feat, SR)

        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_t != SR:
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)
        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def o3_affine_transport(model, pairs, speaker_profiles=None):
    """Affine transport of mel-cepstrum: source→target per register.

    mc_t = μ_t(r) + diag(σ_t(r)/σ_s(r)) * (mc_s - μ_s(r))
    """
    if speaker_profiles is None:
        speaker_profiles = build_speaker_profiles()

    results = []
    for p in pairs:
        wav_src, sr_s = sf.read(p["src_wav"], dtype="float32")
        if sr_s != SR:
            wav_src = librosa.resample(wav_src, orig_sr=sr_s, target_sr=SR)
        feat_src = wav_to_features(wav_src, SR)

        src_spk = p["src_spk"]
        tgt_spk = p["tgt_spk"]
        prof_s = speaker_profiles.get(src_spk)
        prof_t = speaker_profiles.get(tgt_spk)

        if prof_s is None or prof_t is None:
            continue

        reg_src = compute_register(feat_src)
        T = len(reg_src)

        mc_s = feat_src["mc"][:T]
        mc_out = np.zeros_like(mc_s)
        codeap_out = np.zeros((T, feat_src["codeap"].shape[1]), dtype=np.float32)

        for t in range(T):
            r = reg_src[t]
            if r in prof_s["mean_mc"] and r in prof_t["mean_mc"]:
                mu_s = prof_s["mean_mc"][r]
                mu_t = prof_t["mean_mc"][r]
                std_s = prof_s["std_mc"][r] + 1e-4
                std_t = prof_t["std_mc"][r] + 1e-4

                scale = np.clip(std_t / std_s, 0.5, 2.0)
                mc_out[t] = mu_t + scale * (mc_s[t] - mu_s)

                codeap_out[t] = prof_t["mean_codeap"].get(r, prof_t["global_mean_codeap"])
            else:
                mc_out[t] = prof_t["global_mean_mc"]
                codeap_out[t] = prof_t["global_mean_codeap"]

        synth_feat = {
            "f0": feat_src["f0"][:T],
            "mc": mc_out,
            "codeap": codeap_out,
        }
        wav_syn = features_to_wav_mc(synth_feat, SR)

        wav_tgt, sr_t = sf.read(p["tgt_wav"], dtype="float32")
        if sr_t != SR:
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr_t, target_sr=SR)
        cross = compute_cross_secs(model, wav_src, wav_tgt, wav_syn, SR)
        results.append(cross)
    return results


def build_speaker_profiles(n_speakers=30):
    print("Building speaker profiles...")
    spk_dirs = sorted([d for d in VCTK_WAV.iterdir() if d.is_dir()])[:n_speakers]

    profiles = {}
    for spk_dir in spk_dirs:
        spk = spk_dir.name
        all_mc = []
        all_codeap = []
        all_reg = []

        for wav_path in sorted(spk_dir.glob("*.wav"))[:30]:
            try:
                wav, sr = sf.read(str(wav_path), dtype="float32")
                if sr != SR:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
                feat = wav_to_features(wav, SR)
                reg = compute_register(feat)

                all_mc.append(feat["mc"])
                all_codeap.append(feat["codeap"])
                all_reg.append(reg)
            except Exception:
                continue

        if not all_mc:
            continue

        mc_cat = np.concatenate(all_mc, axis=0)
        codeap_cat = np.concatenate(all_codeap, axis=0)
        reg_cat = np.concatenate(all_reg, axis=0)

        mean_mc = {}
        std_mc = {}
        mean_codeap = {}

        for r in np.unique(reg_cat):
            mask = reg_cat == r
            if mask.sum() < 5:
                continue
            mean_mc[int(r)] = mc_cat[mask].mean(axis=0)
            std_mc[int(r)] = mc_cat[mask].std(axis=0)
            mean_codeap[int(r)] = codeap_cat[mask].mean(axis=0)

        profiles[spk] = {
            "mean_mc": mean_mc,
            "std_mc": std_mc,
            "mean_codeap": mean_codeap,
            "global_mean_mc": mc_cat.mean(axis=0),
            "global_mean_codeap": codeap_cat.mean(axis=0),
        }
        print(f"  {spk}: {len(mean_mc)} registers, {mc_cat.shape[0]} frames")

    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=20)
    parser.add_argument("--output", default="results/oracle_sf.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=== Causal Source-Filter VC: Oracle Tests ===\n")
    print("Loading SECS model...")
    secs_model = load_secs_model()

    print(f"Finding {args.n_pairs} same-text pairs...")
    pairs = find_same_text_pairs(args.n_pairs)
    print(f"Found {len(pairs)} pairs")

    all_results = {}

    print("\n--- O4: Self analysis-synthesis (WORLD upper bound) ---")
    t0 = time.time()
    o4 = o4_self_recon(secs_model, pairs)
    print(f"  SECS: {o4.mean():.4f} ± {o4.std():.4f} (took {time.time()-t0:.1f}s)")
    all_results["O4_self_recon"] = {"mean": float(o4.mean()), "std": float(o4.std())}

    print("\n--- O1: Source excitation + Target real envelope ---")
    t0 = time.time()
    o1 = o1_source_excitation_target_envelope(secs_model, pairs)
    tgt_scores = np.array([r["tgt"] for r in o1])
    src_scores = np.array([r["src"] for r in o1])
    print(f"  SECS(tgt): {tgt_scores.mean():.4f} ± {tgt_scores.std():.4f}")
    print(f"  SECS(src): {src_scores.mean():.4f} ± {src_scores.std():.4f}")
    print(f"  (took {time.time()-t0:.1f}s)")
    all_results["O1_synth_tgt_env"] = {
        "secs_tgt_mean": float(tgt_scores.mean()), "secs_tgt_std": float(tgt_scores.std()),
        "secs_src_mean": float(src_scores.mean()), "secs_src_std": float(src_scores.std()),
    }

    print("\n--- Building speaker profiles ---")
    profiles = build_speaker_profiles(n_speakers=40)

    print("\n--- O2: Source F0 + Target mean envelope ---")
    t0 = time.time()
    o2 = o2_mean_envelope(secs_model, pairs, profiles)
    tgt_scores = np.array([r["tgt"] for r in o2])
    src_scores = np.array([r["src"] for r in o2])
    print(f"  SECS(tgt): {tgt_scores.mean():.4f} ± {tgt_scores.std():.4f}")
    print(f"  SECS(src): {src_scores.mean():.4f} ± {src_scores.std():.4f}")
    print(f"  (took {time.time()-t0:.1f}s)")
    all_results["O2_mean_envelope"] = {
        "secs_tgt_mean": float(tgt_scores.mean()), "secs_tgt_std": float(tgt_scores.std()),
        "secs_src_mean": float(src_scores.mean()), "secs_src_std": float(src_scores.std()),
    }

    print("\n--- O3: Affine transport envelope ---")
    t0 = time.time()
    o3 = o3_affine_transport(secs_model, pairs, profiles)
    tgt_scores = np.array([r["tgt"] for r in o3])
    src_scores = np.array([r["src"] for r in o3])
    print(f"  SECS(tgt): {tgt_scores.mean():.4f} ± {tgt_scores.std():.4f}")
    print(f"  SECS(src): {src_scores.mean():.4f} ± {src_scores.std():.4f}")
    print(f"  (took {time.time()-t0:.1f}s)")
    all_results["O3_affine_transport"] = {
        "secs_tgt_mean": float(tgt_scores.mean()), "secs_tgt_std": float(tgt_scores.std()),
        "secs_src_mean": float(src_scores.mean()), "secs_src_std": float(src_scores.std()),
    }

    print("\n=== Summary ===")
    print(f"  O4 (self recon):      {all_results['O4_self_recon']['mean']:.4f}")
    print(f"  O1 (src exc + tgt env): {all_results['O1_synth_tgt_env']['secs_tgt_mean']:.4f}")
    print(f"  O2 (mean envelope):   {all_results['O2_mean_envelope']['secs_tgt_mean']:.4f}")
    print(f"  O3 (affine transport): {all_results['O3_affine_transport']['secs_tgt_mean']:.4f}")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
