"""
Compute per-speaker mel-cepstrum profiles for affine transport.

For each speaker:
  - Per-register: μ(r), σ(r) where r = (pitch_bin, energy_bin, vuv)
  - Global: μ_global, σ_global

Output: data/speaker_profiles.pkl
"""
import sys, os, pickle, time
from pathlib import Path
from collections import defaultdict

import numpy as np

MC_CACHE = Path("data/mc_cache")
N_PITCH_BINS = 8
N_ENERGY_BINS = 3
OUTPUT = "data/speaker_profiles.pkl"


def compute_register(f0, mc, vuv, n_pitch=N_PITCH_BINS, n_energy=N_ENERGY_BINS):
    fmin, fmax = 60, 500
    f0_clipped = np.clip(np.where(f0 > 0, f0, 200), fmin, fmax)
    semitone = 12 * np.log2(f0_clipped / fmin)
    pb = np.clip((semitone / (12 * np.log2(fmax / fmin)) * n_pitch).astype(int), 0, n_pitch - 1)

    energy = mc[:, 0]
    e_min, e_max = np.percentile(energy, 5), np.percentile(energy, 95)
    eb = np.clip(((energy - e_min) / (e_max - e_min + 1e-6) * n_energy).astype(int), 0, n_energy - 1)

    return pb * n_energy * 2 + eb * 2 + vuv.astype(int)


def main():
    print("Computing speaker profiles...")
    spk_dirs = sorted([d for d in MC_CACHE.iterdir() if d.is_dir()])
    print(f"  {len(spk_dirs)} speakers")

    profiles = {}
    t0 = time.time()
    for sid, spk_dir in enumerate(spk_dirs):
        spk = spk_dir.name
        all_mc, all_reg = [], []

        for npz_path in spk_dir.glob("*.npz"):
            try:
                d = np.load(npz_path)
                mc = d["mc"]
                if len(mc) < 20:
                    continue
                reg = compute_register(d["f0"], mc, d["vuv"])
                all_mc.append(mc)
                all_reg.append(reg)
            except:
                continue

        if not all_mc:
            continue

        mc_cat = np.concatenate(all_mc, axis=0)
        reg_cat = np.concatenate(all_reg, axis=0)

        mean_dict, std_dict = {}, {}
        for r in np.unique(reg_cat):
            mask = reg_cat == r
            if mask.sum() < 5:
                continue
            mean_dict[int(r)] = mc_cat[mask].mean(axis=0).astype(np.float32)
            std_dict[int(r)] = (mc_cat[mask].std(axis=0) + 1e-6).astype(np.float32)

        profiles[spk] = {
            "mean": mean_dict,
            "std": std_dict,
            "global_mean": mc_cat.mean(axis=0).astype(np.float32),
            "global_std": (mc_cat.std(axis=0) + 1e-6).astype(np.float32),
        }

        if (sid + 1) % 20 == 0:
            print(f"  {sid+1}/{len(spk_dirs)} speakers ({time.time()-t0:.0f}s)", flush=True)

    with open(OUTPUT, "wb") as f:
        pickle.dump(profiles, f)

    total_regs = sum(len(p["mean"]) for p in profiles.values())
    print(f"\nDone: {len(profiles)} speakers, avg {total_regs/len(profiles):.1f} registers/speaker")
    print(f"Saved to {OUTPUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
