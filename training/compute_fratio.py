"""Compute per-coefficient F-ratio for mel-cepstrum speaker discrimination."""
import numpy as np
from pathlib import Path
from collections import defaultdict

MC_CACHE = Path("data/mc_cache")

spk_mcs = defaultdict(list)
for spk_dir in sorted(MC_CACHE.iterdir()):
    if not spk_dir.is_dir():
        continue
    spk = spk_dir.name
    for npz_path in spk_dir.glob("*.npz"):
        try:
            d = np.load(npz_path)
            mc = d["mc"]
            if len(mc) >= 20:
                spk_mcs[spk].append(mc.astype(np.float32))
        except:
            continue

n_coef = 25
spk_means = []
all_data = []
for spk, mcs in spk_mcs.items():
    cat = np.concatenate(mcs, axis=0)
    spk_means.append(cat.mean(axis=0))
    all_data.append(cat)

spk_means = np.array(spk_means)
all_cat = np.concatenate(all_data, axis=0)
global_mean = all_cat.mean(axis=0)

between_var = np.var(spk_means, axis=0)
within_var = np.zeros(n_coef)
for spk, mcs in spk_mcs.items():
    cat = np.concatenate(mcs, axis=0)
    within_var += np.sum((cat - cat.mean(axis=0))**2, axis=0)
within_var /= len(all_cat)

f_ratio = between_var / (within_var + 1e-8)
weights = f_ratio / f_ratio.mean()

print("Mel-cepstrum coefficient F-ratios:")
print(f"{'Coef':>4} {'F-ratio':>10} {'Weight':>10}")
for i in range(n_coef):
    bar = '#' * int(f_ratio[i] / f_ratio.max() * 40)
    print(f"{i:4d} {f_ratio[i]:10.4f} {weights[i]:10.4f}  {bar}")

print(f"\nWeights saved to data/mc_fratio_weights.npy")
np.save("data/mc_fratio_weights.npy", weights.astype(np.float32))
