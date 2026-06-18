import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from train_flow import load_latent_corpus, sample_flow_batch

speakers = load_latent_corpus("data/vctk_latents_200/", max_frames=200)
spk_list = list(speakers.keys())

print("Computing speaker means...")
spk_means = {}
for spk, utts in speakers.items():
    all_latents = np.stack([u[0].mean(axis=1) for u in utts])  # [N, D]
    spk_means[spk] = all_latents.mean(axis=0)  # [D]

print(f"Speaker means computed for {len(spk_means)} speakers")

all_means = np.stack(list(spk_means.values()))  # [N_spk, D]
print(f"Speaker mean stats: mean={all_means.mean():.4f} std={all_means.std():.4f}")
print(f"  Between-speaker variance: {all_means.var(axis=0).mean():.4f}")

within_var = []
for spk, utts in speakers.items():
    utt_means = np.stack([u[0].mean(axis=1) for u in utts])  # [N, D]
    within_var.append(utt_means.var(axis=0).mean())
print(f"  Within-speaker variance:  {np.mean(within_var):.4f}")
print(f"  Ratio (between/within):   {all_means.var(axis=0).mean() / np.mean(within_var):.2f}")

device = torch.device("cuda")
spk_means_t = {spk: torch.from_numpy(m).float().to(device) for spk, m in spk_means.items()}

src, tgt, ref, src_spk = sample_flow_batch(speakers, 8, 200, device)

tgt_spks = []
spk_list = list(speakers.keys())
for i in range(8):
    s = src_spk[i]
    t_spk = s
    while t_spk == s:
        t_spk = spk_list[np.random.randint(0, len(spk_list))]
    tgt_spks.append(t_spk)

v_full = tgt - src
v_mean_shift = torch.stack([spk_means_t[tgt_spks[i]] - spk_means_t[src_spk[i]] for i in range(8)])
v_mean_shift = v_mean_shift.unsqueeze(-1).expand_as(v_full)

print(f"\n=== Velocity comparison ===")
print(f"v_full (z_tgt-z_src):     std={v_full.std():.4f} abs_mean={v_full.abs().mean():.4f}")
print(f"v_mean_shift (spk diff):  std={v_mean_shift.std():.4f} abs_mean={v_mean_shift.abs().mean():.4f}")

residual = v_full - v_mean_shift
print(f"v_residual (content etc): std={residual.std():.4f} abs_mean={residual.abs().mean():.4f}")

fm_full = F.mse_loss(torch.zeros_like(v_full), v_full)
fm_shift = F.mse_loss(torch.zeros_like(v_mean_shift), v_mean_shift)
fm_resid = F.mse_loss(torch.zeros_like(residual), residual)
print(f"\nMSE baseline (pred=0):")
print(f"  v_full:     {fm_full:.4f}")
print(f"  v_shift:    {fm_shift:.4f}")
print(f"  v_residual: {fm_resid:.4f}")
print(f"  shift/full: {fm_shift/fm_full:.1%} of variance is speaker-mean shift")

corr = torch.corrcoef(torch.stack([v_full.flatten(), v_mean_shift.flatten()]))[0, 1]
print(f"\nCorrelation(v_full, v_mean_shift): {corr:.4f}")
