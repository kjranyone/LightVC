"""
Causal Source-Filter VC — Affine Transport + Bounded Residual.

Architecture (user's original design):
  1. Affine transport (deterministic):
     mc_transport = μ_t(r) + scale(r) * (mc_s - μ_s(r))
     where scale = clip(σ_t(r) / σ_s(r), 0.5, 2.0)

  2. Neural residual Rθ (bounded):
     delta = ε * tanh(Rθ(mc_s, f0, energy, target_emb) / ε)
     
  3. Output:
     mc_t = mc_transport + delta

Training: DTW-aligned same-text pairs
  residual_target = mc_tgt_aligned - mc_transport
  loss = L1(Rθ(mc_s, f0, target_emb), residual_target)

The transport handles the bulk speaker shift.
The residual learns frame-level corrections.
"""
import sys, os, json, time, pickle, argparse, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda")
MC_DIM = 25
N_PITCH_BINS = 8
N_ENERGY_BINS = 3
PAIRS_DIR = Path("data/sf_pairs")
MC_CACHE = Path("data/mc_cache")


def compute_register(f0, mc, vuv, n_pitch=N_PITCH_BINS, n_energy=N_ENERGY_BINS):
    fmin, fmax = 60, 500
    f0_clipped = np.clip(np.where(f0 > 0, f0, 200), fmin, fmax)
    semitone = 12 * np.log2(f0_clipped / fmin)
    pb = np.clip((semitone / (12 * np.log2(fmax / fmin)) * n_pitch).astype(int), 0, n_pitch - 1)
    energy = mc[:, 0] if mc.ndim == 2 else mc
    e_min, e_max = np.percentile(energy, 5), np.percentile(energy, 95)
    eb = np.clip(((energy - e_min) / (e_max - e_min + 1e-6) * n_energy).astype(int), 0, n_energy - 1)
    return pb * n_energy * 2 + eb * 2 + vuv.astype(int)


def apply_transport(mc_src, f0_src, vuv_src, prof_s, prof_t):
    """Affine transport on ALL coefficients."""
    reg = compute_register(f0_src, mc_src, vuv_src)
    mc_out = np.zeros_like(mc_src)
    for k in range(len(mc_src)):
        r = int(reg[k])
        if r in prof_s["mean"] and r in prof_t["mean"]:
            mu_s = prof_s["mean"][r]
            mu_t = prof_t["mean"][r]
            sigma_s = prof_s["std"][r]
            sigma_t = prof_t["std"][r]
            scale = np.clip(sigma_t / sigma_s, 0.5, 2.0)
            mc_out[k] = mu_t + scale * (mc_src[k] - mu_s)
        else:
            mc_out[k] = prof_t["global_mean"].copy()
    return mc_out.astype(np.float32)


class ResidualConverter(nn.Module):
    """Bounded residual network for frame-level envelope corrections."""

    def __init__(self, mc_dim=MC_DIM, feat_dim=27, spk_dim=768, hidden=128,
                 n_blocks=4, kernel_size=7, epsilon=0.5, causal=True):
        super().__init__()
        self.epsilon = epsilon
        self.causal = causal

        self.proj_in = nn.Conv1d(feat_dim, hidden, 1)
        self.spk_proj = nn.Linear(spk_dim, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.film.bias.data[:hidden] = 1.0
        self.film.bias.data[hidden:] = 0.0

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** (i % 4)
            pad = (kernel_size - 1) * dilation if causal else (kernel_size - 1) * dilation // 2
            self.blocks.append(nn.ModuleDict({
                "conv1": nn.Conv1d(hidden, hidden, kernel_size, dilation=dilation, padding=pad),
                "conv2": nn.Conv1d(hidden, hidden, 1),
            }))

        self.proj_out = nn.Linear(hidden, mc_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, mc_src, feat_extra, spk_emb):
        T = mc_src.shape[-1]
        x = torch.cat([mc_src, feat_extra], dim=1)
        h = self.proj_in(x)
        s = self.spk_proj(spk_emb)
        gb = self.film(F.gelu(s))
        gamma, beta = gb.chunk(2, dim=-1)
        h = h * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

        for block in self.blocks:
            res = h
            h = F.gelu(block["conv1"](h))
            if self.causal:
                h = h[:, :, :T]
            h = block["conv2"](h)
            h = h + res

        raw = self.proj_out(h.transpose(1, 2)).transpose(1, 2)
        delta = self.epsilon * torch.tanh(raw / self.epsilon)
        return delta


class SFTransportDataset:
    def __init__(self, profiles, spk_emb, pairs_dir=PAIRS_DIR, max_len=400):
        self.profiles = profiles
        self.spk_emb = spk_emb
        self.max_len = max_len
        self.pair_files = sorted(pairs_dir.glob("pair_*.npz"))
        print(f"  Loaded {len(self.pair_files)} pre-computed pairs")

    def sample(self):
        npz_path = random.choice(self.pair_files)
        data = np.load(npz_path, allow_pickle=True)

        mc_src = data["mc_src"]
        mc_tgt = data["mc_tgt"]
        f0_src = data["f0_src"]
        codeap_tgt = data["codeap_tgt"]
        spk_src = str(data["spk_src"])
        spk_tgt = str(data["spk_tgt"])

        energy_src = mc_src[:, 0].copy()
        vuv_src = (f0_src > 0).astype(np.float32)

        prof_s = self.profiles.get(spk_src)
        prof_t = self.profiles.get(spk_tgt)
        if prof_s is None or prof_t is None:
            return self.sample()

        mc_transport = apply_transport(mc_src, f0_src, vuv_src, prof_s, prof_t)
        delta_target = (mc_tgt - mc_transport).astype(np.float32)
        delta_target[:, 0:2] = 0.0

        T = min(len(mc_src), self.max_len)
        start = random.randint(0, max(0, len(mc_src) - T)) if len(mc_src) > T else 0

        feat_extra = np.stack([f0_src, energy_src], axis=-1).astype(np.float32)

        return {
            "mc_src": torch.from_numpy(mc_src[start:start+T]).float(),
            "feat_extra": torch.from_numpy(feat_extra[start:start+T]).float(),
            "mc_transport": torch.from_numpy(mc_transport[start:start+T]).float(),
            "delta_target": torch.from_numpy(delta_target[start:start+T]).float(),
            "f0_src": torch.from_numpy(f0_src[start:start+T]).float(),
            "codeap_tgt": torch.from_numpy(codeap_tgt[start:start+T]).float(),
            "spk_src": spk_src,
            "spk_tgt": spk_tgt,
        }

    def collate(self, batch):
        min_t = min(item["mc_src"].shape[0] for item in batch)
        return {
            "mc_src": torch.stack([b["mc_src"][:min_t] for b in batch]).to(DEVICE).transpose(1, 2),
            "feat_extra": torch.stack([b["feat_extra"][:min_t] for b in batch]).to(DEVICE).transpose(1, 2),
            "mc_transport": torch.stack([b["mc_transport"][:min_t] for b in batch]).to(DEVICE).transpose(1, 2),
            "delta_target": torch.stack([b["delta_target"][:min_t] for b in batch]).to(DEVICE).transpose(1, 2),
            "spk_tgt_emb": torch.stack([self.spk_emb[b["spk_tgt"]] for b in batch]).to(DEVICE),
        }


def load_speaker_embeddings():
    cache_path = Path("data/wavlm_sv_embeddings.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    spk_avg = {}
    for key, emb in cache.items():
        spk = key.split("/")[0]
        spk_avg.setdefault(spk, []).append(emb)
    return {spk: torch.from_numpy(np.mean(embs, axis=0)).float() for spk, embs in spk_avg.items()}


def load_profiles():
    with open("data/speaker_profiles.pkl", "rb") as f:
        return pickle.load(f)


def train(args):
    print("=== SF-VC: Transport + Bounded Residual ===\n")

    print("Loading speaker profiles...")
    profiles = load_profiles()
    print(f"  {len(profiles)} speakers")

    print("Loading speaker embeddings...")
    spk_emb = load_speaker_embeddings()
    spk_dim = next(iter(spk_emb.values())).shape[0]

    print("Loading pairs...")
    dataset = SFTransportDataset(profiles, spk_emb, max_len=400)

    delta_mags = []
    for _ in range(50):
        s = dataset.sample()
        delta_mags.append(s["delta_target"].abs().mean().item())
    print(f"  Delta target L1 magnitude: {np.mean(delta_mags):.4f} (ε={args.epsilon})")

    model = ResidualConverter(
        spk_dim=spk_dim, hidden=args.hidden, n_blocks=args.n_blocks,
        epsilon=args.epsilon, causal=not args.non_causal,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResidualConverter: {n_params:,} ({n_params/1e6:.2f}M), ε={args.epsilon}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99998)

    os.makedirs(args.output, exist_ok=True)
    losses = []

    print(f"\nTraining {args.max_steps} steps (B={args.batch_size})...")
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        batch_items = [dataset.sample() for _ in range(args.batch_size)]
        batch = dataset.collate(batch_items)

        optim.zero_grad()
        delta_pred = model(batch["mc_src"], batch["feat_extra"], batch["spk_tgt_emb"])
        loss = F.l1_loss(delta_pred, batch["delta_target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 100 == 0:
            avg = np.mean(losses[-100:])
            elapsed = time.time() - t0
            sps = step / elapsed
            eta = (args.max_steps - step) / sps
            print(f"step {step}/{args.max_steps} | delta_l1={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} | {sps:.1f}/s ETA {eta:.0f}s", flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            path = os.path.join(args.output, f"step_{step:06d}.pt")
            ckpt = {
                "model": model.state_dict(), "step": step,
                "config": {
                    "spk_dim": spk_dim, "hidden": args.hidden,
                    "n_blocks": args.n_blocks, "epsilon": args.epsilon,
                    "causal": not args.non_causal,
                },
            }
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(args.output, "latest.pt"))
            print(f"  Saved: {path}", flush=True)

    print(f"\nDone. Final delta L1: {np.mean(losses[-100:]):.4f}")


def evaluate(args):
    import soundfile as sf
    import pyworld as world
    import pysptk as sptk
    import librosa

    print("\n=== Transport + Residual Evaluation ===\n")

    profiles = load_profiles()
    spk_emb = load_speaker_embeddings()
    spk_dim = next(iter(spk_emb.values())).shape[0]

    f0_stats = {}
    for spk_dir in sorted(MC_CACHE.iterdir()):
        if not spk_dir.is_dir(): continue
        spk = spk_dir.name
        f0s = []
        for npz_path in spk_dir.glob("*.npz"):
            d = np.load(npz_path)
            v = d["f0"][d["f0"] > 0]
            if len(v) > 0: f0s.extend(v.tolist())
        if f0s:
            f0_stats[spk] = float(np.exp(np.mean(np.log(np.array(f0s)))))

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = ResidualConverter(
        spk_dim=cfg["spk_dim"], hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
        epsilon=cfg["epsilon"], causal=cfg.get("causal", True),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    VCTK_WAV = Path("../data/vctk_200")
    pair_files = sorted(PAIRS_DIR.glob("pair_*.npz"))[:args.n_eval]

    results = {"transport_only": [], "transport_residual": [], "src": []}

    for i, npz_path in enumerate(pair_files):
        data = np.load(npz_path, allow_pickle=True)
        mc_src = data["mc_src"]
        mc_tgt = data["mc_tgt"]
        f0_src = data["f0_src"]
        codeap_tgt = data["codeap_tgt"]
        spk_src = str(data["spk_src"])
        spk_tgt = str(data["spk_tgt"])

        prof_s = profiles.get(spk_src)
        prof_t = profiles.get(spk_tgt)
        if prof_s is None or prof_t is None: continue

        energy_src = mc_src[:, 0].copy()
        vuv_src = (f0_src > 0).astype(np.float32)

        mc_transport = apply_transport(mc_src, f0_src, vuv_src, prof_s, prof_t)

        T = min(len(mc_src), 800)
        mc_src_t = torch.from_numpy(mc_src[:T]).float().unsqueeze(0).to(DEVICE).transpose(1, 2)
        feat_extra = torch.from_numpy(np.stack([f0_src, energy_src], axis=-1)[:T]).float().unsqueeze(0).to(DEVICE).transpose(1, 2)
        spk_t = spk_emb[spk_tgt].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            delta_pred = model(mc_src_t, feat_extra, spk_t)

        mc_transport_t = mc_transport[:T]
        mc_residual = mc_transport_t + delta_pred.squeeze(0).cpu().numpy().T

        tgt_mean_f0 = f0_stats.get(spk_tgt, 200.0)
        voiced = f0_src[f0_src > 0]
        src_mean_f0 = float(np.exp(np.mean(np.log(voiced)))) if len(voiced) > 0 else 200.0
        ratio = tgt_mean_f0 / src_mean_f0 if src_mean_f0 > 0 else 1.0
        f0_shifted = np.where(f0_src[:T] > 0, f0_src[:T] * ratio, 0).astype(np.float64)

        def synth(mc_array):
            mc64 = np.ascontiguousarray(mc_array, dtype=np.float64)
            sp = sptk.mc2sp(mc64, 0.410, 2048)
            codeap = np.ascontiguousarray(codeap_tgt[:T], dtype=np.float64)
            ap = world.decode_aperiodicity(codeap, 16000, 2048) if codeap.shape[1] > 0 else np.ones_like(sp)
            return world.synthesize(f0_shifted[:T], sp, ap, 16000, frame_period=5.0).astype(np.float32)

        wav_transport = synth(mc_transport_t)
        wav_residual = synth(mc_residual)

        src_wavs = list((VCTK_WAV / spk_src).glob("*.wav"))
        tgt_wavs = list((VCTK_WAV / spk_tgt).glob("*.wav"))
        if not src_wavs or not tgt_wavs: continue

        wav_src, sr = sf.read(str(src_wavs[0]), dtype="float32")
        wav_tgt, sr = sf.read(str(tgt_wavs[0]), dtype="float32")
        if sr != 16000:
            wav_src = librosa.resample(wav_src, orig_sr=sr, target_sr=16000)
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr, target_sr=16000)

        with torch.no_grad():
            e_src = secs_model.encode_batch(torch.from_numpy(wav_src.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = secs_model.encode_batch(torch.from_numpy(wav_tgt.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tr = secs_model.encode_batch(torch.from_numpy(wav_transport.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_re = secs_model.encode_batch(torch.from_numpy(wav_residual.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)

        tr_tgt = F.cosine_similarity(e_tgt, e_tr, dim=-1).item()
        re_tgt = F.cosine_similarity(e_tgt, e_re, dim=-1).item()
        re_src = F.cosine_similarity(e_src, e_re, dim=-1).item()

        results["transport_only"].append(tr_tgt)
        results["transport_residual"].append(re_tgt)
        results["src"].append(re_src)
        print(f"  [{i+1}/{len(pair_files)}] {spk_src}→{spk_tgt}: transport={tr_tgt:.3f} +residual={re_tgt:.3f} src={re_src:.3f}", flush=True)

    tr_arr = np.array(results["transport_only"])
    re_arr = np.array(results["transport_residual"])
    src_arr = np.array(results["src"])

    print(f"\n=== Results ===")
    print(f"Transport only:     SECS(tgt) = {tr_arr.mean():.4f} ± {tr_arr.std():.4f}")
    print(f"Transport+Residual: SECS(tgt) = {re_arr.mean():.4f} ± {re_arr.std():.4f}")
    print(f"Transport+Residual: SECS(src) = {src_arr.mean():.4f} ± {src_arr.std():.4f}")
    print(f"Residual improvement: {re_arr.mean() - tr_arr.mean():+.4f}")

    out = {
        "transport_only_tgt": float(tr_arr.mean()),
        "transport_residual_tgt": float(re_arr.mean()),
        "transport_residual_src": float(src_arr.mean()),
        "improvement": float(re_arr.mean() - tr_arr.mean()),
        "epsilon": cfg["epsilon"],
    }
    with open(os.path.join(os.path.dirname(args.checkpoint), "eval_transport.json"), "w") as f:
        json.dump(out, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    tr_p = sub.add_parser("train")
    tr_p.add_argument("--output", default="checkpoints/sf_transport")
    tr_p.add_argument("--batch_size", type=int, default=16)
    tr_p.add_argument("--lr", type=float, default=5e-4)
    tr_p.add_argument("--max_steps", type=int, default=30000)
    tr_p.add_argument("--save_every", type=int, default=5000)
    tr_p.add_argument("--hidden", type=int, default=128)
    tr_p.add_argument("--n_blocks", type=int, default=4)
    tr_p.add_argument("--epsilon", type=float, default=0.5)
    tr_p.add_argument("--non_causal", action="store_true")
    ev_p = sub.add_parser("eval")
    ev_p.add_argument("--checkpoint", required=True)
    ev_p.add_argument("--n_eval", type=int, default=30)
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
