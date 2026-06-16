"""
Phase B: Bottleneck autoencoder warm-start.

Trains the `Converter` (residual-prediction variant) as an AutoVC-style
bottleneck autoencoder in DAC latent space. This gives the flow converter
a stable initialization.

Losses:
  - reconstruction L1 (primary)
  - speaker consistency (pred speaker ≈ ref speaker)
  - content preservation (content code should be speaker-invariant)

Roles:
  - reconstruction (60%): z_src → z_src (autoencode)
  - cross_speaker  (40%): z_src(A) + ref(B) → z_src(A) with B's timbre
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from converter import Converter, ConverterConfig


# ---------------------------------------------------------------------------
# Data loading (in-memory, XPU-optimized)
# ---------------------------------------------------------------------------


def load_latent_corpus(data_dir: str, max_frames: int = 400):
    """Load all latents from the corpus, grouped by speaker.

    Returns:
        speakers: dict {speaker_id: list of np.ndarray [latent_dim, T]}
    """
    index_path = Path(data_dir) / "index.tsv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index.tsv in {data_dir}. Run encode_corpus.py first."
        )

    import csv

    speakers = {}
    with open(index_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spk = row["speaker_id"]
            npy_path = row["path"]
            if not os.path.isabs(npy_path):
                npy_path = os.path.join(data_dir, spk, os.path.basename(npy_path))
            if not os.path.exists(npy_path):
                continue
            latent = np.load(npy_path)
            if latent.shape[1] < 30:
                continue
            if latent.shape[1] > max_frames:
                latent = latent[:, :max_frames]
            speakers.setdefault(spk, []).append(latent.astype(np.float32))

    total = sum(len(v) for v in speakers.values())
    print(f"Loaded {total} latents from {len(speakers)} speakers", flush=True)
    return speakers


def make_batch(speakers: dict, batch_size: int, max_frames: int, device: torch.device):
    """Sample a training batch.

    For each item: pick a source utterance and (optionally) a different
    speaker's reference.
    """
    spk_list = list(speakers.keys())
    src_list = []
    ref_list = []
    tgt_list = []

    for _ in range(batch_size):
        src_spk = spk_list[np.random.randint(0, len(spk_list))]
        src_utts = speakers[src_spk]
        src = src_utts[np.random.randint(0, len(src_utts))]
        src_list.append(src)

        # 50% chance: cross-speaker reference
        if np.random.random() < 0.5 and len(spk_list) > 1:
            ref_spk = src_spk
            while ref_spk == src_spk:
                ref_spk = spk_list[np.random.randint(0, len(spk_list))]
            ref_utts = speakers[ref_spk]
            ref = ref_utts[np.random.randint(0, len(ref_utts))]
        else:
            ref = src

        ref_list.append(ref)
        # Target = source (reconstruction) in warm-start
        tgt_list.append(src)

    # Random crop to common length
    min_T_src = min(s.shape[1] for s in src_list)
    min_T_ref = min(r.shape[1] for r in ref_list)
    T = min(min_T_src, max_frames)
    T_ref = min(min_T_ref, max_frames)

    D = src_list[0].shape[0]
    src = torch.zeros(batch_size, D, T)
    ref = torch.zeros(batch_size, D, T_ref)
    tgt = torch.zeros(batch_size, D, T)

    for i in range(batch_size):
        s = src_list[i]
        if s.shape[1] > T:
            start = np.random.randint(0, s.shape[1] - T)
            s = s[:, start : start + T]
        src[i] = torch.from_numpy(s[:, :T])

        r = ref_list[i]
        if r.shape[1] > T_ref:
            start = np.random.randint(0, r.shape[1] - T_ref)
            r = r[:, start : start + T_ref]
        ref[i] = torch.from_numpy(r[:, :T_ref])

        t = tgt_list[i]
        if t.shape[1] > T:
            start = np.random.randint(0, t.shape[1] - T)
            t = t[:, start : start + T]
        tgt[i] = torch.from_numpy(t[:, :T])

    return src.to(device), tgt.to(device), ref.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(config_path, data_dir, output_dir):
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ConverterConfig(**cfg["model"])
    train_cfg = cfg["training"]
    loss_cfg = cfg["losses"]
    max_frames = cfg["data"]["max_utterance_frames"]

    # Device
    configured = train_cfg.get("device", "auto")
    if configured == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(configured)
    print(f"Device: {device}", flush=True)

    # Data
    speakers = load_latent_corpus(data_dir, max_frames)

    # Model
    model = Converter(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Converter parameters: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)

    # Init from checkpoint if specified
    if "init_from" in train_cfg and train_cfg["init_from"]:
        ckpt = torch.load(train_cfg["init_from"], map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Initialized from {train_cfg['init_from']}", flush=True)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg.get("optimizer_betas", [0.8, 0.99])),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=train_cfg.get("lr_scheduler_gamma", 0.9995)
    )

    os.makedirs(output_dir, exist_ok=True)
    batch_size = train_cfg["batch_size"]
    max_steps = train_cfg["max_steps"]
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    model.train()
    losses_log = {"total": [], "recon": [], "spk": [], "content": []}

    print(f"Starting warm-start training for {max_steps} steps...", flush=True)
    for step in range(1, max_steps + 1):
        src, tgt, ref = make_batch(speakers, batch_size, max_frames, device)

        # Precompute target speaker embedding
        with torch.no_grad():
            tgt_embed = model.speaker_embedding(tgt)

        optim.zero_grad()
        pred = model(src, ref)

        # Losses
        loss_recon = F.l1_loss(pred, tgt) * loss_cfg["reconstruction_l1"]

        pred_embed = model.speaker_embedding(pred)
        loss_spk = (
            1.0 - F.cosine_similarity(pred_embed, tgt_embed, dim=-1).mean()
        ) * loss_cfg.get("speaker_consistency", 0.5)

        # Content code should be speaker-invariant
        content_src = model.content_code(src)
        content_tgt = model.content_code(tgt)
        loss_content = F.l1_loss(content_src, content_tgt) * loss_cfg.get(
            "content_preservation", 0.3
        )

        loss = loss_recon + loss_spk + loss_content
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        scheduler.step()

        losses_log["total"].append(loss.item())
        losses_log["recon"].append(loss_recon.item())
        losses_log["spk"].append(loss_spk.item())
        losses_log["content"].append(loss_content.item())

        if step % 100 == 0:
            avg = {k: np.mean(v[-100:]) for k, v in losses_log.items()}
            print(
                f"step {step}/{max_steps} | loss={avg['total']:.4f} "
                f"recon={avg['recon']:.4f} spk={avg['spk']:.4f} "
                f"content={avg['content']:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if step % cfg.get("checkpoint", {}).get("save_every_steps", 5000) == 0:
            ckpt_path = os.path.join(output_dir, f"step_{step:06d}.pt")
            torch.save(
                {"model": model.state_dict(), "step": step, "config": cfg}, ckpt_path
            )
            latest_path = os.path.join(output_dir, "latest.pt")
            torch.save(
                {"model": model.state_dict(), "step": step, "config": cfg}, latest_path
            )

    best_path = os.path.join(output_dir, "best.pt")
    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, best_path)
    print(f"\nWarm-start complete. Final checkpoint: {best_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase B: warm-start training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
