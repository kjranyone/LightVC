"""
Phase C: Mean-flow matching training (the core).

Trains the `FlowConverter` to learn the velocity field that transports
source latents to target-speaker latents. No VC teacher — the target is
a real recording of the target speaker.

Training:
  z_0 = z_src (or timbre-shifted z_src)
  z_tgt = DAC.encode(real_target_speaker_utterance)
  t ~ U[0, 1]
  z_t = (1-t)*z_0 + t*z_tgt
  v_target = z_tgt - z_0
  loss = MSE(v_pred(z_t, t, ref), v_target)

Inference (1-step):
  z_converted = z_src + v_pred(z_src, t=1, ref)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from converter import FlowConverter, ConverterConfig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_latent_corpus(data_dir: str, max_frames: int = 400):
    """Load all latents grouped by speaker."""
    import csv

    index_path = Path(data_dir) / "index.tsv"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.tsv in {data_dir}")

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


def sample_flow_batch(
    speakers: dict, batch_size: int, max_frames: int, device: torch.device
):
    """Sample a flow-matching training batch.

    For each item:
      - Pick source utterance (any speaker, any text)
      - Pick a DIFFERENT target speaker
      - z_tgt = a real utterance from target speaker (any text)
      - z_ref = another real utterance from target speaker (for speaker conditioning)
    """
    spk_list = list(speakers.keys())
    src_list, tgt_list, ref_list = [], [], []

    for _ in range(batch_size):
        src_spk = spk_list[np.random.randint(0, len(spk_list))]
        src_utts = speakers[src_spk]
        src = src_utts[np.random.randint(0, len(src_utts))]
        src_list.append(src)

        # Different target speaker
        tgt_spk = src_spk
        while tgt_spk == src_spk and len(spk_list) > 1:
            tgt_spk = spk_list[np.random.randint(0, len(spk_list))]

        tgt_utts = speakers[tgt_spk]
        # Target utterance (real recording, any text)
        tgt = tgt_utts[np.random.randint(0, len(tgt_utts))]
        tgt_list.append(tgt)

        # Reference utterance (different from target, same speaker)
        if len(tgt_utts) > 1:
            ref = tgt_utts[np.random.randint(0, len(tgt_utts))]
            while ref is tgt:
                ref = tgt_utts[np.random.randint(0, len(tgt_utts))]
        else:
            ref = tgt
        ref_list.append(ref)

    # Crop to common lengths
    T = min(
        min(s.shape[1] for s in src_list), min(t.shape[1] for t in tgt_list), max_frames
    )
    T_ref = min(min(r.shape[1] for r in ref_list), max_frames)

    D = src_list[0].shape[0]
    src = torch.zeros(batch_size, D, T)
    tgt = torch.zeros(batch_size, D, T)
    ref = torch.zeros(batch_size, D, T_ref)

    for i in range(batch_size):
        for dest, data_list, T_len in [
            (src, src_list, T),
            (tgt, tgt_list, T),
            (ref, ref_list, T_ref),
        ]:
            d = data_list[i]
            if d.shape[1] > T_len:
                start = np.random.randint(0, d.shape[1] - T_len)
                d = d[:, start : start + T_len]
            dest[i] = torch.from_numpy(d[:, :T_len])

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

    speakers = load_latent_corpus(data_dir, max_frames)

    model = FlowConverter(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowConverter parameters: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)

    # Init from warm-start checkpoint
    if "init_from" in train_cfg and train_cfg["init_from"]:
        ckpt = torch.load(
            train_cfg["init_from"], map_location=device, weights_only=False
        )
        # Load shared modules (bottleneck, speaker_encoder, blocks)
        # FlowConverter has different keys than Converter, so load with strict=False
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"Initialized from {train_cfg['init_from']} "
            f"(missing: {len(missing)}, unexpected: {len(unexpected)})",
            flush=True,
        )

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg.get("optimizer_betas", [0.8, 0.99])),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=train_cfg.get("lr_scheduler_gamma", 0.9998)
    )

    os.makedirs(output_dir, exist_ok=True)
    batch_size = train_cfg["batch_size"]
    max_steps = train_cfg["max_steps"]
    grad_clip = train_cfg.get("gradient_clip", 1.0)

    # AMP for XPU
    use_amp = train_cfg.get("mixed_precision") == "bf16"
    scaler = None  # GradScaler not safe on Arc A-series (no FP64)

    model.train()
    losses_log = {"total": [], "fm": [], "l1": [], "spk": [], "content": []}

    print(f"Starting flow matching training for {max_steps} steps...", flush=True)
    for step in range(1, max_steps + 1):
        z_src, z_tgt, z_ref = sample_flow_batch(
            speakers, batch_size, max_frames, device
        )

        B = z_src.shape[0]

        # Source-conditioned flow: z_0 = z_src (not noise)
        z_0 = z_src

        # Sample time t ~ U[0, 1]
        t = torch.rand(B, device=device)

        # Interpolation: z_t = (1-t)*z_0 + t*z_tgt
        t_expand = t[:, None, None]  # [B, 1, 1]
        z_t = (1.0 - t_expand) * z_0 + t_expand * z_tgt

        # Target velocity (constant for linear flow)
        v_target = z_tgt - z_0

        # Predict velocity
        optim.zero_grad()

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                v_pred = model.forward_velocity(z_t, t, z_ref)
                loss_fm = F.mse_loss(v_pred, v_target) * loss_cfg["fm_velocity"]
        else:
            v_pred = model.forward_velocity(z_t, t, z_ref)
            loss_fm = F.mse_loss(v_pred, v_target) * loss_cfg["fm_velocity"]

        # Endpoint L1 (latent at t=1)
        z_pred_end = z_0 + v_pred  # one-step estimate
        loss_l1 = F.l1_loss(z_pred_end, z_tgt) * loss_cfg.get("latent_l1", 2.0)

        # Speaker similarity
        with torch.no_grad():
            tgt_embed = model.speaker_embedding(z_tgt)
        pred_embed = model.speaker_embedding(z_pred_end)
        loss_spk = (
            1.0 - F.cosine_similarity(pred_embed, tgt_embed, dim=-1).mean()
        ) * loss_cfg.get("speaker_sim", 1.0)

        # Content preservation
        content_src = model.bottleneck(z_src)
        content_pred = model.bottleneck(z_pred_end.detach())
        loss_content = F.l1_loss(content_pred, content_src.detach()) * loss_cfg.get(
            "content_inv", 0.5
        )

        loss = loss_fm + loss_l1 + loss_spk + loss_content

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        scheduler.step()

        losses_log["total"].append(loss.item())
        losses_log["fm"].append(loss_fm.item())
        losses_log["l1"].append(loss_l1.item())
        losses_log["spk"].append(loss_spk.item())
        losses_log["content"].append(loss_content.item())

        if step % 100 == 0:
            avg = {k: np.mean(v[-100:]) for k, v in losses_log.items()}
            print(
                f"step {step}/{max_steps} | loss={avg['total']:.4f} "
                f"fm={avg['fm']:.4f} l1={avg['l1']:.4f} "
                f"spk={avg['spk']:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )

        if step % cfg.get("checkpoint", {}).get("save_every_steps", 10000) == 0:
            ckpt_path = os.path.join(output_dir, f"step_{step:06d}.pt")
            torch.save(
                {"model": model.state_dict(), "step": step, "config": cfg}, ckpt_path
            )
            latest_path = os.path.join(output_dir, "latest.pt")
            torch.save(
                {"model": model.state_dict(), "step": step, "config": cfg}, latest_path
            )
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    best_path = os.path.join(output_dir, "best.pt")
    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, best_path)
    print(
        f"\nFlow matching training complete. Final checkpoint: {best_path}", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description="Phase C: flow matching training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
