"""
FlowConverter distillation with kNN-VC targets (WavLM-matched).

For each training step:
1. Sample (source, reference) pair — different speakers
2. Load pre-cached WavLM L6 features for both
3. kNN match: find k=4 nearest ref frames per source frame (in WavLM space)
4. Build z_target: average matched reference DAC latent frames
5. Train FlowConverter: v_pred = model(z_src, t, z_ref)
6. loss = L1(z_src + v_pred, z_target) + content_preservation + MSE(v_pred, z_target - z_src)

The key insight: z_target has SAME content as z_src (matched by WavLM phonetic
similarity) but TARGET speaker (all frames from reference). So
v_target = z_target - z_src is a clean speaker-only transformation — no content noise.

At inference: source WAV → DAC encode → FlowConverter → DAC decode → output.
No WavLM needed at inference. Pure DAC pipeline.
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


def load_corpus(data_dir, wavlm_dir, max_frames=200, min_frames=30):
    """Load index of DAC latents + WavLM L6 feature paths indexed by speaker."""
    import csv
    speakers = {}
    index_path = Path(data_dir) / "index.tsv"
    with open(index_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            spk = row["speaker_id"]
            utt_id = row["utterance_id"]
            npy_path = row["path"]
            wlm_path = Path(wavlm_dir) / spk / f"{utt_id}.npy"
            if not Path(npy_path).exists() or not wlm_path.exists():
                continue
            speakers.setdefault(spk, []).append({
                "latent_path": npy_path,
                "wlm_path": str(wlm_path),
                "utt_id": utt_id,
            })
    total = sum(len(v) for v in speakers.values())
    print(f"Loaded index: {total} utterances from {len(speakers)} speakers", flush=True)
    return speakers


def sample_knn_batch(speakers, batch_size, max_frames, device, k=4):
    """Sample a batch and compute kNN-VC targets."""
    spk_list = sorted(speakers.keys())
    src_latents, ref_latents, tgt_latents = [], [], []

    for _ in range(batch_size):
        src_spk = spk_list[np.random.randint(0, len(spk_list))]
        tgt_spk = src_spk
        while tgt_spk == src_spk and len(spk_list) > 1:
            tgt_spk = spk_list[np.random.randint(0, len(spk_list))]

        src_item = speakers[src_spk][np.random.randint(0, len(speakers[src_spk]))]
        ref_item = speakers[tgt_spk][np.random.randint(0, len(speakers[tgt_spk]))]

        z_src = np.load(src_item["latent_path"]).astype(np.float32)
        w_src = np.load(src_item["wlm_path"]).astype(np.float32)
        z_ref = np.load(ref_item["latent_path"]).astype(np.float32)
        w_ref = np.load(ref_item["wlm_path"]).astype(np.float32)

        T_src = min(z_src.shape[1], max_frames)
        z_src = z_src[:, :T_src]

        T_wlm_src = min(w_src.shape[0], int(T_src / 86 * 50) + 1)
        w_src_t = w_src[:T_wlm_src]

        T_ref_dac = z_ref.shape[1]
        T_ref_wlm = w_ref.shape[0]
        dac_per_wlm = T_ref_dac / max(T_ref_wlm, 1)

        w_src_norm = w_src_t / (np.linalg.norm(w_src_t, axis=-1, keepdims=True) + 1e-8)
        w_ref_norm = w_ref / (np.linalg.norm(w_ref, axis=-1, keepdims=True) + 1e-8)
        sim = w_src_norm @ w_ref_norm.T

        topk_idx = np.argpartition(-sim, min(k, sim.shape[1]-1), axis=1)[:, :k]

        z_target = np.zeros_like(z_src)
        for t_dac in range(T_src):
            t_wlm = min(int(t_dac / 86 * 50), T_wlm_src - 1)
            matched_wlm = topk_idx[t_wlm]
            matched_dac = np.clip((matched_wlm * dac_per_wlm).astype(int), 0, T_ref_dac - 1)
            z_target[:, t_dac] = z_ref[:, matched_dac].mean(axis=-1)

        src_latents.append(z_src)
        ref_latents.append(z_ref[:, :max_frames] if z_ref.shape[1] > max_frames else z_ref)
        tgt_latents.append(z_target)

    min_t_src = min(s.shape[1] for s in src_latents)
    min_t_ref = min(r.shape[1] for r in ref_latents)

    src_batch = torch.zeros(batch_size, 1024, min_t_src)
    ref_batch = torch.zeros(batch_size, 1024, min_t_ref)
    tgt_batch = torch.zeros(batch_size, 1024, min_t_src)

    for i in range(batch_size):
        src_batch[i] = torch.from_numpy(src_latents[i][:, :min_t_src])
        ref_batch[i] = torch.from_numpy(ref_latents[i][:, :min_t_ref])
        tgt_batch[i] = torch.from_numpy(tgt_latents[i][:, :min_t_src])

    return src_batch.to(device), ref_batch.to(device), tgt_batch.to(device)


def train(config_path, data_dir, output_dir):
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ConverterConfig(**cfg["model"])
    train_cfg = cfg["training"]
    loss_cfg = cfg["losses"]
    max_frames = cfg["data"]["max_utterance_frames"]
    min_frames = cfg["data"].get("min_utterance_frames", 30)

    wavlm_dir = cfg["data"].get("wavlm_dir", "data/wavlm_l6")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    speakers = load_corpus(data_dir, wavlm_dir, max_frames, min_frames)
    spk_list = sorted(speakers.keys())

    model = FlowConverter(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowConverter parameters: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)

    if "init_from" in train_cfg and train_cfg["init_from"]:
        ckpt = torch.load(train_cfg["init_from"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Initialized from {train_cfg['init_from']}", flush=True)

    cold_keywords = {"vel_proj", "vel_heads", "timbre", "cond_mlp", "time_embed"}
    warm_params, cold_params = [], []
    for name, param in model.named_parameters():
        if any(kw in name for kw in cold_keywords):
            cold_params.append(param)
        else:
            warm_params.append(param)
    cold_lr = train_cfg.get("cold_lr", train_cfg["learning_rate"] * 10)
    optim = torch.optim.AdamW(
        [
            {"params": warm_params, "lr": train_cfg["learning_rate"]},
            {"params": cold_params, "lr": cold_lr},
        ],
        betas=tuple(train_cfg.get("optimizer_betas", [0.8, 0.99])),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optim, gamma=train_cfg.get("lr_scheduler_gamma", 0.9999)
    )

    os.makedirs(output_dir, exist_ok=True)
    batch_size = train_cfg["batch_size"]
    max_steps = train_cfg["max_steps"]
    grad_clip = train_cfg.get("gradient_clip", 10.0)
    k = train_cfg.get("knn_k", 4)

    l1_weight = loss_cfg.get("latent_l1", 1.0)
    fm_weight = loss_cfg.get("fm_velocity", 0.5)
    content_weight = loss_cfg.get("content_inv", 1.0)

    model.train()
    losses_log = {"total": [], "l1": [], "fm": [], "content": []}

    print(f"Starting kNN distillation training for {max_steps} steps (k={k})...", flush=True)

    for step in range(1, max_steps + 1):
        z_src, z_ref, z_target = sample_knn_batch(
            speakers, batch_size, max_frames, device, k=k
        )
        B = z_src.shape[0]
        v_target = z_target - z_src

        optim.zero_grad()

        t = torch.rand(B, device=device)
        t_expand = t[:, None, None]
        z_t = (1 - t_expand) * z_src + t_expand * z_target
        v_pred = model.forward_velocity(z_t, t, z_ref)

        z_pred_end = z_src + v_pred

        loss_l1 = F.l1_loss(z_pred_end, z_target) * l1_weight
        loss_fm = F.mse_loss(v_pred, v_target) * fm_weight

        content_src = model.bottleneck(z_src)
        content_pred = model.bottleneck(z_pred_end)
        loss_content = F.l1_loss(content_pred, content_src.detach()) * content_weight

        loss = loss_l1 + loss_fm + loss_content

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"step {step} | NaN/Inf, skipping", flush=True)
            optim.zero_grad()
            continue

        loss.backward()

        grad_has_nan = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                grad_has_nan = True
                break

        if grad_has_nan:
            print(f"step {step} | NaN/Inf in grad, skipping", flush=True)
            optim.zero_grad()
            grad_norm = float("nan")
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            optim.step()
            scheduler.step()

        losses_log["total"].append(loss.item())
        losses_log["l1"].append(loss_l1.item())
        losses_log["fm"].append(loss_fm.item())
        losses_log["content"].append(loss_content.item())

        if step % 100 == 0:
            avg = {k_: np.mean(v[-100:]) for k_, v in losses_log.items()}
            print(
                f"step {step}/{max_steps} | loss={avg['total']:.4f} "
                f"l1={avg['l1']:.4f} fm={avg['fm']:.4f} "
                f"content={avg['content']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} gnorm={grad_norm:.2f}",
                flush=True,
            )

        if step % cfg.get("checkpoint", {}).get("save_every_steps", 5000) == 0:
            ckpt_path = os.path.join(output_dir, f"step_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": cfg}, ckpt_path)
            latest_path = os.path.join(output_dir, "latest.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": cfg}, latest_path)
            print(f"  Saved: {ckpt_path}", flush=True)

    best_path = os.path.join(output_dir, "best.pt")
    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, best_path)
    print(f"\nkNN distillation complete. Final checkpoint: {best_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="kNN-VC distillation training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
