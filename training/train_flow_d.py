"""
Phase D: Flow matching with WavLM-SV loss on decoded audio.

The key difference from train_flow.py: the speaker similarity loss is computed
on the DECODED WAVEFORM, not on DAC latents. This eliminates metric gaming:

  z_pred → DAC decode → 44.1kHz wav → resample 16kHz → WavLM-SV → cosine loss

The gradient flows through the (frozen) DAC decoder, so the FlowConverter
learns to produce latents that decode to target-speaker audio.

Memory: B=4, T=200 → 11.2 GB peak. 22GB GPU is safe.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent))
from converter import FlowConverter, ConverterConfig
from train_flow import load_latent_corpus, sample_flow_batch


def train(config_path, data_dir, output_dir):
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = ConverterConfig(**cfg["model"])
    train_cfg = cfg["training"]
    loss_cfg = cfg["losses"]
    max_frames = cfg["data"]["max_utterance_frames"]
    min_frames = cfg["data"].get("min_utterance_frames", 30)
    timbre_shift_prob = train_cfg.get("timbre_shift_prob", 0.0)

    configured = train_cfg.get("device", "auto")
    if configured == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(configured)
    print(f"Device: {device}", flush=True)

    speakers = load_latent_corpus(data_dir, max_frames, min_frames)
    spk_list = sorted(speakers.keys())
    print(f"Loaded {sum(len(v) for v in speakers.values())} latents from {len(speakers)} speakers", flush=True)

    model = FlowConverter(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowConverter parameters: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)

    if "init_from" in train_cfg and train_cfg["init_from"]:
        ckpt = torch.load(train_cfg["init_from"], map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Initialized from {train_cfg['init_from']} (missing: {len(missing)}, unexpected: {len(unexpected)})", flush=True)

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

    print("Loading DAC decoder (for differentiable decode)...", flush=True)
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(device)
    dac.eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    resampler = torchaudio.transforms.Resample(44100, 16000).to(device)

    print("Loading WavLM-SV (for speaker loss)...", flush=True)
    wavlm = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(device)
    wavlm.eval()
    for p in wavlm.parameters():
        p.requires_grad_(False)

    wavlm_cache_path = os.path.join(os.path.dirname(data_dir.rstrip("/")), "wavlm_sv_embeddings.pkl")
    spk_teacher = {}
    if os.path.exists(wavlm_cache_path):
        with open(wavlm_cache_path, "rb") as f:
            wavlm_cache = pickle.load(f)
        spk_avg = {}
        for key, emb in wavlm_cache.items():
            spk = key.split("/")[0]
            spk_avg.setdefault(spk, []).append(emb)
        for spk, embeds in spk_avg.items():
            t = torch.from_numpy(np.mean(embeds, axis=0)).float().to(device)
            spk_teacher[spk] = F.normalize(t, dim=-1)
        print(f"WavLM-SV teacher: {len(spk_teacher)} speakers", flush=True)
    else:
        print("WARNING: WavLM-SV cache not found!", flush=True)

    model.train()
    losses_log = {"total": [], "fm": [], "l1": [], "wavlm_spk": [], "content": []}

    spk_weight = loss_cfg.get("wavlm_speaker", 5.0)
    fm_weight = loss_cfg.get("fm_velocity", 0.1)
    l1_weight = loss_cfg.get("latent_l1", 1.0)
    content_weight = loss_cfg.get("content_inv", 1.0)

    print(f"Losses: wavlm_spk={spk_weight} fm={fm_weight} l1={l1_weight} content={content_weight}", flush=True)
    print(f"Starting Phase D training for {max_steps} steps (B={batch_size})...", flush=True)

    for step in range(1, max_steps + 1):
        z_src, z_tgt, z_ref, src_spk_ids = sample_flow_batch(
            speakers, batch_size, max_frames, device, timbre_shift_prob
        )
        B = z_src.shape[0]

        tgt_spks = []
        for i in range(B):
            src_s = src_spk_ids[i]
            tgt_s = src_s
            while tgt_s == src_s and len(spk_list) > 1:
                tgt_s = spk_list[np.random.randint(0, len(spk_list))]
            tgt_spks.append(tgt_s)

        z_0 = z_src
        t = torch.rand(B, device=device)
        t_expand = t[:, None, None]
        z_t = (1.0 - t_expand) * z_0 + t_expand * z_tgt
        v_target = z_tgt - z_0

        optim.zero_grad()

        v_pred = model.forward_velocity(z_t, t, z_ref)
        loss_fm = F.mse_loss(v_pred, v_target) * fm_weight

        z_pred_end = z_0 + v_pred
        loss_l1 = F.l1_loss(z_pred_end, z_tgt) * l1_weight

        audio_44k = dac.decoder(z_pred_end)
        if audio_44k.ndim == 3:
            audio_44k = audio_44k.squeeze(1)
        audio_16k = resampler(audio_44k)
        wavlm_out = wavlm(input_values=audio_16k)
        pred_wavlm_embed = F.normalize(wavlm_out.last_hidden_state.mean(dim=1), dim=-1)

        tgt_wavlm_embeds = torch.stack([spk_teacher[s] for s in tgt_spks])
        loss_wavlm_spk = (
            1.0 - F.cosine_similarity(pred_wavlm_embed, tgt_wavlm_embeds, dim=-1).mean()
        ) * spk_weight

        content_src = model.bottleneck(z_src)
        content_pred = model.bottleneck(z_pred_end.detach())
        loss_content = F.l1_loss(content_pred, content_src.detach()) * content_weight

        loss = loss_fm + loss_l1 + loss_wavlm_spk + loss_content

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"step {step} | NaN/Inf detected, skipping", flush=True)
            optim.zero_grad()
            continue

        loss.backward()

        grad_has_nan = False
        for name, p in model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                grad_has_nan = True
                break

        if grad_has_nan:
            print(f"step {step} | NaN/Inf in gradient, skipping", flush=True)
            optim.zero_grad()
            grad_norm = float("nan")
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            optim.step()
            scheduler.step()

        losses_log["total"].append(loss.item())
        losses_log["fm"].append(loss_fm.item())
        losses_log["l1"].append(loss_l1.item())
        losses_log["wavlm_spk"].append(loss_wavlm_spk.item())
        losses_log["content"].append(loss_content.item())

        if step % 100 == 0:
            avg = {k: np.mean(v[-100:]) for k, v in losses_log.items()}
            print(
                f"step {step}/{max_steps} | loss={avg['total']:.4f} "
                f"fm={avg['fm']:.4f} l1={avg['l1']:.4f} "
                f"wavlm_spk={avg['wavlm_spk']:.4f} content={avg['content']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} gnorm={grad_norm:.2f}",
                flush=True,
            )

        if step % cfg.get("checkpoint", {}).get("save_every_steps", 5000) == 0:
            ckpt_path = os.path.join(output_dir, f"step_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": cfg}, ckpt_path)
            latest_path = os.path.join(output_dir, "latest.pt")
            torch.save({"model": model.state_dict(), "step": step, "config": cfg}, latest_path)
            print(f"  Saved checkpoint: {ckpt_path}", flush=True)

    best_path = os.path.join(output_dir, "best.pt")
    torch.save({"model": model.state_dict(), "step": step, "config": cfg}, best_path)
    print(f"\nPhase D training complete. Final checkpoint: {best_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase D: Flow matching with WavLM-SV loss")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
