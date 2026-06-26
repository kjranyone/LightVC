"""
Phase 3c: Timbre-Conditioned Pre-Decoder Adapter Training

Pipeline:
  z_s → TLG generator → z_pred → soft RVQ → z_q
    → TimbreAdapter(z_q, target_timbre) → z_q_adapted
    → frozen DAC decoder → audio

The adapter provides a SHORT gradient path from timbre to audio:
  speaker_loss → ECAPA → audio → decoder → z_q_adapted → adapter params

This bypasses the weak Jacobian of the long generator → soft RVQ → decoder
chain that caused Phase 3b to plateau at target SECS ≈ 0.14.

Adapter: ConvFiLM
  conv_in(1024, bottleneck) → FiLM(timbre) → GELU → conv_out(bottleneck, 1024)
  Zero-init conv_out so adapter starts as identity (delta = 0).

Options:
  --adapter_only: skip generator, use z_pred = z_s. Tests whether the adapter
                  alone can convert speaker identity starting from source z_q.
"""
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from train_phase3b import (
    DEVICE,
    DATA_DIR,
    DAC_SR,
    SECS_SR,
    PairDataset,
    collate,
    load_dac,
    load_ecapa,
    resample_16k,
    ecapa_embed,
    soft_rvq_requantize,
    hard_rvq_requantize,
    hard_quantize_all,
    multi_scale_stft_loss,
    decoded_losses,
)

CKPT_DIR = Path("checkpoints/phase3c")


@torch.no_grad()
def extract_speaker_depths(dac, ref_latent, speaker_depths=(1, 2, 3)):
    """Extract speaker-depth quantization from reference latent.

    Sequentially quantizes ref_latent through all depths, keeps only
    the specified speaker depths. Filters out content (d0) and fine detail (d4-8).
    """
    residual = ref_latent.clone()
    speaker_q = torch.zeros_like(ref_latent)
    for d in range(9):
        out = dac.quantizer.quantizers[d](residual)
        if d in speaker_depths:
            speaker_q = speaker_q + out[0]
        residual = residual - out[0]
    return speaker_q

class TimbreAdapter(nn.Module):
    def __init__(self, latent_dim=1024, timbre_dim=192, bottleneck=256,
                 kernel=3, n_blocks=1,
                 utte_mode="none", film_mode="full",
                 n_tokens=32, n_heads=4):
        super().__init__()
        self.n_blocks = n_blocks
        self.utte_mode = utte_mode
        self.film_mode = film_mode
        self.n_tokens = n_tokens
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = nn.ModuleDict({
                "conv_in": nn.Conv1d(latent_dim, bottleneck, kernel, padding=kernel // 2),
                "film_gamma": nn.Linear(timbre_dim, bottleneck),
                "film_beta": nn.Linear(timbre_dim, bottleneck),
                "conv_out": nn.Conv1d(bottleneck, latent_dim, kernel, padding=kernel // 2),
            })
            nn.init.zeros_(block["conv_out"].weight)
            nn.init.zeros_(block["conv_out"].bias)
            self.blocks.append(block)

        if utte_mode == "ecapa":
            self.ecapa_to_tokens = nn.Linear(timbre_dim, n_tokens * bottleneck)
        elif utte_mode in ("target", "target_film"):
            self.token_proj = nn.Linear(latent_dim, bottleneck)
        elif utte_mode == "ref_latent":
            self.ref_proj = nn.Linear(latent_dim, bottleneck)

        if utte_mode != "none":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=bottleneck, num_heads=n_heads, batch_first=True,
            )
            nn.init.zeros_(self.cross_attn.out_proj.weight)
            nn.init.zeros_(self.cross_attn.out_proj.bias)

    def _make_tokens(self, timbre, z_target, ref_latent=None):
        if self.utte_mode == "ecpa":
            B = timbre.shape[0]
            t = self.ecapa_to_tokens(timbre)
            return t.reshape(B, self.n_tokens, -1)
        elif self.utte_mode in ("target", "target_film"):
            if z_target is None:
                return None
            z_det = z_target.detach()
            B, C, T = z_det.shape
            n_actual = min(self.n_tokens, T)
            seg_size = max(T // n_actual, 1)
            z_trimmed = z_det[:, :, :seg_size * n_actual]
            z_pooled = z_trimmed.reshape(B, C, n_actual, seg_size).mean(dim=-1)
            return self.token_proj(z_pooled.transpose(1, 2))
        elif self.utte_mode == "ref_latent":
            if ref_latent is None:
                return None
            return self.ref_proj(ref_latent.transpose(1, 2))
        return None

    def forward(self, z_q, timbre, z_target=None, ref_latent=None):
        h = z_q
        tokens = self._make_tokens(timbre, z_target, ref_latent) if self.utte_mode != "none" else None
        for block in self.blocks:
            x = block["conv_in"](h)
            if self.film_mode == "full":
                gamma = block["film_gamma"](timbre).unsqueeze(2)
                beta = block["film_beta"](timbre).unsqueeze(2)
                x = x * (1 + gamma) + beta
            if tokens is not None:
                h_t = x.transpose(1, 2)
                attn_out, _ = self.cross_attn(h_t, tokens, tokens)
                x = x + attn_out.transpose(1, 2)
            x = F.gelu(x)
            delta = block["conv_out"](x)
            h = h + delta
        return h


def train(args):
    print("=== Phase 3c: Timbre-Conditioned Adapter Training ===")
    print(f"device={DEVICE} adapter_only={args.adapter_only} tau={args.tau}")
    data_dir = Path(args.data_dir)
    print(f"data_dir={data_dir}")

    dac = load_dac()
    ecapa = load_ecapa()

    train_ds = PairDataset(data_dir / "train", args.max_frames,
                           ref_latent_dir=args.ref_latent_dir if args.utte_mode == "ref_latent" else None)
    eval_ds = PairDataset(data_dir / "eval", args.max_frames,
                          ref_latent_dir=args.ref_latent_dir if args.utte_mode == "ref_latent" else None)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.num_workers,
                          drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False,
                         collate_fn=collate, num_workers=args.num_workers)

    generator = None
    if not args.adapter_only:
        from phase3_model import TLG
        generator = TLG(
            content_dim=1024,
            hidden_dim=args.hidden_dim,
            timbre_dim=192,
            n_heads=8,
            n_layers=args.n_layers,
            causal=True,
        ).to(DEVICE)
        if args.resume_generator:
            ck = torch.load(args.resume_generator, map_location="cpu", weights_only=False)
            generator.load_state_dict(ck["model"])
            print(f"Resumed generator: {args.resume_generator}")
        gen_params = list(generator.parameters())
        gen_n = sum(p.numel() for p in gen_params)
        print(f"Generator: {gen_n / 1e6:.1f}M params")
    else:
        gen_params = []
        print("Generator: none (adapter_only mode, z_pred = z_s)")

    adapter = TimbreAdapter(
        latent_dim=1024,
        timbre_dim=192,
        bottleneck=args.bottleneck,
        kernel=args.kernel,
        n_blocks=args.n_blocks,
        utte_mode=args.utte_mode,
        film_mode=args.film_mode,
        n_tokens=args.n_tokens,
        n_heads=args.n_heads,
    ).to(DEVICE)
    adapter_params = list(adapter.parameters())
    adapter_n = sum(p.numel() for p in adapter_params)
    print(f"Adapter: {adapter_n / 1e6:.1f}M params")

    param_groups = [{"params": adapter_params, "lr": args.adapter_lr}]
    if generator:
        param_groups.insert(0, {"params": gen_params, "lr": args.gen_lr})

    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay,
                            betas=(0.9, 0.98))

    total_steps = max(1, args.epochs * len(train_dl))
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else CKPT_DIR
    if args.adapter_only and not args.ckpt_dir:
        ckpt_dir = Path("checkpoints/phase3c_adapter_only")
    print(f"ckpt_dir={ckpt_dir}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_margin = -999.0
    global_step = 0

    for epoch in range(args.epochs):
        if generator:
            generator.train()
        adapter.train()
        meters = {k: [] for k in (
            "total", "speaker", "leak", "stft", "latent",
            "spk_sim", "leak_sim", "delta_norm", "margin_l",
        )}
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            z_s, q0_s, z_t, f0, energy, timbre, ref_latent = [x.to(DEVICE) if x is not None else None for x in batch]

            if generator:
                z_pred = generator(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
            else:
                z_pred = z_s

            z_q = soft_rvq_requantize(dac, q0_s, z_pred, args.tau)
            ref_cond = extract_speaker_depths(dac, ref_latent) if ref_latent is not None else None
            z_q_adapted = adapter(z_q, timbre, z_t, ref_latent=ref_cond)
            audio = dac.decoder(z_q_adapted).squeeze(1)

            with torch.no_grad():
                oracle_z = hard_rvq_requantize(dac, q0_s, z_t)
                source_z = hard_quantize_all(dac, z_s)
                oracle_audio = dac.decoder(oracle_z).squeeze(1)
                source_audio = dac.decoder(source_z).squeeze(1)

            loss_spk, loss_leak, loss_stft, spk_sim, leak_sim = decoded_losses(
                args, ecapa, audio, oracle_audio, source_audio, timbre
            )

            delta = z_q_adapted - z_q
            delta_norm_val = delta.pow(2).mean().sqrt()
            loss_delta = delta.pow(2).mean()

            loss = (
                args.speaker_weight * loss_spk
                + args.leak_weight * loss_leak
                + args.stft_weight * loss_stft
                + args.delta_reg * loss_delta
            )

            if args.margin_weight > 0:
                loss_margin = F.relu(
                    args.margin_m + leak_sim - spk_sim
                ).mean()
                loss = loss + args.margin_weight * loss_margin
                meters["margin_l"].append(float(loss_margin.detach().cpu()))

            if generator:
                loss_latent = 1.0 - F.cosine_similarity(
                    z_pred.transpose(1, 2), z_t.transpose(1, 2), dim=-1
                ).mean()
                loss = loss + args.latent_weight * loss_latent
                meters["latent"].append(float(loss_latent.detach().cpu()))
            else:
                meters["latent"].append(0.0)

            if args.margin_weight <= 0:
                meters["margin_l"].append(0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, args.grad_clip)
            if generator:
                torch.nn.utils.clip_grad_norm_(gen_params, args.grad_clip)
            opt.step()
            sched.step()
            global_step += 1

            for name, value in (
                ("total", loss), ("speaker", loss_spk), ("leak", loss_leak),
                ("stft", loss_stft), ("spk_sim", spk_sim), ("leak_sim", leak_sim),
                ("delta_norm", delta_norm_val),
            ):
                meters[name].append(float(value.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"E{epoch} S{step}/{len(train_dl)} "
                    f"loss={meters['total'][-1]:.3f} "
                    f"spk={meters['spk_sim'][-1]:.3f} "
                    f"src={meters['leak_sim'][-1]:.3f} "
                    f"Δ={meters['delta_norm'][-1]:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )

            if args.max_steps and global_step >= args.max_steps:
                break

        elapsed = time.time() - t0
        avg = {k: float(np.mean(v)) if v else 0.0 for k, v in meters.items()}
        print(
            f"Epoch {epoch}: loss={avg['total']:.3f} "
            f"spk={avg['spk_sim']:.3f} src={avg['leak_sim']:.3f} "
            f"Δ={avg['delta_norm']:.4f} ({elapsed:.0f}s)"
        )

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_result = evaluate(args, generator, adapter, dac, ecapa, eval_dl)
            margin = eval_result["secs_target"] - eval_result["secs_source"]
            print(
                f"Eval: target={eval_result['secs_target']:.3f} "
                f"source={eval_result['secs_source']:.3f} "
                f"margin={margin:+.3f} "
                f"Δ={eval_result['delta_norm']:.4f}"
            )
            ckpt = {
                "generator": generator.state_dict() if generator else None,
                "adapter": adapter.state_dict(),
                "epoch": epoch,
                "args": vars(args),
                "eval": eval_result,
            }
            torch.save(ckpt, ckpt_dir / "latest.pt")
            if margin > best_margin:
                best_margin = margin
                torch.save(ckpt, ckpt_dir / "best.pt")
                print(f"new best margin={best_margin:+.3f}")

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"\nBest margin: {best_margin:+.3f}")


@torch.no_grad()
def evaluate(args, generator, adapter, dac, ecapa, eval_dl):
    if generator:
        generator.eval()
    adapter.eval()
    target_scores = []
    source_scores = []
    delta_norms = []

    for bi, batch in enumerate(eval_dl):
        if args.eval_batches >= 0 and bi >= args.eval_batches:
            break

        z_s, q0_s, z_t, f0, energy, timbre, ref_latent = [x.to(DEVICE) if x is not None else None for x in batch]

        if generator:
            z_pred = generator(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
        else:
            z_pred = z_s

        z_q = soft_rvq_requantize(dac, q0_s, z_pred, args.tau)
        ref_cond = extract_speaker_depths(dac, ref_latent) if ref_latent is not None else None
        z_q_adapted = adapter(z_q, timbre, z_t, ref_latent=ref_cond)
        audio = dac.decoder(z_q_adapted).squeeze(1)
        emb = ecapa_embed(ecapa, resample_16k(audio))

        source_z = hard_quantize_all(dac, z_s)
        source_audio = dac.decoder(source_z).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))

        target_scores.extend(
            F.cosine_similarity(emb, timbre, dim=-1).detach().cpu().tolist()
        )
        source_scores.extend(
            F.cosine_similarity(emb, source_emb, dim=-1).detach().cpu().tolist()
        )
        delta_norms.append(
            float((z_q_adapted - z_q).pow(2).mean().sqrt().cpu())
        )

    return {
        "secs_target": float(np.mean(target_scores)) if target_scores else 0.0,
        "secs_source": float(np.mean(source_scores)) if source_scores else 0.0,
        "delta_norm": float(np.mean(delta_norms)) if delta_norms else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3c timbre-conditioned adapter")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--n_blocks", type=int, default=1,
                        help="number of residual FiLM conv blocks in adapter")
    parser.add_argument("--utte_mode", type=str, default="none",
                        choices=["none", "ecpa", "target", "target_film", "ref_latent"],
                        help="UTTE cross-attention token source")
    parser.add_argument("--film_mode", type=str, default="full",
                        choices=["full", "none"],
                        help="FiLM conditioning mode")
    parser.add_argument("--n_tokens", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--gen_lr", type=float, default=1e-4)
    parser.add_argument("--adapter_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--speaker_weight", type=float, default=1.0)
    parser.add_argument("--leak_weight", type=float, default=0.2)
    parser.add_argument("--stft_weight", type=float, default=0.3)
    parser.add_argument("--latent_weight", type=float, default=0.1)
    parser.add_argument("--delta_reg", type=float, default=0.1,
                        help="L2 regularization on adapter delta to prevent explosion")
    parser.add_argument("--leak_margin", type=float, default=0.2)
    parser.add_argument("--margin_weight", type=float, default=0.0,
                        help="weight for margin ranking loss relu(m + src_sim - tgt_sim)")
    parser.add_argument("--margin_m", type=float, default=0.1,
                        help="margin for ranking loss")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=25)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--adapter_only", action="store_true",
                        help="skip generator, z_pred = z_s")
    parser.add_argument("--resume_generator", type=str, default="")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--ref_latent_dir", type=str, default="../data/ref_latents",
                        help="pre-computed reference latent pool directory")
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help="override checkpoint directory")
    train(parser.parse_args())
