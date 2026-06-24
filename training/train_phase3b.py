"""
Phase 3b training: decoded-audio generator objective.

Trains TLG through soft RVQ and the frozen DAC decoder. This avoids the
embedding/code MSE collapse diagnosed in Phase 3 by optimizing speaker and
audio losses after decode.
"""
import sys
import time
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

DAC_SR = 44100
SECS_SR = 16000
DATA_DIR = Path("../data/phase3")
CKPT_DIR = Path("checkpoints/phase3b")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


DEVICE = get_device()


class PairDataset(Dataset):
    def __init__(self, directory, max_frames=384):
        self.files = sorted(Path(directory).glob("*.pt"))
        self.max_frames = max_frames

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location="cpu")
        z_s = d["z_s"].float()
        q0_s = d["q0_s"].float()
        z_t = d["z_t_aligned"].float()
        f0 = d["f0"].float()
        energy = d["energy"].float()
        timbre = d["timbre"].float().squeeze()

        T = z_s.shape[1]
        if T > self.max_frames:
            start = torch.randint(0, T - self.max_frames + 1, ()).item()
            z_s = z_s[:, start:start + self.max_frames]
            q0_s = q0_s[:, start:start + self.max_frames]
            z_t = z_t[:, start:start + self.max_frames]
            f0 = f0[start:start + self.max_frames]
            energy = energy[start:start + self.max_frames]

        return z_s, q0_s, z_t, f0, energy, timbre


def collate(batch):
    z_s, q0_s, z_t, f0, energy, timbre = zip(*batch)
    T_min = min(x.shape[1] for x in z_s)
    return (
        torch.stack([x[:, :T_min] for x in z_s]),
        torch.stack([x[:, :T_min] for x in q0_s]),
        torch.stack([x[:, :T_min] for x in z_t]),
        torch.stack([x[:T_min] for x in f0]),
        torch.stack([x[:T_min] for x in energy]),
        torch.stack(timbre),
    )


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


def load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def resample_16k(audio_44k):
    flat = audio_44k.reshape(audio_44k.shape[0], 1, -1)
    out = F.interpolate(flat, scale_factor=SECS_SR / DAC_SR,
                        mode="linear", align_corners=False)
    return out.squeeze(1)


def ecapa_embed(ecapa, audio_16k):
    emb = ecapa.encode_batch(audio_16k)
    if emb.dim() == 3 and emb.shape[1] == 1:
        emb = emb.squeeze(1)
    return emb


def soft_rvq_requantize(dac, q0_s, z_input, tau):
    z_q = q0_s
    residual = z_input - q0_s

    for depth in range(1, 9):
        quantizer = dac.quantizer.quantizers[depth]
        z_e = quantizer.in_proj(residual).transpose(1, 2)
        cb = quantizer.codebook.weight
        dist = torch.cdist(z_e, cb.unsqueeze(0)).pow(2)
        weights = F.softmax(-dist / tau, dim=-1)
        z_q_soft = (weights @ cb).transpose(1, 2)
        q_depth = quantizer.out_proj(z_q_soft)
        z_q = z_q + q_depth
        residual = residual - q_depth

    return z_q


@torch.no_grad()
def hard_rvq_requantize(dac, q0_s, z_input):
    z_q = q0_s.clone()
    residual = z_input - q0_s
    for depth in range(1, 9):
        q_depth, _, _, _, _ = dac.quantizer.quantizers[depth](residual)
        z_q = z_q + q_depth
        residual = residual - q_depth
    return z_q


@torch.no_grad()
def hard_quantize_all(dac, z_input):
    z_q = torch.zeros_like(z_input)
    residual = z_input.clone()
    for depth in range(dac.quantizer.n_codebooks):
        q_depth, _, _, _, _ = dac.quantizer.quantizers[depth](residual)
        z_q = z_q + q_depth
        residual = residual - q_depth
    return z_q


def multi_scale_stft_loss(audio, target):
    loss = audio.new_tensor(0.0)
    eps = 1e-7
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=audio.device)
        x = torch.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, return_complex=True)
        y = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, return_complex=True)
        x_mag = x.abs()
        y_mag = y.abs()
        sc = torch.linalg.norm(x_mag - y_mag) / (torch.linalg.norm(y_mag) + eps)
        mag = F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))
        loss = loss + sc + mag
    return loss / 3.0


def decoded_losses(args, ecapa, audio, oracle_audio, source_audio, timbre):
    audio_16k = resample_16k(audio)
    emb = ecapa_embed(ecapa, audio_16k)
    spk_sim = F.cosine_similarity(emb, timbre, dim=-1)
    loss_speaker = (1.0 - spk_sim).mean()

    with torch.no_grad():
        src_emb = ecapa_embed(ecapa, resample_16k(source_audio))

    leak_sim = F.cosine_similarity(emb, src_emb, dim=-1)
    loss_leak = F.relu(leak_sim - args.leak_margin).mean()
    loss_stft = multi_scale_stft_loss(audio, oracle_audio)
    return loss_speaker, loss_leak, loss_stft, spk_sim.mean(), leak_sim.mean()


def train(args):
    from phase3_model import TLG

    print("=== Phase 3b Training (Decoded Audio Loss) ===")
    print(f"device={DEVICE} tau={args.tau} batch={args.batch_size}")

    dac = load_dac()
    ecapa = load_ecapa()

    train_ds = PairDataset(DATA_DIR / "train", args.max_frames)
    eval_ds = PairDataset(DATA_DIR / "eval", args.max_frames)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.num_workers,
                          drop_last=True)
    eval_dl = DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False,
                         collate_fn=collate, num_workers=args.num_workers)

    model = TLG(
        content_dim=1024,
        hidden_dim=args.hidden_dim,
        timbre_dim=192,
        n_heads=8,
        n_layers=args.n_layers,
        causal=True,
    ).to(DEVICE)

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"resumed {args.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"train={len(train_ds)} eval={len(eval_ds)} params={n_params/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.98))
    total_steps = max(1, args.epochs * len(train_dl))
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_margin = -999.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        meters = {k: [] for k in ("total", "speaker", "leak", "stft", "latent",
                                  "spk_sim", "leak_sim")}
        t0 = time.time()

        for step, batch in enumerate(train_dl):
            z_s, q0_s, z_t, f0, energy, timbre = [
                x.to(DEVICE) for x in batch
            ]

            z_pred = model(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
            z_q = soft_rvq_requantize(dac, q0_s, z_pred, args.tau)
            audio = dac.decoder(z_q).squeeze(1)

            with torch.no_grad():
                oracle_z = hard_rvq_requantize(dac, q0_s, z_t)
                source_z = hard_quantize_all(dac, z_s)
                oracle_audio = dac.decoder(oracle_z).squeeze(1)
                source_audio = dac.decoder(source_z).squeeze(1)

            loss_spk, loss_leak, loss_stft, spk_sim, leak_sim = decoded_losses(
                args, ecapa, audio, oracle_audio, source_audio, timbre
            )
            loss_latent = 1.0 - F.cosine_similarity(
                z_pred.transpose(1, 2), z_t.transpose(1, 2), dim=-1
            ).mean()

            loss = (
                args.speaker_weight * loss_spk
                + args.leak_weight * loss_leak
                + args.stft_weight * loss_stft
                + args.latent_weight * loss_latent
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            global_step += 1

            for name, value in (
                ("total", loss), ("speaker", loss_spk), ("leak", loss_leak),
                ("stft", loss_stft), ("latent", loss_latent),
                ("spk_sim", spk_sim), ("leak_sim", leak_sim),
            ):
                meters[name].append(float(value.detach().cpu()))

            if step % args.log_every == 0:
                print(
                    f"E{epoch} S{step}/{len(train_dl)} "
                    f"loss={meters['total'][-1]:.3f} "
                    f"spk={meters['spk_sim'][-1]:.3f} "
                    f"src={meters['leak_sim'][-1]:.3f} "
                    f"stft={meters['stft'][-1]:.3f} "
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
            f"stft={avg['stft']:.3f} ({elapsed:.0f}s)"
        )

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            eval_result = evaluate(args, model, dac, ecapa, eval_dl)
            margin = eval_result["secs_target"] - eval_result["secs_source"]
            print(
                f"Eval: target={eval_result['secs_target']:.3f} "
                f"source={eval_result['secs_source']:.3f} "
                f"margin={margin:+.3f}"
            )
            ckpt = {"model": model.state_dict(), "epoch": epoch,
                    "args": vars(args), "eval": eval_result}
            torch.save(ckpt, CKPT_DIR / "latest.pt")
            if margin > best_margin:
                best_margin = margin
                torch.save(ckpt, CKPT_DIR / "best.pt")
                print(f"new best margin={best_margin:+.3f}")

        if args.max_steps and global_step >= args.max_steps:
            break


@torch.no_grad()
def evaluate(args, model, dac, ecapa, eval_dl):
    model.eval()
    target_scores = []
    source_scores = []

    for bi, batch in enumerate(eval_dl):
        if args.eval_batches >= 0 and bi >= args.eval_batches:
            break

        z_s, q0_s, _z_t, f0, energy, timbre = [x.to(DEVICE) for x in batch]
        z_pred = model(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
        z_q = soft_rvq_requantize(dac, q0_s, z_pred, args.tau)
        audio = dac.decoder(z_q).squeeze(1)
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

    return {
        "secs_target": float(np.mean(target_scores)) if target_scores else 0.0,
        "secs_source": float(np.mean(source_scores)) if source_scores else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b decoded-audio training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--speaker_weight", type=float, default=1.0)
    parser.add_argument("--leak_weight", type=float, default=0.2)
    parser.add_argument("--stft_weight", type=float, default=0.3)
    parser.add_argument("--latent_weight", type=float, default=0.02)
    parser.add_argument("--leak_margin", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=25,
                        help="number of eval batches, -1 for all, 0 to skip")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    train(parser.parse_args())
