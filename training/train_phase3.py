"""
Phase 3 training: Codebook Embedding Regression

Model predicts 8-dim codebook embeddings per depth (regression, not classification).
At inference: nearest-neighbor in codebook → discrete codes → from_codes → decode.

Loss: MSE on embeddings + cosine.
Anti-collapse: diverse 8-dim targets, no median collapse risk.
"""
import sys, json, time, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
DAC_SR = 44100
SECS_SR = 16000
DATA_DIR = Path("../data/phase3")
CKPT_DIR = Path("checkpoints/phase3")


class PairDataset(Dataset):
    def __init__(self, directory, max_frames=512):
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
            start = (T - self.max_frames) // 2
            z_s = z_s[:, start:start+self.max_frames]
            q0_s = q0_s[:, start:start+self.max_frames]
            z_t = z_t[:, start:start+self.max_frames]
            f0 = f0[start:start+self.max_frames]
            energy = energy[start:start+self.max_frames]

        return z_s, q0_s, z_t, f0, energy, timbre


def collate(batch):
    z_s, q0_s, z_t, f0, energy, timbre = zip(*batch)
    T_min = min(x.shape[1] for x in z_s)
    z_s = torch.stack([x[:, :T_min] for x in z_s])
    q0_s = torch.stack([x[:, :T_min] for x in q0_s])
    z_t = torch.stack([x[:, :T_min] for x in z_t])
    f0 = torch.stack([x[:T_min] for x in f0])
    energy = torch.stack([x[:T_min] for x in energy])
    timbre = torch.stack(timbre)
    return z_s, q0_s, z_t, f0, energy, timbre


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


@torch.no_grad()
def rvq_requantize(dac, q0_s, z_t_like):
    """Returns z_q [B, 1024, T], codes [B, 8, T]"""
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    B, _, T = q0_s.shape

    z_q = q0_s.clone()
    residual = z_t_like - q0_s
    codes_all = []

    for d in range(1, n):
        q_out, _, _, codes_d, _ = qs[d](residual)
        z_q = z_q + q_out
        residual = residual - q_out
        codes_all.append(codes_d)

    codes = torch.stack(codes_all, dim=1)
    return z_q, codes


@torch.no_grad()
def get_target_codes(dac, q0_s, z_t_aligned):
    """Target codes from re-quantizing z_t_aligned - q0_s"""
    return rvq_requantize(dac, q0_s, z_t_aligned)[1]


def differentiable_rvq_requantize(dac, q0_s, z_t_like):
    """
    RVQ re-quantization with straight-through estimator.
    Forward: exact quantized z_q_hard
    Backward: gradient passes through to z_t_like (STE identity)
    """
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks

    with torch.no_grad():
        z_q_hard = q0_s.clone()
        residual = z_t_like.detach() - q0_s
        for d in range(1, n):
            q_out, _, _, _, _ = qs[d](residual)
            z_q_hard = z_q_hard + q_out
            residual = residual - q_out

    z_q = z_t_like + (z_q_hard - z_t_like).detach()
    return z_q


def differentiable_decode(dac, z_q):
    """Decode with gradient through z_q"""
    return dac.decoder(z_q)


def resample_16k(audio_44k):
    """Differentiable-ish resample 44.1k → 16k via conv"""
    B = audio_44k.shape[0]
    audio_44k_flat = audio_44k.reshape(B, 1, -1)
    audio_16k = F.interpolate(audio_44k_flat, scale_factor=SECS_SR/DAC_SR,
                              mode='linear', align_corners=False)
    return audio_16k.squeeze(1)


def train(args):
    print("=== Phase 3 Training (Embedding Regression) ===\n")

    from phase3_model import TLG_Embed
    dac = load_dac()
    print("DAC loaded")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    for p in secs_model.parameters():
        p.requires_grad_(False)
    print("ECAPA loaded")

    train_ds = PairDataset(DATA_DIR / "train", args.max_frames)
    eval_ds = PairDataset(DATA_DIR / "eval", args.max_frames)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=2, drop_last=True)
    print(f"Train: {len(train_ds)} pairs, Eval: {len(eval_ds)} pairs\n")

    model = TLG_Embed(
        content_dim=1024, hidden_dim=args.hidden_dim, timbre_dim=192,
        n_heads=8, n_layers=args.n_layers,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M\n")

    codebooks = []
    for d in range(1, 9):
        cb = dac.quantizer.quantizers[d].codebook.weight  # [1024, 8]
        codebooks.append(cb)
    codebooks = torch.stack(codebooks).to(DEVICE)  # [8, 1024, 8]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01, betas=(0.9, 0.98))
    total_steps = args.epochs * len(train_dl)
    warmup_steps = min(500, total_steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_secs = 0.0

    for epoch in range(args.epochs):
        model.train()
        mse_losses, cos_losses, code_accs = [], [], []

        for step, (z_s, q0_s, z_t, f0, energy, timbre) in enumerate(train_dl):
            z_s = z_s.to(DEVICE); q0_s = q0_s.to(DEVICE)
            z_t = z_t.to(DEVICE); f0 = f0.to(DEVICE)
            energy = energy.to(DEVICE); timbre = timbre.to(DEVICE)

            with torch.no_grad():
                target_codes = get_target_codes(dac, q0_s, z_t)  # [B, 8, T]
                target_embeds = torch.stack([
                    codebooks[d, target_codes[:, d].long()]
                    for d in range(8)
                ], dim=1)  # [B, 8, T, 8]

            pred_embeds = model(z_s.transpose(1, 2), f0, energy, timbre)  # [B, 8, T, 8]

            mse_loss = F.mse_loss(pred_embeds, target_embeds)
            cos_loss = (1.0 - F.cosine_similarity(pred_embeds, target_embeds, dim=-1)).mean()
            loss = mse_loss + 0.5 * cos_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                dist = torch.cdist(pred_embeds.detach(), codebooks)  # [B, 8, T, 1024]
                pred_codes = dist.argmin(dim=-1)  # [B, 8, T]
                acc = (pred_codes == target_codes).float().mean().item()

            mse_losses.append(mse_loss.item())
            cos_losses.append(cos_loss.item())
            code_accs.append(acc)

            if step % 50 == 0:
                print(f"  E{epoch} S{step}/{len(train_dl)} "
                      f"mse={mse_loss.item():.4f} cos={cos_loss.item():.4f} "
                      f"acc={acc:.3f} lr={scheduler.get_last_lr()[0]:.2e}",
                      flush=True)

        print(f"\nEpoch {epoch}: mse={np.mean(mse_losses):.4f} "
              f"cos={np.mean(cos_losses):.4f} acc={np.mean(code_accs):.3f}\n")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            secs_score = evaluate(model, dac, secs_model, eval_ds, codebooks)
            print(f"  Eval SECS: {secs_score:.4f}\n")
            if secs_score > best_secs:
                best_secs = secs_score
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "secs": secs_score, "args": vars(args)},
                           CKPT_DIR / "best.pt")
                print(f"  ★ New best: {best_secs:.4f}")
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                   CKPT_DIR / "latest.pt")

    print(f"\nBest SECS: {best_secs:.4f}")


@torch.no_grad()
def evaluate(model, dac, secs_model, dataset, codebooks):
    model.eval()
    scores = []
    dl = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)

    for z_s, q0_s, z_t, f0, energy, timbre in dl:
        z_s = z_s.to(DEVICE); q0_s = q0_s.to(DEVICE)
        f0 = f0.to(DEVICE); energy = energy.to(DEVICE)
        timbre = timbre.to(DEVICE)

        pred_embeds = model(z_s.transpose(1, 2), f0, energy, timbre)
        dist = torch.cdist(pred_embeds, codebooks)  # [B, 8, T, 1024]
        codes_pred = dist.argmin(dim=-1)  # [B, 8, T]

        _, codes_s = get_source_codes(dac, z_s)
        codes_0 = codes_s[0]  # [B, T]
        full_codes = torch.cat([codes_0.unsqueeze(1), codes_pred], dim=1)
        z_q, _, _ = dac.quantizer.from_codes(full_codes)
        audio_44k = dac.decoder(z_q).squeeze(1)
        audio_16k = resample_16k(audio_44k)

        for b in range(audio_16k.shape[0]):
            a = audio_16k[b]
            if a.shape[0] < 8000:
                continue
            e_out = secs_model.encode_batch(a.unsqueeze(0))
            sim = F.cosine_similarity(e_out.squeeze(0), timbre[b], dim=-1).item()
            scores.append(sim)

    return np.mean(scores) if scores else 0.0


@torch.no_grad()
def get_source_codes(dac, z_s):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    res = z_s.clone()
    codes_all = []
    for d in range(n):
        _, _, _, codes_d, _ = qs[d](res)
        q_out, _, _, _, _ = qs[d](res)
        codes_all.append(codes_d)
        res = res - q_out
    return None, codes_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    args = parser.parse_args()
    train(args)
