"""
Train a lightweight mel-cepstrum speaker classifier.

Input: mc [B, 25, T]
Output: speaker logits [B, n_speakers]

Used as auxiliary loss during VC training to ensure
mel-cepstrum output matches target speaker identity.
"""
import sys, os, time, random, pickle, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda")
MC_DIM = 25
MC_CACHE = Path("data/mc_cache")


class MCSpeakerClassifier(nn.Module):
    def __init__(self, mc_dim=MC_DIM, hidden=128, n_speakers=109, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(mc_dim, hidden, 5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden, 5, padding=2)
        self.ln1 = nn.GroupNorm(8, hidden)
        self.ln2 = nn.GroupNorm(8, hidden)
        self.ln3 = nn.GroupNorm(8, hidden)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, n_speakers)

    def forward(self, mc):
        h = F.gelu(self.ln1(self.conv1(mc)))
        h = self.drop(h)
        h = F.gelu(self.ln2(self.conv2(h)))
        h = self.drop(h)
        h = F.gelu(self.ln3(self.conv3(h)))
        h = h.mean(dim=-1)
        return self.proj(h)

    def embed(self, mc):
        h = F.gelu(self.ln1(self.conv1(mc)))
        h = F.gelu(self.ln2(self.conv2(h)))
        h = F.gelu(self.ln3(self.conv3(h)))
        return h.mean(dim=-1)


def load_dataset():
    data = []
    spk_to_id = {}
    spk_dirs = sorted([d for d in MC_CACHE.iterdir() if d.is_dir()])
    for sid, spk_dir in enumerate(spk_dirs):
        spk_to_id[spk_dir.name] = sid

    for spk_dir in spk_dirs:
        sid = spk_to_id[spk_dir.name]
        for npz_path in spk_dir.glob("*.npz"):
            try:
                d = np.load(npz_path)
                mc = d["mc"]
                if len(mc) >= 20:
                    data.append((mc.astype(np.float32), sid))
            except:
                continue
    return data, spk_to_id, len(spk_dirs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="checkpoints/mc_speaker_clf.pt")
    args = parser.parse_args()

    print("Loading mel-cepstrum data...")
    data, spk_to_id, n_speakers = load_dataset()
    print(f"  {len(data)} utterances, {n_speakers} speakers")

    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    model = MCSpeakerClassifier(n_speakers=n_speakers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Classifier: {n_params:,} ({n_params/1e6:.2f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        random.shuffle(train_data)
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i+args.batch_size]
            min_t = min(len(mc) for mc, _ in batch)
            min_t = min(min_t, 200)

            mc_batch = torch.stack([
                torch.from_numpy(mc[:min_t]) for mc, _ in batch
            ]).to(DEVICE).transpose(1, 2)

            labels = torch.tensor([sid for _, sid in batch]).to(DEVICE)

            optim.zero_grad()
            logits = model(mc_batch)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optim.step()

            total_loss += loss.item() * len(batch)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(batch)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_data), args.batch_size):
                batch = val_data[i:i+args.batch_size]
                if not batch:
                    continue
                min_t = min(len(mc) for mc, _ in batch)
                min_t = min(min_t, 200)

                mc_batch = torch.stack([
                    torch.from_numpy(mc[:min_t]) for mc, _ in batch
                ]).to(DEVICE).transpose(1, 2)
                labels = torch.tensor([sid for _, sid in batch]).to(DEVICE)

                logits = model(mc_batch)
                val_correct += (logits.argmax(-1) == labels).sum().item()
                val_total += len(batch)
        model.train()

        val_acc = val_correct / max(val_total, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "spk_to_id": spk_to_id,
        "n_speakers": n_speakers,
        "config": {"mc_dim": MC_DIM, "hidden": 128, "n_speakers": n_speakers},
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
