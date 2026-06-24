"""
Causal Source-Filter VC — Envelope Converter Training (v2).

Uses pre-computed DTW-aligned pairs from data/sf_pairs/.
No on-the-fly DTW → fast training.

Model: Causal TCN with FiLM conditioning
  Input: source mel-cepstrum [B, 25, T]
  Conditioning: target speaker embedding [B, D] (FiLM)
  Output: predicted target mel-cepstrum [B, 25, T]

Loss: L1(predicted_mc, aligned_target_mc)
"""
import sys, os, json, time, pickle, argparse, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda")
MC_DIM = 25
PAIRS_DIR = Path("data/sf_pairs")


class EnvelopeConverter(nn.Module):
    def __init__(self, mc_dim=MC_DIM, spk_dim=768, hidden=256, n_blocks=8, kernel_size=7, causal=False):
        super().__init__()
        self.causal = causal
        self.proj_in = nn.Conv1d(mc_dim, hidden, 1)
        self.spk_proj = nn.Linear(spk_dim, hidden)
        self.film = nn.Linear(hidden, hidden * 2)
        self.film.bias.data[:hidden] = 1.0
        self.film.bias.data[hidden:] = 0.0

        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** (i % 4)
            self.blocks.append(nn.ModuleDict({
                "conv1": nn.Conv1d(hidden, hidden, kernel_size, dilation=dilation,
                                   padding=(kernel_size - 1) * dilation // 2 if not causal else (kernel_size - 1) * dilation),
                "conv2": nn.Conv1d(hidden, hidden, 1),
            }))

        self.proj_out = nn.Linear(hidden, mc_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, mc_src, spk_emb):
        T = mc_src.shape[-1]
        h = self.proj_in(mc_src)
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

        out = self.proj_out(h.transpose(1, 2)).transpose(1, 2)
        return out


class SFPairDataset:
    def __init__(self, spk_emb, pairs_dir=PAIRS_DIR, max_len=400, augment=True):
        self.spk_emb = spk_emb
        self.max_len = max_len
        self.augment = augment

        self.pair_files = sorted(pairs_dir.glob("pair_*.npz"))
        print(f"  Loaded {len(self.pair_files)} pre-computed pairs")

    def sample(self):
        npz_path = random.choice(self.pair_files)
        data = np.load(npz_path, allow_pickle=True)

        mc_src = data["mc_src"]
        mc_tgt = data["mc_tgt"]
        f0_src = data["f0_src"]
        codeap_tgt = data["codeap_tgt"]
        spk_tgt = str(data["spk_tgt"])

        T = min(len(mc_src), self.max_len)

        if self.augment and T > 50:
            start = random.randint(0, len(mc_src) - T)
            mc_src = mc_src[start:start+T]
            mc_tgt = mc_tgt[start:start+T]
            f0_src = f0_src[start:start+T]
            codeap_tgt = codeap_tgt[start:start+T]
        else:
            mc_src = mc_src[:T]
            mc_tgt = mc_tgt[:T]
            f0_src = f0_src[:T]
            codeap_tgt = codeap_tgt[:T]

        return {
            "mc_src": torch.from_numpy(mc_src).float(),
            "mc_tgt": torch.from_numpy(mc_tgt).float(),
            "f0_src": torch.from_numpy(f0_src).float(),
            "codeap_tgt": torch.from_numpy(codeap_tgt).float(),
            "spk_tgt": spk_tgt,
        }

    def collate(self, batch):
        min_t = min(item["mc_src"].shape[0] for item in batch)
        mc_src = torch.stack([b["mc_src"][:min_t] for b in batch]).to(DEVICE).transpose(1, 2)
        mc_tgt = torch.stack([b["mc_tgt"][:min_t] for b in batch]).to(DEVICE).transpose(1, 2)
        spk_emb = torch.stack([self.spk_emb[b["spk_tgt"]] for b in batch]).to(DEVICE)
        return mc_src, mc_tgt, spk_emb


def load_speaker_embeddings():
    cache_path = Path("data/wavlm_sv_embeddings.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    spk_avg = {}
    for key, emb in cache.items():
        spk = key.split("/")[0]
        spk_avg.setdefault(spk, []).append(emb)

    return {spk: torch.from_numpy(np.mean(embs, axis=0)).float() for spk, embs in spk_avg.items()}


def train(args):
    print("=== Causal Source-Filter VC Training (v2) ===\n")

    print("Loading speaker embeddings...")
    spk_emb = load_speaker_embeddings()
    spk_dim = next(iter(spk_emb.values())).shape[0]
    print(f"  {len(spk_emb)} speakers, dim={spk_dim}")

    print("Loading pre-computed pairs...")
    dataset = SFPairDataset(spk_emb, max_len=400)

    model = EnvelopeConverter(mc_dim=MC_DIM, spk_dim=spk_dim, hidden=args.hidden, n_blocks=args.n_blocks).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"EnvelopeConverter: {n_params:,} ({n_params/1e6:.2f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99998)

    os.makedirs(args.output, exist_ok=True)
    losses = []

    print(f"\nTraining {args.max_steps} steps (B={args.batch_size}, {args.n_blocks} blocks, hidden={args.hidden})...")
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        batch_items = [dataset.sample() for _ in range(args.batch_size)]
        mc_src, mc_tgt, spk_tgt_emb = dataset.collate(batch_items)

        optim.zero_grad()
        mc_pred = model(mc_src, spk_tgt_emb)
        loss = F.l1_loss(mc_pred, mc_tgt)
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
            print(f"step {step}/{args.max_steps} | l1={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e} | {sps:.1f}step/s ETA {eta:.0f}s", flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            path = os.path.join(args.output, f"step_{step:06d}.pt")
            ckpt = {
                "model": model.state_dict(), "step": step,
                "config": {"mc_dim": MC_DIM, "spk_dim": spk_dim, "hidden": args.hidden, "n_blocks": args.n_blocks},
            }
            torch.save(ckpt, path)
            torch.save(ckpt, os.path.join(args.output, "latest.pt"))
            print(f"  Saved: {path}", flush=True)

    print(f"\nTraining complete. Final L1: {np.mean(losses[-100:]):.4f}")


def evaluate(args):
    import soundfile as sf
    import pyworld as world
    import pysptk as sptk
    import librosa

    print("\n=== SF-VC Evaluation ===\n")
    spk_emb = load_speaker_embeddings()
    spk_dim = next(iter(spk_emb.values())).shape[0]

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]
    model = EnvelopeConverter(
        mc_dim=cfg["mc_dim"], spk_dim=cfg["spk_dim"],
        hidden=cfg["hidden"], n_blocks=cfg["n_blocks"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    dataset = SFPairDataset(spk_emb, max_len=800, augment=False)

    n_eval = min(args.n_eval, len(dataset.pair_files))
    tgt_scores, src_scores = [], []

    print(f"Evaluating {n_eval} pairs...")
    for i in range(n_eval):
        npz_path = dataset.pair_files[i]
        data = np.load(npz_path, allow_pickle=True)

        mc_src = data["mc_src"]
        mc_tgt = data["mc_tgt"]
        f0_src = data["f0_src"]
        codeap_tgt = data["codeap_tgt"]
        spk_src = str(data["spk_src"])
        spk_tgt = str(data["spk_tgt"])

        T = min(len(mc_src), 800)

        mc_src_t = torch.from_numpy(mc_src[:T]).float().unsqueeze(0).to(DEVICE).transpose(1, 2)
        spk_t = spk_emb[spk_tgt].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mc_pred = model(mc_src_t, spk_t)

        mc_pred_np = mc_pred.squeeze(0).cpu().numpy().T

        tgt_voiced = f0_src[f0_src > 0]
        tgt_mean_f0 = 200.0
        if spk_tgt in spk_emb:
            pass

        src_voiced = f0_src[f0_src > 0]
        if len(src_voiced) > 0:
            src_mean = np.exp(np.mean(np.log(src_voiced)))
            ratio = 1.0
        else:
            src_mean = 200.0
            ratio = 1.0

        f0_syn = f0_src[:T].astype(np.float64)

        mc_syn = np.ascontiguousarray(mc_pred_np, dtype=np.float64)
        sp = sptk.mc2sp(mc_syn, 0.410, 2048)
        codeap = np.ascontiguousarray(codeap_tgt[:T], dtype=np.float64)
        ap = world.decode_aperiodicity(codeap, 16000, 2048) if codeap.shape[1] > 0 else np.ones_like(sp)
        wav_syn = world.synthesize(f0_syn[:T], sp, ap, 16000, frame_period=5.0).astype(np.float32)

        wav_src_path = None
        for spk_dir in Path("../data/vctk_200").iterdir():
            if spk_dir.name == spk_src:
                wavs = sorted(spk_dir.glob("*.wav"))
                if wavs:
                    wav_src_path = str(wavs[0])
                break

        wav_tgt_path = None
        for spk_dir in Path("../data/vctk_200").iterdir():
            if spk_dir.name == spk_tgt:
                wavs = sorted(spk_dir.glob("*.wav"))
                if wavs:
                    wav_tgt_path = str(wavs[0])
                break

        if wav_src_path is None or wav_tgt_path is None:
            continue

        wav_src, sr = sf.read(wav_src_path, dtype="float32")
        wav_tgt, sr = sf.read(wav_tgt_path, dtype="float32")
        if sr != 16000:
            wav_src = librosa.resample(wav_src, orig_sr=sr, target_sr=16000)
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr, target_sr=16000)

        min_len = min(len(wav_src), len(wav_tgt), len(wav_syn))
        wav_src, wav_tgt, wav_syn = wav_src[:min_len], wav_tgt[:min_len], wav_syn[:min_len]

        with torch.no_grad():
            e_src = secs_model.encode_batch(torch.from_numpy(wav_src.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = secs_model.encode_batch(torch.from_numpy(wav_tgt.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_syn = secs_model.encode_batch(torch.from_numpy(wav_syn.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)

        tgt_sim = F.cosine_similarity(e_tgt, e_syn, dim=-1).item()
        src_sim = F.cosine_similarity(e_src, e_syn, dim=-1).item()
        tgt_scores.append(tgt_sim)
        src_scores.append(src_sim)
        print(f"  [{i+1}/{n_eval}] {spk_src}→{spk_tgt}: tgt={tgt_sim:.3f} src={src_sim:.3f}")

    tgt_arr = np.array(tgt_scores)
    src_arr = np.array(src_scores)
    print(f"\n=== Results ===")
    print(f"SECS(target): {tgt_arr.mean():.4f} ± {tgt_arr.std():.4f}")
    print(f"SECS(source): {src_arr.mean():.4f} ± {src_arr.std():.4f}")

    results = {
        "secs_tgt_mean": float(tgt_arr.mean()),
        "secs_tgt_std": float(tgt_arr.std()),
        "secs_src_mean": float(src_arr.mean()),
        "secs_src_std": float(src_arr.std()),
    }
    with open(os.path.join(os.path.dirname(args.checkpoint), "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    train_p = sub.add_parser("train")
    train_p.add_argument("--output", default="checkpoints/sf_vc")
    train_p.add_argument("--batch_size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=3e-4)
    train_p.add_argument("--max_steps", type=int, default=30000)
    train_p.add_argument("--save_every", type=int, default=3000)
    train_p.add_argument("--hidden", type=int, default=192)
    train_p.add_argument("--n_blocks", type=int, default=6)
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--n_eval", type=int, default=20)
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
