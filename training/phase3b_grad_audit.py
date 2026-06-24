"""
Phase 3b gradient audit.

Measures whether decoded-audio losses actually provide useful gradients to the
generator output before changing the decoder. Run this before partial decoder
fine-tuning.
"""
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from phase3_model import TLG
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
)


RESULTS_DIR = Path("../results")


def grad_stats(grad):
    if grad is None:
        return {"rms": 0.0, "norm": 0.0, "max": 0.0, "finite": False}
    g = grad.detach()
    return {
        "rms": float(g.pow(2).mean().sqrt().cpu()),
        "norm": float(g.norm().cpu()),
        "max": float(g.abs().max().cpu()),
        "finite": bool(torch.isfinite(g).all().cpu()),
    }


def cosine_grad(a, b):
    if a is None or b is None:
        return 0.0
    af = a.detach().flatten()
    bf = b.detach().flatten()
    return float(F.cosine_similarity(af, bf, dim=0).cpu())


def load_model(args):
    model = TLG(
        content_dim=1024,
        hidden_dim=args.hidden_dim,
        timbre_dim=192,
        n_heads=8,
        n_layers=args.n_layers,
        causal=True,
    ).to(DEVICE)
    if args.checkpoint and Path(args.checkpoint).exists():
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using fresh TLG initialization")
    model.eval()
    return model


def make_batch(args):
    ds = PairDataset(DATA_DIR / args.split, args.max_frames)
    items = [ds[i] for i in range(min(args.batch_size, len(ds)))]
    return [x.to(DEVICE) for x in collate(items)]


def compute_graph(args, model, dac, ecapa, batch):
    z_s, q0_s, z_t, f0, energy, timbre = batch

    z_pred = model(z_s.transpose(1, 2), f0, energy, timbre).transpose(1, 2)
    z_pred.retain_grad()
    z_q = soft_rvq_requantize(dac, q0_s, z_pred, args.tau)
    audio = dac.decoder(z_q).squeeze(1)
    audio.retain_grad()

    with torch.no_grad():
        oracle_z = hard_rvq_requantize(dac, q0_s, z_t)
        source_z = hard_quantize_all(dac, z_s)
        oracle_audio = dac.decoder(oracle_z).squeeze(1)
        source_audio = dac.decoder(source_z).squeeze(1)
        src_emb = ecapa_embed(ecapa, resample_16k(source_audio))

    audio_16k = resample_16k(audio)
    emb = ecapa_embed(ecapa, audio_16k)
    spk_sim = F.cosine_similarity(emb, timbre, dim=-1)
    src_sim = F.cosine_similarity(emb, src_emb, dim=-1)

    losses = {
        "speaker": (1.0 - spk_sim).mean(),
        "leak": F.relu(src_sim - args.leak_margin).mean(),
        "stft": multi_scale_stft_loss(audio, oracle_audio),
        "latent": 1.0 - F.cosine_similarity(
            z_pred.transpose(1, 2), z_t.transpose(1, 2), dim=-1
        ).mean(),
    }
    sims = {
        "target_secs": float(spk_sim.mean().detach().cpu()),
        "source_secs": float(src_sim.mean().detach().cpu()),
    }
    return z_pred, audio, losses, sims


def audit(args):
    print("=== Phase 3b Gradient Audit ===")
    print(f"device={DEVICE} split={args.split} batch={args.batch_size} tau={args.tau}")

    dac = load_dac()
    ecapa = load_ecapa()
    model = load_model(args)
    batch = make_batch(args)

    z_pred, audio, losses, sims = compute_graph(args, model, dac, ecapa, batch)
    results = {"sims": sims, "losses": {}, "grad": {}, "grad_cosine": {}}
    z_grads = {}

    for name, loss in losses.items():
        grads = torch.autograd.grad(
            loss,
            (z_pred, audio),
            retain_graph=True,
            allow_unused=True,
        )
        z_grad, audio_grad = grads
        z_grads[name] = z_grad
        results["losses"][name] = float(loss.detach().cpu())
        results["grad"][name] = {
            "z_pred": grad_stats(z_grad),
            "audio": grad_stats(audio_grad),
        }

    names = list(losses.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            results["grad_cosine"][f"{a}_vs_{b}"] = cosine_grad(z_grads[a], z_grads[b])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"phase3b_grad_audit_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSECS target={sims['target_secs']:.3f} source={sims['source_secs']:.3f}")
    print(f"{'loss':<10} {'value':>9} | {'z_rms':>10} {'z_norm':>10} | {'audio_rms':>10} {'audio_norm':>10}")
    print("-" * 78)
    for name in names:
        g = results["grad"][name]
        print(
            f"{name:<10} {results['losses'][name]:>9.4f} | "
            f"{g['z_pred']['rms']:>10.3e} {g['z_pred']['norm']:>10.3e} | "
            f"{g['audio']['rms']:>10.3e} {g['audio']['norm']:>10.3e}"
        )

    print("\nz_pred gradient cosine:")
    for k, v in results["grad_cosine"].items():
        print(f"  {k:<22} {v:+.3f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3b gradient audit")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase3b/latest.pt")
    parser.add_argument("--split", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--tau", type=float, default=5.0)
    parser.add_argument("--leak_margin", type=float, default=0.2)
    audit(parser.parse_args())
