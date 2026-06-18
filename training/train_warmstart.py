"""
Phase B: Bottleneck autoencoder warm-start.

Trains the `Converter` (residual-prediction variant) as an AutoVC-style
bottleneck autoencoder in DAC latent space. This gives the flow converter
a stable initialization.

Teacher-free paradigm ([04-7] revision): since there is no VC teacher,
cross-speaker targets like "z_src(A) with B's timbre" cannot be created.
The warm-start is therefore **pure reconstruction with bottleneck
disentanglement** (the original AutoVC recipe):

  - src and tgt are the same utterance (autoencode)
  - ref is from the **same speaker** (different utterance if available,
    so the speaker encoder generalizes across recordings)
  - Disentanglement emerges from the bottleneck being too narrow for
    speaker info; the model MUST take speaker from the reference.

Losses ([04-8] [04-9] [04-11] fixes):
  - reconstruction L1: pred ≈ tgt(=src)
  - speaker consistency: speaker_embed(pred) ≈ speaker_embed(ref)
    ([04-8]: was targeting src/tgt speaker — same as ref in reconstruction,
     but now explicit and correct for any role)
  - content preservation: content_code(pred) ≈ content_code(src)
    ([04-9]: was content_code(src) vs content_code(tgt) = identically zero
     because tgt=src; now uses pred — the model's output — which differs)
  - speaker classify: auxiliary CE on ref_embed, prevents encoder collapse
    ([04-11]: now documented in config, was hidden default 0.5)
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


def load_latent_corpus(
    data_dir: str, max_frames: int = 400, min_frames: int = 30
):
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
            if latent.shape[1] < min_frames:
                continue
            if latent.shape[1] > max_frames:
                latent = latent[:, :max_frames]
            speakers.setdefault(spk, []).append(latent.astype(np.float32))

    total = sum(len(v) for v in speakers.values())
    print(f"Loaded {total} latents from {len(speakers)} speakers", flush=True)
    return speakers


def make_batch(
    speakers: dict,
    batch_size: int,
    max_frames: int,
    device: torch.device,
    spk2idx: dict,
    cross_speaker_prob: float = 0.0,
):
    """Sample a warm-start training batch.

    Teacher-free AutoVC warm-start ([04-10] revision):

    - **Reconstruction role** (default, ``cross_speaker_prob=0``):
      src = tgt = utterance from speaker A.
      ref = a *different* utterance from the same speaker A (if available),
      so the speaker encoder generalizes. Target = src (autoencode).

    - **Cross-speaker regularizer** (``cross_speaker_prob > 0``, optional):
      src = tgt from speaker A, ref from speaker B.
      The model must reconstruct A's latent despite being fed B's reference.
      This is a regularizer that forces the bottleneck to carry
      speaker-invariant content — it does NOT produce "A with B's timbre"
      (that requires a teacher and is deferred to Phase C flow matching).
    """
    spk_list = list(speakers.keys())
    src_list = []
    ref_list = []
    tgt_list = []
    ref_spk_idx = []

    for _ in range(batch_size):
        src_spk = spk_list[np.random.randint(0, len(spk_list))]
        src_utts = speakers[src_spk]
        src_idx = np.random.randint(0, len(src_utts))
        src = src_utts[src_idx]
        src_list.append(src)

        use_cross = (
            cross_speaker_prob > 0.0
            and np.random.random() < cross_speaker_prob
            and len(spk_list) > 1
        )

        if use_cross:
            ref_spk = src_spk
            while ref_spk == src_spk:
                ref_spk = spk_list[np.random.randint(0, len(spk_list))]
        else:
            ref_spk = src_spk

        ref_utts = speakers[ref_spk]
        if ref_spk == src_spk and len(ref_utts) > 1:
            ref_idx = np.random.randint(0, len(ref_utts))
            if ref_idx == src_idx and len(ref_utts) > 1:
                ref_idx = (ref_idx + 1) % len(ref_utts)
            ref = ref_utts[ref_idx]
        else:
            ref = ref_utts[np.random.randint(0, len(ref_utts))]

        ref_list.append(ref)
        ref_spk_idx.append(spk2idx[ref_spk])
        # Target = source (autoencode). Cross-speaker role does NOT change
        # the target — there is no teacher to create "src with ref timbre".
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

    ref_idx = torch.tensor(ref_spk_idx, dtype=torch.long, device=device)
    return src.to(device), tgt.to(device), ref.to(device), ref_idx


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
    min_frames = cfg["data"].get("min_utterance_frames", 30)

    # Role assignment ([04-10]): cross-speaker regularizer probability.
    role_cfg = cfg.get("role_assignment", {})
    cross_speaker_prob = role_cfg.get("cross_speaker", 0.0)

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
    speakers = load_latent_corpus(data_dir, max_frames, min_frames)
    spk2idx = {spk: i for i, spk in enumerate(sorted(speakers.keys()))}
    num_speakers = len(spk2idx)

    # Model
    model = Converter(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Converter parameters: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)
    if cross_speaker_prob > 0:
        print(f"Cross-speaker regularizer: prob={cross_speaker_prob}", flush=True)

    # Speaker classifier (training-only, prevents speaker-encoder collapse)
    spk_classifier = torch.nn.Linear(model_cfg.speaker_embed_dim, num_speakers).to(device)
    print(f"Speaker classifier: {num_speakers} speakers", flush=True)

    # WavLM-SV teacher distillation (09-B2): train-only projection head.
    # SpeakerEncoder output (256-dim) → projection → 768-dim → cosine loss vs teacher.
    # At inference, only SpeakerEncoder (p1,p2) is used. Projection is discarded.
    wavlm_cache_path = os.path.join(os.path.dirname(data_dir.rstrip("/")), "wavlm_sv_embeddings.pkl")
    distill_proj = None
    idx2teacher = None
    if os.path.exists(wavlm_cache_path):
        import pickle
        with open(wavlm_cache_path, "rb") as f:
            wavlm_cache = pickle.load(f)
        spk_avg = {}
        for key, emb in wavlm_cache.items():
            spk = key.split("/")[0]
            spk_avg.setdefault(spk, []).append(emb)
        idx2teacher = torch.zeros(num_speakers, 768)
        idx2spk = {v: k for k, v in spk2idx.items()}
        for i in range(num_speakers):
            spk = idx2spk[i]
            if spk in spk_avg:
                idx2teacher[i] = torch.from_numpy(np.mean(spk_avg[spk], axis=0))
        idx2teacher = F.normalize(idx2teacher, dim=-1).to(device)
        distill_proj = torch.nn.Linear(model_cfg.speaker_embed_dim, 768).to(device)
        torch.nn.init.xavier_normal_(distill_proj.weight)
        torch.nn.init.zeros_(distill_proj.bias)
        distill_weight = loss_cfg.get("distill_cosine", 1.0)
        print(f"WavLM-SV distillation: {len(spk_avg)} speakers, weight={distill_weight}", flush=True)
    else:
        distill_weight = 0.0
        print("WavLM-SV cache not found, skipping distillation", flush=True)

    # Init from checkpoint if specified
    if "init_from" in train_cfg and train_cfg["init_from"]:
        ckpt = torch.load(train_cfg["init_from"], map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Initialized from {train_cfg['init_from']}", flush=True)

    optim_params = list(model.parameters()) + list(spk_classifier.parameters())
    if distill_proj is not None:
        optim_params += list(distill_proj.parameters())
    optim = torch.optim.AdamW(
        optim_params,
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
    spk_classifier.train()
    losses_log = {"total": [], "recon": [], "spk": [], "content": [], "cls": [], "distill": []}

    print(f"Starting warm-start training for {max_steps} steps...", flush=True)
    for step in range(1, max_steps + 1):
        src, tgt, ref, ref_idx = make_batch(
            speakers, batch_size, max_frames, device, spk2idx, cross_speaker_prob
        )

        optim.zero_grad()
        pred = model(src, ref)

        # Losses

        # 1. Reconstruction L1: pred ≈ tgt(=src)
        loss_recon = F.l1_loss(pred, tgt) * loss_cfg["reconstruction_l1"]

        # 2. Speaker consistency ([04-8]): pred speaker ≈ REFERENCE speaker.
        #    Was targeting src/tgt speaker (identical in reconstruction), but
        #    now explicit so cross-speaker regularizer roles push pred toward
        #    the reference's speaker identity.
        with torch.no_grad():
            ref_embed = model.speaker_embedding(ref)
        pred_embed = model.speaker_embedding(pred)
        loss_spk = (
            1.0 - F.cosine_similarity(pred_embed, ref_embed, dim=-1).mean()
        ) * loss_cfg.get("speaker_consistency", 0.5)

        # 3. Content preservation ([04-9]): content_code(pred) ≈ content_code(src).
        #    Was content_code(src) vs content_code(tgt=src) = identically zero.
        #    Now uses pred (the model's output) which differs from src, so the
        #    loss is meaningful: the bottleneck content of the output should
        #    match the bottleneck content of the input.
        content_src = model.content_code(src)
        content_pred = model.content_code(pred)
        loss_content = F.l1_loss(content_pred, content_src.detach()) * loss_cfg.get(
            "content_preservation", 0.3
        )

        # 4. Speaker classification auxiliary loss ([04-11]): prevents the
        #    speaker encoder from collapsing. Now documented in config.
        ref_embed_for_cls = model.speaker_embedding(ref)
        logits = spk_classifier(ref_embed_for_cls)
        loss_cls = F.cross_entropy(logits, ref_idx) * loss_cfg.get(
            "speaker_classify", 0.5
        )

        # 5. WavLM-SV distillation (09-B2): project SpeakerEncoder output
        #    to 768-dim and match teacher speaker-level average embedding.
        loss_distill = torch.tensor(0.0, device=device)
        if distill_proj is not None and distill_weight > 0:
            ref_embed_for_distill = model.speaker_embedding(ref)
            projected = F.normalize(distill_proj(ref_embed_for_distill), dim=-1)
            teacher = idx2teacher[ref_idx]
            loss_distill = (1.0 - F.cosine_similarity(projected, teacher, dim=-1).mean()) * distill_weight

        loss = loss_recon + loss_spk + loss_content + loss_cls + loss_distill
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        scheduler.step()

        losses_log["total"].append(loss.item())
        losses_log["recon"].append(loss_recon.item())
        losses_log["spk"].append(loss_spk.item())
        losses_log["content"].append(loss_content.item())
        losses_log["cls"].append(loss_cls.item())
        losses_log["distill"].append(loss_distill.item())

        if step % 100 == 0:
            avg = {k: np.mean(v[-100:]) for k, v in losses_log.items()}
            print(
                f"step {step}/{max_steps} | loss={avg['total']:.4f} "
                f"recon={avg['recon']:.4f} spk={avg['spk']:.4f} "
                f"content={avg['content']:.4f} cls={avg['cls']:.4f} "
                f"distill={avg['distill']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
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
