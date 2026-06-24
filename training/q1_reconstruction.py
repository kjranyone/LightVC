"""
Q1: Same-Speaker Reconstruction — minimal analysis-resynthesis pipeline.

content = WavLM L6 (1024-dim, 50Hz) → interpolate to 86Hz
speaker = WavLM-SV (256-dim, cached)
target = BigVGAN mel (128-band, 86Hz)

Predictor: Conv1d stack with FiLM conditioning → mel [128, T]

Tests: can we reconstruct mel from content + speaker?
If SECS(self-recon) > 0.7 through BigVGAN, the approach works.
"""
import sys, os, csv, pickle, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from bigvgan import BigVGAN, mel_spectrogram
from bigvgan.env import AttrDict

DEVICE = torch.device("cuda")
VCTK_WAV = Path("../data/vctk_200")


class MelPredictor(nn.Module):
    def __init__(self, content_dim=1024, speaker_dim=256, mel_dim=128, hidden=512, n_blocks=4):
        super().__init__()
        self.proj_in = nn.Conv1d(content_dim, hidden, 1)
        self.film = nn.Linear(speaker_dim, hidden * 2)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            dilation = 2 ** (i % 3)
            self.blocks.append(nn.ModuleDict({
                "conv1": nn.Conv1d(hidden, hidden, 7, dilation=dilation, padding=6*dilation),
                "conv2": nn.Conv1d(hidden, hidden, 1),
            }))
        self.proj_out = nn.Conv1d(hidden, mel_dim, 1)

    def forward(self, content, speaker_embed):
        h = self.proj_in(content)
        gb = self.film(speaker_embed)
        gamma, beta = gb.chunk(2, dim=-1)
        h = h * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

        for block in self.blocks:
            res = h
            h = F.gelu(block["conv1"](h))
            h = block["conv2"](h)
            h = h + res

        return self.proj_out(h)


def load_wavlm_sv_cache():
    cache_path = Path("data/wavlm_sv_embeddings.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    spk_avg = {}
    for key, emb in cache.items():
        spk = key.split("/")[0]
        spk_avg.setdefault(spk, []).append(emb)
    return {spk: torch.from_numpy(np.mean(emb, axis=0)).float() for spk, emb in spk_avg.items()}


def load_bigvgan():
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download("nvidia/bigvgan_v2_44khz_128band_512x")
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    h = AttrDict(config)
    if h.fmax is None:
        h.fmax = h.sampling_rate // 2
    m = BigVGAN(h, use_cuda_kernel=False)
    weights_path = os.path.join(model_dir, "bigvgan_generator.pt")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "generator" in state:
        state = state["generator"]
    m.load_state_dict(state, strict=True)
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, h


def make_batch(speakers, spk_list, batch_size, max_mel_frames, device, sv_cache, h):
    src_content_list, src_spk_list_ids, tgt_mel_list = [], [], []
    for _ in range(batch_size):
        spk = spk_list[np.random.randint(0, len(spk_list))]
        utts = speakers[spk]
        utt = utts[np.random.randint(0, len(utts))]

        wav_path = utt["wav"]
        wlm_path = utt["wlm"]

        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != h.sampling_rate:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=h.sampling_rate)

        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
        if wav_t.shape[-1] % h.hop_size != 0:
            wav_t = wav_t[..., :wav_t.shape[-1] // h.hop_size * h.hop_size]

        with torch.no_grad():
            mel = mel_spectrogram(
                wav_t, h.n_fft, h.num_mels, h.sampling_rate,
                h.hop_size, h.win_size, h.fmin, h.fmax,
            ).squeeze(0).cpu()

        T_mel = min(mel.shape[1], max_mel_frames)
        mel = mel[:, :T_mel]

        wlm = np.load(str(wlm)).astype(np.float32)
        T_wlm = wlm.shape[0]
        T_mel_from_wlm = int(T_mel * 50 / 86)
        T_wlm_use = min(T_wlm, T_mel_from_wlm)
        wlm_use = wlm[:T_wlm_use]

        wlm_resampled = np.zeros((T_mel, wlm_use.shape[1]), dtype=np.float32)
        for t in range(T_mel):
            src_t = t * 50 / 86
            lo = int(src_t)
            hi = min(lo + 1, T_wlm_use - 1)
            w = src_t - lo
            wlm_resampled[t] = (1 - w) * wlm_use[lo] + w * wlm_use[hi]

        src_content_list.append(torch.from_numpy(wlm_resampled.T))
        src_spk_list_ids.append(spk)
        tgt_mel_list.append(mel)

    min_t = min(c.shape[1] for c in src_content_list)

    content = torch.stack([c[:, :min_t] for c in src_content_list]).to(device)
    mel = torch.stack([m[:, :min_t] for m in tgt_mel_list]).to(device)
    speaker = torch.stack([sv_cache[s] for s in src_spk_list_ids]).to(device)

    return content, speaker, mel


def train(args):
    h_cfg = {"lr": 1e-4, "batch_size": 8, "max_steps": 20000, "save_every": 2000}

    print("Loading BigVGAN...")
    vocoder, vh = load_bigvgan()

    print("Loading WavLM-SV cache...")
    sv_cache = load_wavlm_sv_cache()

    print("Building dataset index...")
    speakers = {}
    for spk_dir in sorted(VCTK_WAV.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for wav_path in spk_dir.glob("*.wav"):
            wlm_path = Path("data/wavlm_l6") / spk / (wav_path.stem + ".npy")
            if wlm_path.exists():
                speakers.setdefault(spk, []).append({
                    "wav": str(wav_path),
                    "wlm": str(wlm_path),
                })
    spk_list = sorted(speakers.keys())
    print(f"  {sum(len(v) for v in speakers.values())} utterances, {len(spk_list)} speakers")

    model = MelPredictor(content_dim=1024, speaker_dim=256, mel_dim=vh.num_mels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MelPredictor: {n_params:,} ({n_params/1e6:.1f}M)")

    optim = torch.optim.AdamW(model.parameters(), lr=h_cfg["lr"], betas=(0.8, 0.99), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9999)

    os.makedirs(args.output, exist_ok=True)
    losses = []

    print(f"Training for {h_cfg['max_steps']} steps (B={h_cfg['batch_size']})...")
    for step in range(1, h_cfg["max_steps"] + 1):
        content, speaker, mel_tgt = make_batch(
            speakers, spk_list, h_cfg["batch_size"], 200, DEVICE, sv_cache, vh
        )

        optim.zero_grad()
        mel_pred = model(content, speaker)
        loss = F.l1_loss(mel_pred, mel_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        scheduler.step()

        losses.append(loss.item())
        if step % 100 == 0:
            avg = np.mean(losses[-100:])
            print(f"step {step}/{h_cfg['max_steps']} | l1={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if step % h_cfg["save_every"] == 0:
            path = os.path.join(args.output, f"step_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "step": step}, path)
            torch.save({"model": model.state_dict(), "step": step}, os.path.join(args.output, "latest.pt"))
            print(f"  Saved: {path}", flush=True)

    best = os.path.join(args.output, "best.pt")
    torch.save({"model": model.state_dict(), "step": step}, best)
    print(f"\nDone. Checkpoint: {best}")


def evaluate(args):
    import librosa
    from speechbrain.inference.speaker import EncoderClassifier

    print("\n=== Q1 Evaluation ===")
    print("Loading models...")
    vocoder, vh = load_bigvgan()
    sv_cache = load_wavlm_sv_cache()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = MelPredictor(content_dim=1024, speaker_dim=256, mel_dim=vh.num_mels).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    def get_embed(wav):
        w16 = librosa.resample(wav.astype(np.float32), orig_sr=44100, target_sr=16000)
        return secs_model.encode_batch(torch.from_numpy(w16).unsqueeze(0).to(DEVICE)).squeeze().cpu()

    test_files = sorted(VCTK_WAV.glob("p225/*.wav"))[:5]
    scores = []

    for wav_path in test_files:
        spk = wav_path.parent.name
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if sr != vh.sampling_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=vh.sampling_rate)

        wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(DEVICE)
        if wav_t.shape[-1] % vh.hop_size != 0:
            wav_t = wav_t[..., :wav_t.shape[-1] // vh.hop_size * vh.hop_size]

        wlm_path = Path("data/wavlm_l6") / spk / (wav_path.stem + ".npy")
        wlm = np.load(str(wlm_path)).astype(np.float32)
        T_mel = wav_t.shape[-1] // vh.hop_size

        wlm_resampled = np.zeros((T_mel, wlm.shape[1]), dtype=np.float32)
        for t in range(T_mel):
            src_t = t * 50 / 86
            lo = min(int(src_t), wlm.shape[0] - 1)
            hi = min(lo + 1, wlm.shape[0] - 1)
            w = src_t - int(src_t)
            wlm_resampled[t] = (1 - w) * wlm[lo] + w * wlm[hi]

        content = torch.from_numpy(wlm_resampled.T).unsqueeze(0).to(DEVICE)
        speaker = sv_cache[spk].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            mel_pred = model(content, speaker)
            recon_wav = vocoder(mel_pred).squeeze().cpu().numpy()

        real_embed = get_embed(wav)
        recon_embed = get_embed(recon_wav)
        secs = F.cosine_similarity(real_embed.unsqueeze(0), recon_embed.unsqueeze(0), dim=-1).item()
        scores.append(secs)
        print(f"  {wav_path.name}: SECS={secs:.4f}")

    print(f"\nQ1 SECS: {np.mean(scores):.4f} (target > 0.70)")
    if np.mean(scores) > 0.70:
        print("PASS: Analysis-resynthesis reconstructs speaker identity")
    else:
        print("NEEDS WORK: Reconstruction quality insufficient")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    train_p = sub.add_parser("train")
    train_p.add_argument("--output", default="checkpoints/q1_melpredictor")
    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
