"""
R2-mini: DAC decoder last-2-blocks fine-tune for short-window robustness.

Teacher: frozen full DAC (immutable copy)
Student: same init, block.2+block.3 trainable, rest frozen

Loss:
  L_short_distill  (0.60): student short-window vs teacher full-window
  L_full_distill   (0.25): student full-window vs teacher full-window
  L_full_orig      (0.15): student full-window vs original audio

Usage:
  cd training
  uv run python train_decoder_finetune.py --n_utts 1000 --epochs 8
"""
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR

HOP = 512

VCTK_WAV = Path("../data/vctk_200")
CKPT_DIR = Path("checkpoints/decoder_finetune")


def load_wav_44k(p):
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def multi_scale_stft_l1(audio_a, audio_b):
    loss = audio_a.new_tensor(0.0)
    eps = 1e-7
    a = audio_a.squeeze(1) if audio_a.dim() == 3 else audio_a
    b = audio_b.squeeze(1) if audio_b.dim() == 3 else audio_b
    for n_fft in (512, 1024, 2048):
        hop = n_fft // 4
        win = torch.hann_window(n_fft, device=a.device)
        sa = torch.stft(a, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, return_complex=True)
        sb = torch.stft(b, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=win, return_complex=True)
        loss = loss + F.l1_loss(sa.abs(), sb.abs() + eps)
    return loss / 3.0


def load_dac_pair(warm_start=None):
    from transformers import AutoModel

    dac_teacher = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac_teacher.parameters():
        p.requires_grad_(False)

    dac_student = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE)
    for p in dac_student.parameters():
        p.requires_grad_(False)
    trainable = 0
    for name, p in dac_student.named_parameters():
        if "decoder.block.2." in name or "decoder.block.3." in name:
            p.requires_grad_(True)
            trainable += p.numel()
    print(f"Student trainable: {trainable:,} params ({trainable / 1e6:.2f}M)")

    if warm_start and Path(warm_start).exists():
        ck = torch.load(warm_start, map_location="cpu", weights_only=False)
        delta = ck["decoder_state"]
        full_sd = dac_student.state_dict()
        n_loaded = 0
        for k, v in delta.items():
            if k in full_sd:
                full_sd[k] = v.to(DEVICE)
                n_loaded += 1
        dac_student.load_state_dict(full_sd)
        print(f"Warm start: {n_loaded} tensors from {warm_start} (epoch {ck.get('epoch', '?')})")

    return dac_teacher, dac_student


def collect_files(n_utts, eval_speakers=None):
    all_wavs = sorted(VCTK_WAV.glob("*/*.wav"))
    if eval_speakers:
        eval_set = set(eval_speakers.split(","))
        train_wavs = [w for w in all_wavs if w.parent.name not in eval_set]
        eval_wavs = [w for w in all_wavs if w.parent.name in eval_set]
    else:
        train_wavs = all_wavs
        eval_wavs = []

    rng = np.random.default_rng(42)
    n = min(n_utts, len(train_wavs))
    indices = list(rng.choice(len(train_wavs), size=n, replace=False))
    train_selected = [train_wavs[i] for i in indices]

    speaker_info = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/speaker-info.txt")
    if speaker_info.exists():
        genders = {}
        for line in speaker_info.read_text().strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 3:
                genders[parts[0]] = parts[2]
        train_genders = [genders.get(w.parent.name.lstrip("p"), "?") for w in train_selected]
        n_f = train_genders.count("F")
        n_m = train_genders.count("M")
        print(f"Data: {len(train_selected)} train utterances")
        print(f"  Gender: F={n_f} M={n_m} other={len(train_genders)-n_f-n_m}")
        print(f"  Speakers: {len(set(w.parent.name for w in train_selected))}")
        if eval_wavs:
            print(f"  Eval: {len(eval_wavs)} utterances from {len(set(w.parent.name for w in eval_wavs))} held-out speakers")

    lengths = []
    import soundfile as sf
    for w in train_selected[:200]:
        info = sf.info(str(w))
        lengths.append(info.frames / info.samplerate)
    lengths = np.array(lengths)
    print(f"  Duration (200 sample): mean={lengths.mean():.1f}s median={np.median(lengths):.1f}s range=[{lengths.min():.1f}, {lengths.max():.1f}]s")

    return train_selected, eval_wavs


def train(args):
    print("=== R2 Decoder Fine-Tune ===\n")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dac_teacher, dac_student = load_dac_pair(warm_start=args.warm_start)

    files, eval_files = collect_files(args.n_utts, eval_speakers=args.eval_speakers)
    print(f"\nLR: {args.lr}, epochs: {args.epochs}, windows: {args.windows}")
    print(f"Loss weights: short={args.w_short} full_distill={args.w_full_distill} full_orig={args.w_full_orig}")
    if args.warm_start:
        print(f"Warm start: {args.warm_start}")
    print()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    trainable_params = [p for p in dac_student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(files))

    window_sizes = [int(w) for w in args.windows.split(",")]
    FIXED_LAG = 0

    for epoch in range(args.epochs):
        dac_student.train()
        meters = {"short": [], "full_d": [], "full_o": [], "total": []}
        t0 = time.time()

        for fi, wav_path in enumerate(files):
            wav = load_wav_44k(wav_path)
            if len(wav) < DAC_SR * 2:
                continue

            x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                z = dac_teacher.encoder(x)
            T = z.shape[2]

            with torch.no_grad():
                audio_full_teacher = dac_teacher.decoder(z).squeeze(1)

            loss_total = x.new_tensor(0.0)
            n_windows = 0
            loss_short_sum = x.new_tensor(0.0)

            for w in window_sizes:
                if w >= T:
                    continue
                starts = list(range(0, T - w, max(w, 1)))[:args.max_windows_per_size]
                for s in starts:
                    z_chunk = z[:, :, s:s + w].detach()
                    audio_short = dac_student.decoder(z_chunk)

                    start_sample = s * HOP + FIXED_LAG
                    end_sample = start_sample + w * HOP
                    if end_sample > audio_full_teacher.shape[-1]:
                        end_sample = audio_full_teacher.shape[-1]
                    ref_region = audio_full_teacher[:, start_sample:end_sample]

                    min_len = min(ref_region.shape[-1], audio_short.shape[-1])
                    loss_short_sum = loss_short_sum + multi_scale_stft_l1(
                        audio_short[:, :min_len], ref_region[:, :min_len]
                    )
                    n_windows += 1

            if n_windows > 0:
                loss_short = loss_short_sum / n_windows
            else:
                loss_short = x.new_tensor(0.0)

            audio_full_student = dac_student.decoder(z.detach())
            min_len = min(audio_full_student.shape[-1], audio_full_teacher.shape[-1], x.shape[-1])
            loss_full_distill = multi_scale_stft_l1(
                audio_full_student[:, :min_len], audio_full_teacher[:, :min_len]
            )
            loss_full_orig = multi_scale_stft_l1(
                audio_full_student[:, :min_len], x[:, :, :min_len].squeeze(1)
            )

            loss = (
                args.w_short * loss_short
                + args.w_full_distill * loss_full_distill
                + args.w_full_orig * loss_full_orig
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            opt.step()
            scheduler.step()

            meters["short"].append(float(loss_short.detach()))
            meters["full_d"].append(float(loss_full_distill.detach()))
            meters["full_o"].append(float(loss_full_orig.detach()))
            meters["total"].append(float(loss.detach()))

            if (fi + 1) % 50 == 0:
                el = time.time() - t0
                print(
                    f"E{epoch} [{fi+1}/{len(files)}] "
                    f"total={np.mean(meters['total'][-50:]):.4f} "
                    f"short={np.mean(meters['short'][-50:]):.4f} "
                    f"full_d={np.mean(meters['full_d'][-50:]):.4f} "
                    f"full_o={np.mean(meters['full_o'][-50:]):.4f} "
                    f"| {el:.0f}s",
                    flush=True,
                )

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch}: total={np.mean(meters['total']):.4f} "
            f"short={np.mean(meters['short']):.4f} "
            f"full_d={np.mean(meters['full_d']):.4f} "
            f"full_o={np.mean(meters['full_o']):.4f} "
            f"({elapsed:.0f}s)\n"
        )

        ckpt = {
            "decoder_state": {k: v.cpu() for k, v in dac_student.state_dict().items()
                              if "decoder.block.2." in k or "decoder.block.3." in k},
            "epoch": epoch,
            "args": vars(args),
            "metrics": {k: float(np.mean(v)) for k, v in meters.items()},
        }
        torch.save(ckpt, CKPT_DIR / "latest.pt")
        if epoch == 0 or np.mean(meters["total"]) < best_loss:
            best_loss = np.mean(meters["total"])
            torch.save(ckpt, CKPT_DIR / "best.pt")
            print(f"  saved best (loss={best_loss:.4f})")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {CKPT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R2-mini decoder fine-tune")
    parser.add_argument("--n_utts", type=int, default=1000)
    parser.add_argument("--warm_start", type=str, default=None,
                        help="path to R2-mini checkpoint for warm start")
    parser.add_argument("--eval_speakers", type=str, default=None,
                        help="comma-separated speaker IDs to hold out for eval (e.g. p361,p364)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=0.5)
    parser.add_argument("--windows", type=str, default="4,8",
                        help="comma-separated window sizes in frames")
    parser.add_argument("--max_windows_per_size", type=int, default=4,
                        help="max short-window chunks per utterance per window size")
    parser.add_argument("--w_short", type=float, default=0.60)
    parser.add_argument("--w_full_distill", type=float, default=0.25)
    parser.add_argument("--w_full_orig", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    best_loss = float("inf")
    train(args)
