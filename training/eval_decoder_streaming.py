"""
Step 4: Decoder-only streaming eval.

Compares frozen vs fine-tuned decoder on short-window decoding.
Uses REAL audio (not VC), no adapter involved.

A. Frozen decoder: full decode → short decode comparison
B. Fine-tuned decoder: full decode → short decode comparison

Metrics: SNR (aligned), MCD

Usage:
  cd training
  uv run python eval_decoder_streaming.py --ckpt checkpoints/decoder_finetune/best.pt
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR
from eval_streaming import compute_mcd

HOP = 512
VCTK_WAV = Path("../data/vctk_200")


def aligned_snr(ref, est, max_lag=200):
    if len(ref) < len(est):
        ref = np.pad(ref, (0, len(est) - len(ref)))
    elif len(est) < len(ref):
        est = np.pad(est, (0, len(ref) - len(est)))
    corr = np.correlate(ref, est, mode="full")
    center = len(ref) - 1
    lo = max(0, center - max_lag)
    hi = min(len(corr), center + max_lag + 1)
    best_idx = np.argmax(np.abs(corr[lo:hi]))
    best_lag = best_idx - max_lag
    est_shifted = np.roll(est, best_lag)
    noise = ref - est_shifted
    signal_power = np.sum(ref ** 2) + 1e-10
    noise_power = np.sum(noise ** 2) + 1e-10
    snr = 10 * np.log10(signal_power / noise_power)
    return snr, best_lag


def load_finetuned_decoder(ckpt_path):
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        delta = ck["decoder_state"]
        full_sd = dac.state_dict()
        for k, v in delta.items():
            if k in full_sd:
                full_sd[k] = v.to(DEVICE)
        dac.load_state_dict(full_sd)
        print(f"Loaded fine-tuned weights from {ckpt_path} (epoch {ck.get('epoch', '?')})")
    else:
        print("Using frozen decoder (no fine-tune checkpoint)")
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


def load_wav_44k(p):
    import soundfile as sf
    import librosa
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def run_eval(args):
    print("=== Decoder-Only Streaming Eval ===\n")
    from transformers import AutoModel

    dac_frozen = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac_frozen.parameters():
        p.requires_grad_(False)

    dac_finetuned = load_finetuned_decoder(args.ckpt)

    wavs = sorted(VCTK_WAV.glob("*/*.wav"))
    rng = np.random.default_rng(123)
    selected = list(rng.choice(len(wavs), size=min(args.n_utts, len(wavs)), replace=False))

    print(f"Evaluating {len(selected)} utterances\n")

    windows = [int(w) for w in args.windows.split(",")]
    conditions = ["frozen", "finetuned"]
    results = {c: {w: [] for w in windows} for c in conditions}

    for ii, wi in enumerate(selected):
        wav = load_wav_44k(wavs[wi])
        if len(wav) < DAC_SR * 3:
            continue
        x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            z = dac_frozen.encoder(x)
        T = z.shape[2]

        for dec, cond in [(dac_frozen, "frozen"), (dac_finetuned, "finetuned")]:
            with torch.no_grad():
                audio_full = dec.decoder(z).squeeze().cpu().numpy()

            for w in windows:
                if w >= T:
                    continue
                starts = list(range(0, T - w, w))[:3]
                for s in starts:
                    z_chunk = z[:, :, s:s + w]
                    with torch.no_grad():
                        audio_short = dec.decoder(z_chunk).squeeze().cpu().numpy()

                    start_sample = s * HOP
                    end_sample = start_sample + w * HOP
                    ref_region = audio_full[start_sample:end_sample]

                    min_len = min(len(ref_region), len(audio_short))
                    ref_r = ref_region[:min_len]
                    short_r = audio_short[:min_len]

                    snr, lag = aligned_snr(ref_r, short_r)
                    mcd_v = compute_mcd(ref_r, short_r)["mcd"]

                    results[cond][w].append({"snr": snr, "lag": lag, "mcd": mcd_v})

        if (ii + 1) % 10 == 0:
            f4 = results["frozen"][4] if 4 in results["frozen"] else []
            t4 = results["finetuned"][4] if 4 in results["finetuned"] else []
            f_snr = np.mean([r["snr"] for r in f4]) if f4 else 0
            t_snr = np.mean([r["snr"] for r in t4]) if t4 else 0
            print(f"  [{ii+1}/{len(selected)}] 4f SNR: frozen={f_snr:.1f}dB finetuned={t_snr:.1f}dB", flush=True)

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"{'decoder':<12} {'window':>6} {'SNR (dB)':>10} {'MCD':>8} {'n':>4}")
    print("-" * 80)

    summary = {}
    for cond in conditions:
        for w in windows:
            rs = results[cond][w]
            if not rs:
                continue
            snr_mean = float(np.mean([r["snr"] for r in rs]))
            mcd_mean = float(np.mean([r["mcd"] for r in rs]))
            key = f"{cond}_{w}f"
            summary[key] = {"snr": snr_mean, "mcd": mcd_mean, "n": len(rs)}
            print(f"{cond:<12} {w:>5}f {snr_mean:>9.1f}dB {mcd_mean:>8.2f} {len(rs):>4}")
    print(f"{'='*80}")

    print("\nImprovement (finetuned - frozen):")
    for w in windows:
        f_key = f"frozen_{w}f"
        t_key = f"finetuned_{w}f"
        if f_key in summary and t_key in summary:
            d_snr = summary[t_key]["snr"] - summary[f_key]["snr"]
            d_mcd = summary[t_key]["mcd"] - summary[f_key]["mcd"]
            print(f"  {w}f: SNR {d_snr:+.1f}dB  MCD {d_mcd:+.2f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_pair": results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decoder-only streaming eval")
    parser.add_argument("--ckpt", default="checkpoints/decoder_finetune/best.pt")
    parser.add_argument("--n_utts", type=int, default=50)
    parser.add_argument("--windows", type=str, default="4,8")
    parser.add_argument("--output", default="../results/decoder_streaming_eval.json")
    args = parser.parse_args()
    run_eval(args)
