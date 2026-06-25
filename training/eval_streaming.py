"""
Streaming evaluation: SECS/margin/SNR + F0/CER/MCD on offline vs streaming.

Usage:
  cd training
  uv run python eval_streaming.py --n_pairs 25
  uv run python eval_streaming.py --n_pairs 200 --data_dir ../data/phase3_10k/eval
  uv run python eval_streaming.py --n_pairs 25 --skip_whisper   # skip CER
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from train_phase3b import (
    DEVICE, DAC_SR, SECS_SR, load_dac, load_ecapa,
    resample_16k, ecapa_embed, soft_rvq_requantize,
    hard_rvq_requantize,
)
from train_phase3c_adapter import TimbreAdapter
from export_streaming_samples import (
    streaming_pipeline, offline_pipeline, load_adapter,
    quantize_q0, aligned_snr,
)

HOP = 512
TAU = 5.0
VCTK_TEXT_ROOT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")


# =========================================================================
# SECS (speaker embedding cosine similarity)
# =========================================================================

def compute_secs(ecapa, audio_tensor, timbre, source_emb):
    audio_16k = resample_16k(audio_tensor)
    if audio_16k.shape[-1] < 8000:
        return None
    emb = ecapa_embed(ecapa, audio_16k)
    t = F.cosine_similarity(emb, timbre, dim=-1).mean().item()
    s = F.cosine_similarity(emb, source_emb, dim=-1).mean().item()
    return {"secs_target": t, "secs_source": s, "margin": t - s}


# =========================================================================
# F0 correlation + RMSE (pyworld)
# =========================================================================

def extract_f0(audio_np: np.ndarray, sr: int = DAC_SR, frame_period: float = 5.0):
    import pyworld as pw
    audio_f64 = audio_np.astype(np.float64)
    _f0, tnorm = pw.dio(audio_f64, sr, frame_period=frame_period)
    f0 = pw.stonemask(audio_f64, _f0, tnorm, sr)
    voiced = f0 > 0
    return f0, voiced


def compute_f0_metrics(ref_np: np.ndarray, est_np: np.ndarray):
    f0_ref, v_ref = extract_f0(ref_np)
    f0_est, v_est = extract_f0(est_np)
    n = min(len(f0_ref), len(f0_est))
    f0_ref, v_ref = f0_ref[:n], v_ref[:n]
    f0_est, v_est = f0_est[:n], v_est[:n]
    both_voiced = v_ref & v_est
    if both_voiced.sum() < 10:
        return {"f0_corr": None, "f0_rmse_hz": None, "f0_voiced_overlap": float(both_voiced.mean())}
    r_vals = f0_ref[both_voiced]
    e_vals = f0_est[both_voiced]
    corr = float(np.corrcoef(r_vals, e_vals)[0, 1])
    rmse = float(np.sqrt(np.mean((r_vals - e_vals) ** 2)))
    return {
        "f0_corr": corr,
        "f0_rmse_hz": rmse,
        "f0_voiced_overlap": float(both_voiced.mean()),
    }


# =========================================================================
# Whisper CER (content preservation)
# =========================================================================

_whisper_pipe = None

def load_whisper():
    global _whisper_pipe
    if _whisper_pipe is not None:
        return _whisper_pipe
    from transformers import pipeline
    _whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    return _whisper_pipe


def load_vctk_text(src_spk: str, text_id: str) -> str | None:
    spk_num = src_spk.lstrip("p")
    txt_path = VCTK_TEXT_ROOT / src_spk / f"{src_spk}_{text_id}.txt"
    if txt_path.exists():
        return txt_path.read_text().strip()
    return None


def compute_cer(whisper_pipe, audio_np: np.ndarray, ref_text: str | None):
    from jiwer import cer as jiwer_cer
    audio_f32 = (audio_np.astype(np.float32)).squeeze()
    result = whisper_pipe({"raw": audio_f32, "sampling_rate": DAC_SR})
    hyp = result["text"].strip().lower()
    metrics = {"whisper_hyp": hyp}
    if ref_text:
        ref = ref_text.strip().lower()
        metrics["cer_vs_ref"] = float(jiwer_cer(ref, hyp))
    return metrics


# =========================================================================
# MCD (Mel-Cepstral Distortion) offline vs streaming
# =========================================================================

def compute_mcd(ref_np: np.ndarray, est_np: np.ndarray, sr: int = DAC_SR):
    import librosa
    min_len = min(len(ref_np), len(est_np))
    ref_np, est_np = ref_np[:min_len], est_np[:min_len]
    n_fft = 1024
    hop = 256
    ref_mfcc = librosa.feature.mfcc(
        y=ref_np.astype(np.float32), sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
    )
    est_mfcc = librosa.feature.mfcc(
        y=est_np.astype(np.float32), sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop,
    )
    n_frames = min(ref_mfcc.shape[1], est_mfcc.shape[1])
    diff = ref_mfcc[1:, :n_frames] - est_mfcc[1:, :n_frames]
    frame_dist = np.sqrt(np.sum(diff ** 2, axis=0))
    mcd = float(np.mean(frame_dist))
    return {"mcd": mcd}


# =========================================================================
# Main eval loop
# =========================================================================

def run_eval(args):
    print("=== Streaming Evaluation ===\n")
    dac = load_dac()
    ecapa = load_ecapa()
    adapter = load_adapter(args.adapter_ckpt)
    print(f"Adapter: {args.adapter_ckpt}")

    whisper_pipe = None
    if not args.skip_whisper:
        whisper_pipe = load_whisper()
        print("Whisper: openai/whisper-tiny.en loaded")
    else:
        print("Whisper: skipped (--skip_whisper)")

    print()

    files = sorted(Path(args.data_dir).glob("*.pt"))
    if args.n_pairs > 0:
        files = files[: args.n_pairs]
    print(f"Evaluating {len(files)} pairs\n")

    conditions = ["offline", "balanced_4f"]
    all_results = {c: [] for c in conditions}

    for pi, fpath in enumerate(files):
        d = torch.load(fpath, map_location="cpu")
        z_s = d["z_s"].float().unsqueeze(0).to(DEVICE)
        timbre = d["timbre"].float().squeeze().unsqueeze(0).to(DEVICE)
        src_spk = d.get("src_spk", "")
        text_id = d.get("text_id", "")

        with torch.no_grad():
            source_audio = dac.decoder(z_s).squeeze(1)
        source_emb = ecapa_embed(ecapa, resample_16k(source_audio))
        pcm_np = source_audio.squeeze().cpu().numpy()

        ref_text = None
        if whisper_pipe and src_spk and text_id:
            ref_text = load_vctk_text(src_spk, text_id)

        pair_results = {}

        # --- Offline ---
        with torch.no_grad():
            off_audio = offline_pipeline(dac, adapter, z_s, timbre)
        off_m = compute_secs(ecapa, off_audio, timbre, source_emb)
        off_np = off_audio.squeeze().cpu().numpy()
        if off_m:
            off_f0 = compute_f0_metrics(pcm_np, off_np)
            off_m.update(off_f0)
            off_mcd = compute_mcd(pcm_np, off_np)
            off_m.update(off_mcd)
            if whisper_pipe:
                off_cer = compute_cer(whisper_pipe, off_np, ref_text)
                off_m["cer_vs_ref"] = off_cer.get("cer_vs_ref")
            all_results["offline"].append(off_m)
            pair_results["offline"] = off_m

        # --- Balanced 4f streaming ---
        bal_audio = streaming_pipeline(dac, adapter, pcm_np, timbre, 4, 4)
        bal_tensor = torch.from_numpy(bal_audio).float().unsqueeze(0).to(DEVICE)
        bal_m = compute_secs(ecapa, bal_tensor, timbre, source_emb)
        snr_bal, lag_bal = aligned_snr(off_np, bal_audio)
        if bal_m:
            bal_m["snr_vs_offline"] = snr_bal
            bal_m["align_lag"] = lag_bal
            bal_f0 = compute_f0_metrics(off_np, bal_audio)
            bal_m.update(bal_f0)
            bal_mcd = compute_mcd(off_np, bal_audio)
            bal_m.update(bal_mcd)
            if whisper_pipe:
                bal_cer = compute_cer(whisper_pipe, bal_audio, ref_text)
                bal_m["cer_vs_ref"] = bal_cer.get("cer_vs_ref")
            all_results["balanced_4f"].append(bal_m)
            pair_results["balanced_4f"] = bal_m

        # --- Progress line ---
        off_margin = pair_results.get("offline", {}).get("margin", 0)
        bal_margin = pair_results.get("balanced_4f", {}).get("margin", 0)
        f0c = pair_results.get("balanced_4f", {}).get("f0_corr")
        cer_off = pair_results.get("offline", {}).get("cer_vs_ref")
        cer_bal = pair_results.get("balanced_4f", {}).get("cer_vs_ref")
        extra = ""
        if f0c is not None:
            extra += f"  F0r={f0c:.2f}"
        if cer_off is not None and cer_bal is not None:
            extra += f"  CER={cer_bal:.2f}"
        print(
            f"  [{pi+1:>3}/{len(files)}] {fpath.stem}  "
            f"off:{off_margin:+.3f}  bal:{bal_margin:+.3f}  "
            f"SNR={snr_bal:.1f}dB{extra}",
            flush=True,
        )

    # --- Summary table ---
    print(f"\n{'='*90}")
    hdr = f"{'condition':<16} {'target':>8} {'margin':>8} {'snr':>8} {'f0_corr':>8} {'f0_rmse':>8} {'mcd':>6} {'cer':>6} {'n':>4}"
    print(hdr)
    print("-" * 90)

    summary = {}
    for cond in conditions:
        rs = all_results[cond]
        if not rs:
            continue
        def safe_mean(key):
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        t_mean = safe_mean("secs_target")
        m_mean = safe_mean("margin")
        snr_mean = safe_mean("snr_vs_offline")
        f0c_mean = safe_mean("f0_corr")
        f0r_mean = safe_mean("f0_rmse_hz")
        mcd_mean = safe_mean("mcd")
        cer_mean = safe_mean("cer_vs_ref")

        summary[cond] = {
            "n": len(rs),
            "secs_target": t_mean,
            "margin_mean": m_mean,
            "snr_vs_offline_mean": snr_mean,
            "f0_corr_mean": f0c_mean,
            "f0_rmse_mean_hz": f0r_mean,
            "mcd_mean": mcd_mean,
            "cer_vs_ref_mean": cer_mean,
        }

        def fmt(v, sign=False):
            if v is None:
                return f"{'—':>8}"
            s = "+" if sign else ""
            return f"{v:{s}8.3f}"
        def fmt_s(v):
            if v is None:
                return f"{'—':>6}"
            return f"{v:6.2f}"

        print(
            f"{cond:<16} "
            f"{fmt(t_mean)} {fmt(m_mean, sign=True)} "
            f"{fmt(snr_mean)} "
            f"{fmt(f0c_mean)} {fmt(f0r_mean)} "
            f"{fmt_s(mcd_mean)} "
            f"{fmt_s(cer_mean) if cer_mean is not None else fmt_s(None)} "
            f"{len(rs):>4}"
        )
    print(f"{'='*90}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"summary": summary, "per_pair": all_results},
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming pipeline evaluation")
    parser.add_argument("--n_pairs", type=int, default=25,
                        help="number of eval pairs (-1 for all)")
    parser.add_argument("--data_dir", default="../data/phase3_10k/eval")
    parser.add_argument("--adapter_ckpt", default="checkpoints/phase3c_ao_b1_ecapa/best.pt")
    parser.add_argument("--output", default="../results/streaming_eval_full.json")
    parser.add_argument("--skip_whisper", action="store_true",
                        help="skip Whisper CER (saves time/GPU)")
    args = parser.parse_args()
    run_eval(args)
