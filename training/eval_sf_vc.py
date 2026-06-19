"""
SF-VC Evaluation with F0 shift.

Computes per-speaker mean F0 from MC cache.
Applies F0 shift before WORLD synthesis.
"""
import sys, os, json, time, pickle, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import pyworld as world
import pysptk as sptk
import librosa

DEVICE = torch.device("cuda")
MC_DIM = 25
ALPHA = 0.410
FFTL = 2048
SR = 16000
FP = 5.0

VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")
PAIRS_DIR = Path("data/sf_pairs")


def load_speaker_embeddings():
    cache_path = Path("data/wavlm_sv_embeddings.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    spk_avg = {}
    for key, emb in cache.items():
        spk = key.split("/")[0]
        spk_avg.setdefault(spk, []).append(emb)
    return {spk: torch.from_numpy(np.mean(embs, axis=0)).float() for spk, embs in spk_avg.items()}


def compute_speaker_f0_stats():
    spk_f0 = defaultdict(list)
    for spk_dir in sorted(MC_CACHE.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for npz_path in spk_dir.glob("*.npz"):
            try:
                data = np.load(npz_path)
                f0 = data["f0"]
                voiced = f0[f0 > 0]
                if len(voiced) > 0:
                    spk_f0[spk].extend(voiced.tolist())
            except:
                continue
    stats = {}
    for spk, f0s in spk_f0.items():
        f0s = np.array(f0s)
        stats[spk] = float(np.exp(np.mean(np.log(f0s[f0s > 0]))))
    return stats


def shift_f0(f0, tgt_mean_f0):
    voiced = f0[f0 > 0]
    if len(voiced) == 0 or tgt_mean_f0 <= 0:
        return f0.astype(np.float64)
    src_mean = float(np.exp(np.mean(np.log(voiced))))
    ratio = tgt_mean_f0 / src_mean
    return np.where(f0 > 0, f0 * ratio, 0.0).astype(np.float64)


def synth_wav(mc, f0, codeap, sr=SR, fp=FP):
    mc = np.ascontiguousarray(mc, dtype=np.float64)
    sp = sptk.mc2sp(mc, ALPHA, FFTL)
    codeap = np.ascontiguousarray(codeap, dtype=np.float64)
    if codeap.ndim == 2 and codeap.shape[1] > 0:
        ap = world.decode_aperiodicity(codeap, sr, FFTL)
    else:
        ap = np.ones((len(mc), FFTL // 2 + 1), dtype=np.float64)
    f0 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f0, sp, ap, sr, frame_period=fp).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_eval", type=int, default=30)
    parser.add_argument("--f0_shift", action="store_true", default=True)
    parser.add_argument("--no_f0_shift", dest="f0_shift", action="store_false")
    args = parser.parse_args()

    print("=== SF-VC Evaluation (with F0 shift) ===\n")

    spk_emb = load_speaker_embeddings()
    spk_dim = next(iter(spk_emb.values())).shape[0]

    print("Computing per-speaker F0 stats...")
    f0_stats = compute_speaker_f0_stats()
    print(f"  {len(f0_stats)} speakers")

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    cfg = ckpt["config"]

    from train_sf_vc import EnvelopeConverter
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

    pair_files = sorted(PAIRS_DIR.glob("pair_*.npz"))
    n_eval = min(args.n_eval, len(pair_files))
    print(f"Evaluating {n_eval} pairs (F0 shift: {args.f0_shift})\n")

    tgt_scores, src_scores = [], []
    tgt_scores_nof0, src_scores_nof0 = [], []

    for i in range(n_eval):
        data = np.load(pair_files[i], allow_pickle=True)

        mc_src = data["mc_src"]
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

        tgt_mean_f0 = f0_stats.get(spk_tgt, 200.0)
        f0_original = f0_src[:T].astype(np.float64)
        f0_shifted = shift_f0(f0_src[:T], tgt_mean_f0) if args.f0_shift else f0_original

        wav_syn_f0 = synth_wav(mc_pred_np, f0_shifted, codeap_tgt[:T])
        wav_syn_nof0 = synth_wav(mc_pred_np, f0_original, codeap_tgt[:T]) if args.f0_shift else None

        src_wavs = list((VCTK_WAV / spk_src).glob("*.wav"))
        tgt_wavs = list((VCTK_WAV / spk_tgt).glob("*.wav"))
        if not src_wavs or not tgt_wavs:
            continue

        wav_src, sr = sf.read(str(src_wavs[0]), dtype="float32")
        wav_tgt, sr = sf.read(str(tgt_wavs[0]), dtype="float32")
        if sr != SR:
            wav_src = librosa.resample(wav_src, orig_sr=sr, target_sr=SR)
            wav_tgt = librosa.resample(wav_tgt, orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            e_src = secs_model.encode_batch(torch.from_numpy(wav_src.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = secs_model.encode_batch(torch.from_numpy(wav_tgt.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_syn = secs_model.encode_batch(torch.from_numpy(wav_syn_f0.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)

        tgt_sim = F.cosine_similarity(e_tgt, e_syn, dim=-1).item()
        src_sim = F.cosine_similarity(e_src, e_syn, dim=-1).item()
        tgt_scores.append(tgt_sim)
        src_scores.append(src_sim)

        if wav_syn_nof0 is not None:
            with torch.no_grad():
                e_syn2 = secs_model.encode_batch(torch.from_numpy(wav_syn_nof0.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            tgt_scores_nof0.append(F.cosine_similarity(e_tgt, e_syn2, dim=-1).item())
            src_scores_nof0.append(F.cosine_similarity(e_src, e_syn2, dim=-1).item())

        tag = f"F0shift" if args.f0_shift else "noF0"
        print(f"  [{i+1}/{n_eval}] {spk_src}→{spk_tgt}: tgt={tgt_sim:.3f} src={src_sim:.3f} ({tag})", flush=True)

    tgt_arr = np.array(tgt_scores)
    src_arr = np.array(src_scores)
    print(f"\n=== Results (F0 shift: {args.f0_shift}) ===")
    print(f"SECS(target): {tgt_arr.mean():.4f} ± {tgt_arr.std():.4f}")
    print(f"SECS(source): {src_arr.mean():.4f} ± {src_arr.std():.4f}")
    print(f"Separation:  {tgt_arr.mean() - src_arr.mean():.4f}")

    if tgt_scores_nof0:
        tgt_nof0 = np.array(tgt_scores_nof0)
        src_nof0 = np.array(src_scores_nof0)
        print(f"\n=== Results (no F0 shift) ===")
        print(f"SECS(target): {tgt_nof0.mean():.4f} ± {tgt_nof0.std():.4f}")
        print(f"SECS(source): {src_nof0.mean():.4f} ± {src_nof0.std():.4f}")

    results = {
        "secs_tgt_mean": float(tgt_arr.mean()),
        "secs_tgt_std": float(tgt_arr.std()),
        "secs_src_mean": float(src_arr.mean()),
        "secs_src_std": float(src_arr.std()),
        "f0_shift": args.f0_shift,
        "step": cfg.get("step", 0),
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_f0.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
