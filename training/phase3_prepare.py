"""
Phase 3 data preparation:
  For each same-text pair in VCTK:
    - DAC encode source + target
    - DTW-align target to source timeline
    - Extract q0_s, z_s, f0_s, energy_s
    - Extract target ECAPA embedding
    - Save as .pt

Output: data/phase3_10k/train/*.pt
"""
import sys, time, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import librosa
import pyworld as pw
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
DAC_SR = 44100
SECS_SR = 16000
DAC_FPS = 86.13
VCTK_WAV = Path("../data/vctk_200")
OUT_DIR = Path("../data/phase3")
N_PAIRS = 2000
HOLDOUT = 200


def find_pairs(source_dir, n=2000, seed=1234):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))

    tids = [tid for tid, utts in groups.items() if len({u[0] for u in utts}) >= 2]
    tids = sorted(tids)
    pairs = []
    seen = set()

    while len(pairs) < n and tids:
        order = rng.permutation(len(tids))
        added = 0
        for oi in order:
            tid = tids[int(oi)]
            utts = groups[tid]
            for _ in range(16):
                ia, ib = rng.choice(len(utts), size=2, replace=False)
                sa, wa = utts[int(ia)]
                sb, wb = utts[int(ib)]
                if sa == sb:
                    continue
                key = (tid, sa, sb)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "src": sa,
                    "src_wav": wa,
                    "tgt": sb,
                    "tgt_wav": wb,
                    "text_id": tid,
                })
                added += 1
                break
            if len(pairs) >= n:
                break
        if added == 0:
            break
    return pairs


def load_wav_44k(p):
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


@torch.no_grad()
def encode_dac(dac, wav):
    x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return dac.encoder(x)


@torch.no_grad()
def quantize_q0(dac, z):
    q, _, _, _, _ = dac.quantizer.quantizers[0](z.clone())
    return q


def dtw_align(z_s, z_t):
    zs = z_s.squeeze(0).cpu().numpy().T
    zt = z_t.squeeze(0).cpu().numpy().T
    _, path = fastdtw(zs, zt, radius=15)
    Ts, Tt = len(zs), len(zt)
    m = np.zeros(Ts, dtype=np.int64)
    for s, t in path:
        if s < Ts:
            m[s] = min(t, Tt - 1)
    for i in range(1, Ts):
        if m[i] == 0:
            m[i] = m[i - 1]
    zt_aligned = zt[m].T
    return torch.from_numpy(zt_aligned).float().unsqueeze(0).to(DEVICE)


def extract_f0_energy(wav_16k, T_dac):
    w = wav_16k.astype(np.float64)
    if len(w) < 512:
        return np.zeros(T_dac), np.zeros(T_dac)
    try:
        f0, _sp, _ap = pw.wav2world(w, 16000, frame_period=5.0)
    except Exception:
        return np.zeros(T_dac), np.zeros(T_dac)
    T_f0 = len(f0)
    idx = np.minimum(np.arange(T_dac) * T_f0 // max(T_dac, 1), T_f0 - 1)
    f0_interp = f0[idx]
    hop_44k = 512
    hop_16k = int(hop_44k * 16000 / DAC_SR)
    energy = np.zeros(T_dac)
    for i in range(T_dac):
        s = i * hop_16k
        e = min(s + hop_16k, len(w))
        if e > s:
            energy[i] = np.sqrt(np.mean(w[s:e] ** 2))
    return f0_interp, energy


def main():
    parser = argparse.ArgumentParser(description="Prepare Phase 3 same-text DAC pairs")
    parser.add_argument("--source", type=Path, default=VCTK_WAV)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--n_pairs", type=int, default=N_PAIRS)
    parser.add_argument("--holdout", type=int, default=HOLDOUT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("=== Phase 3 Data Preparation ===\n")
    print(f"source={args.source}")
    print(f"output={args.output}")
    print(f"n_pairs={args.n_pairs} holdout={args.holdout} seed={args.seed}")

    if not args.source.exists():
        raise FileNotFoundError(args.source)

    dac = load_dac()

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    pairs = find_pairs(args.source, args.n_pairs, args.seed)
    print(f"Total pairs: {len(pairs)}")
    print(f"Holdout (eval): first {args.holdout}")
    print(f"Train: {max(len(pairs) - args.holdout, 0)}")

    train_dir = args.output / "train"
    eval_dir = args.output / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_saved = 0

    for idx, p in enumerate(pairs):
        try:
            wav_s = load_wav_44k(p["src_wav"])
            wav_t = load_wav_44k(p["tgt_wav"])
            if len(wav_s) < DAC_SR or len(wav_t) < DAC_SR:
                continue

            z_s = encode_dac(dac, wav_s)
            z_t = encode_dac(dac, wav_t)
            Ts = z_s.shape[2]

            z_t_aligned = dtw_align(z_s, z_t)
            q0_s = quantize_q0(dac, z_s)

            w16s = librosa.resample(
                wav_s.astype(np.float64), orig_sr=DAC_SR,
                target_sr=SECS_SR).astype(np.float32)
            w16t = librosa.resample(
                wav_t.astype(np.float64), orig_sr=DAC_SR,
                target_sr=SECS_SR).astype(np.float32)
            if len(w16s) < 8000 or len(w16t) < 8000:
                continue

            f0_s, energy_s = extract_f0_energy(w16s, Ts)

            f0_norm = np.where(f0_s > 1, np.log(f0_s), 0).astype(np.float32)
            energy_norm = np.log(energy_s + 1e-8).astype(np.float32)
            f0_mean = f0_norm[f0_norm != 0].mean() if (f0_norm != 0).any() else 0
            f0_std = f0_norm[f0_norm != 0].std() if (f0_norm != 0).any() else 1
            f0_norm = np.where(f0_norm != 0, (f0_norm - f0_mean) / (f0_std + 1e-8), 0)
            energy_norm = (energy_norm - energy_norm.mean()) / (energy_norm.std() + 1e-8)

            with torch.no_grad():
                e_tgt = secs_model.encode_batch(
                    torch.from_numpy(w16t).unsqueeze(0).to(DEVICE)
                ).squeeze(0).cpu()

            data = {
                "z_s": z_s.squeeze(0).cpu().half(),
                "q0_s": q0_s.squeeze(0).cpu().half(),
                "z_t_aligned": z_t_aligned.squeeze(0).cpu().half(),
                "f0": torch.from_numpy(f0_norm),
                "energy": torch.from_numpy(energy_norm),
                "timbre": e_tgt,
                "src_spk": p["src"],
                "tgt_spk": p["tgt"],
                "text_id": p["text_id"],
            }

            out_dir = eval_dir if idx < args.holdout else train_dir
            out_path = out_dir / f"pair_{idx:05d}.pt"
            if out_path.exists() and not args.overwrite:
                n_saved += 1
                continue
            torch.save(data, out_path)
            n_saved += 1

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx + 1) % 200 == 0:
            el = time.time() - t0
            sp = (idx + 1) / el
            eta = (len(pairs) - idx - 1) / sp
            print(f"  [{idx+1}/{len(pairs)}] saved={n_saved} "
                  f"| {sp:.1f}p/s ETA {eta:.0f}s", flush=True)

    print(f"\nDone: {n_saved} pairs saved to {args.output}")
    train_count = len(list(train_dir.glob("*.pt")))
    eval_count = len(list(eval_dir.glob("*.pt")))
    print(f"  Train: {train_count}")
    print(f"  Eval: {eval_count}")


if __name__ == "__main__":
    main()
