"""
Phase 2a: Multi-metric oracle sweep (same-text, DTW-aligned)

中心問題: depth 0-2 の話者情報を content を壊さず target へ移す

3系統 × K=1..5 + hybrid:
  Target-led: target depth 0..K-1 + source rest re-quant
  Source-led: source depth 0..K-1 + target rest re-quant
  Hybrid:     depth-level mixing

指標:
  SECS:        ECAPA cosine (output vs target)
  Content CER: Whisper ASR edit distance vs source text
  F0 corr:     log-F0 Pearson (output vs source)
  Leakage:     ECAPA cosine (output vs source)
"""
import sys, json, time, re, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa
import pyworld as pw
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda")
DAC_SR = 44100
SECS_SR = 16000
N_PAIRS = 200
VCTK_WAV = Path("../data/vctk_200")
VCTK_TXT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")

N_DEPTHS = 9

CONFIGS = {
    "source_all": ["src"] * N_DEPTHS,
    "tgt_all_q": ["tgt"] * N_DEPTHS,

    "tgt_K1": ["tgt"] + ["requant"] * 8,
    "tgt_K2": ["tgt"] * 2 + ["requant"] * 7,
    "tgt_K3": ["tgt"] * 3 + ["requant"] * 6,
    "tgt_K4": ["tgt"] * 4 + ["requant"] * 5,
    "tgt_K5": ["tgt"] * 5 + ["requant"] * 4,

    "src_K1": ["src"] + ["requant_tgt"] * 8,
    "src_K2": ["src"] * 2 + ["requant_tgt"] * 7,
    "src_K3": ["src"] * 3 + ["requant_tgt"] * 6,
    "src_K4": ["src"] * 4 + ["requant_tgt"] * 5,
    "src_K5": ["src"] * 5 + ["requant_tgt"] * 4,

    "hyb_t0_sRest": ["tgt", "src", "src", "src", "src", "src", "src", "src", "src"],
    "hyb_t01_sRest": ["tgt", "tgt", "src", "src", "src", "src", "src", "src", "src"],
    "hyb_s0_t1_sRest": ["src", "tgt", "src", "src", "src", "src", "src", "src", "src"],
}

CONFIG_ORDER = list(CONFIGS.keys())


def find_pairs(n=200):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir():
            continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2:
            continue
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                sa, wa = utts[i]
                sb, wb = utts[j]
                if sa == sb:
                    continue
                pairs.append({
                    "src": sa, "src_wav": wa,
                    "tgt": sb, "tgt_wav": wb,
                    "text_id": tid,
                })
                if len(pairs) >= n:
                    return pairs
    return pairs


def load_vctk_text(speaker, text_id):
    txt_path = VCTK_TXT / speaker / f"{speaker}_{text_id}.txt"
    if txt_path.exists():
        return txt_path.read_text().strip()
    return ""


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def cer(reference, hypothesis):
    r = normalize_text(reference)
    h = normalize_text(hypothesis)
    if len(r) == 0:
        return 1.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n] / m


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


def load_whisper():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en"
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


@torch.no_grad()
def encode_dac(dac, wav_44k):
    x = torch.from_numpy(wav_44k).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return dac.encoder(x)


@torch.no_grad()
def decode_dac(dac, z):
    return dac.decoder(z).squeeze().cpu().numpy()


@torch.no_grad()
def quantize_full(dac, z):
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    residual = z.clone()
    q_list = []
    for d in range(n):
        q_out, _, _, _, _ = quantizers[d](residual)
        q_list.append(q_out)
        residual = residual - q_out
    return q_list, residual


@torch.no_grad()
def build_zq(dac, z_s, z_t_aligned, assign, q_s_list, q_t_list):
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    z_q = torch.zeros_like(z_s)
    for d in range(n):
        tag = assign[d]
        if tag == "src":
            z_q = z_q + q_s_list[d]
        elif tag == "tgt":
            z_q = z_q + q_t_list[d]
        elif tag == "requant":
            residual = z_s - z_q
            q_out, _, _, _, _ = quantizers[d](residual)
            z_q = z_q + q_out
        elif tag == "requant_tgt":
            residual = z_t_aligned - z_q
            q_out, _, _, _, _ = quantizers[d](residual)
            z_q = z_q + q_out
    return z_q


def align_latents(z_s, z_t):
    z_s_np = z_s.squeeze(0).cpu().numpy().T
    z_t_np = z_t.squeeze(0).cpu().numpy().T
    _, path = fastdtw(z_s_np, z_t_np, radius=15)
    T_s = len(z_s_np)
    T_t = len(z_t_np)
    src_to_tgt = np.zeros(T_s, dtype=int)
    for s, t in path:
        if s < T_s:
            src_to_tgt[s] = min(t, T_t - 1)
    for i in range(1, T_s):
        if src_to_tgt[i] == 0:
            src_to_tgt[i] = src_to_tgt[i - 1]
    z_t_np_aligned = z_t_np[src_to_tgt].T
    return torch.from_numpy(z_t_np_aligned).float().unsqueeze(0).to(DEVICE)


def load_wav_44k(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def extract_f0(wav_16k):
    wav_f64 = wav_16k.astype(np.float64)
    if len(wav_f64) < 512:
        return np.zeros(1)
    try:
        f0, _sp, _ap = pw.wav2world(wav_f64, 16000, frame_period=5.0)
        return f0
    except Exception:
        return np.zeros(1)


def f0_correlation(f0_a, f0_b):
    min_len = min(len(f0_a), len(f0_b))
    f0_a = f0_a[:min_len]
    f0_b = f0_b[:min_len]
    voiced = (f0_a > 1.0) & (f0_b > 1.0)
    if voiced.sum() < 10:
        return 0.0
    log_a = np.log(f0_a[voiced])
    log_b = np.log(f0_b[voiced])
    if log_a.std() < 1e-6 or log_b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(log_a, log_b)[0, 1])


@torch.no_grad()
def transcribe_batch(processor, model, audio_list_16k):
    max_len = max(len(a) for a in audio_list_16k)
    max_len = min(max_len, 16000 * 30)
    padded = np.zeros((len(audio_list_16k), max_len), dtype=np.float32)
    for i, a in enumerate(audio_list_16k):
        clipped = a[:max_len]
        padded[i, :len(clipped)] = clipped
    inputs = processor(padded, sampling_rate=16000, return_tensors="pt")
    feats = inputs.input_features.to(DEVICE)
    forced = model.generate(feats, max_new_tokens=80)
    texts = processor.batch_decode(forced, skip_special_tokens=True)
    return [t.strip() for t in texts]


def secs_embed(secs_model, wav_16k):
    wav_t = torch.from_numpy(wav_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return secs_model.encode_batch(wav_t).squeeze(0)


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()


def main():
    print("=== Phase 2a: Multi-metric Oracle Sweep ===\n")

    dac = load_dac()
    print(f"RVQ: {dac.quantizer.n_codebooks} codebooks")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    whisper_proc, whisper_model = load_whisper()
    print("Models loaded\n")

    pairs = find_pairs(N_PAIRS)
    print(f"Pairs: {len(pairs)}\n")

    results = defaultdict(lambda: defaultdict(list))
    t0 = time.time()

    for idx, p in enumerate(pairs):
        try:
            wav_s = load_wav_44k(p["src_wav"])
            wav_t = load_wav_44k(p["tgt_wav"])
            if len(wav_s) < DAC_SR or len(wav_t) < DAC_SR:
                continue

            source_text = load_vctk_text(p["src"], p["text_id"])
            if not source_text:
                continue

            z_s = encode_dac(dac, wav_s)
            z_t = encode_dac(dac, wav_t)
            z_t_aligned = align_latents(z_s, z_t)

            q_s_list, _ = quantize_full(dac, z_s)
            q_t_list, _ = quantize_full(dac, z_t_aligned)

            wav_s_16k = librosa.resample(
                wav_s.astype(np.float64), orig_sr=DAC_SR, target_sr=SECS_SR
            ).astype(np.float32)
            wav_t_16k = librosa.resample(
                wav_t.astype(np.float64), orig_sr=DAC_SR, target_sr=SECS_SR
            ).astype(np.float32)
            if len(wav_s_16k) < 8000 or len(wav_t_16k) < 8000:
                continue

            e_src = secs_embed(secs_model, wav_s_16k)
            e_tgt = secs_embed(secs_model, wav_t_16k)
            f0_src = extract_f0(wav_s_16k)

            audio_batch = []
            valid_configs = []
            for name in CONFIG_ORDER:
                assign = CONFIGS[name]
                z_q = build_zq(dac, z_s, z_t_aligned, assign, q_s_list, q_t_list)
                audio_44k = decode_dac(dac, z_q)
                if len(audio_44k) < DAC_SR * 0.3:
                    continue
                audio_16k = librosa.resample(
                    audio_44k.astype(np.float64), orig_sr=DAC_SR, target_sr=SECS_SR
                ).astype(np.float32)
                if len(audio_16k) < 1600:
                    continue
                audio_batch.append(audio_16k)
                valid_configs.append(name)

            asr_texts = transcribe_batch(whisper_proc, whisper_model, audio_batch)

            for ci, name in enumerate(valid_configs):
                audio_16k = audio_batch[ci]

                e_out = secs_embed(secs_model, audio_16k)
                secs_val = cosine(e_tgt, e_out)
                leak_val = cosine(e_src, e_out)

                cer_val = cer(source_text, asr_texts[ci])

                f0_out = extract_f0(audio_16k)
                f0c = f0_correlation(f0_src, f0_out)

                results[name]["secs"].append(secs_val)
                results[name]["cer"].append(cer_val)
                results[name]["f0"].append(f0c)
                results[name]["leak"].append(leak_val)

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (len(pairs) - idx - 1) / speed
            tk5 = results.get("tgt_K5", {})
            sk1 = results.get("src_K1", {})
            tk5_s = np.mean(tk5.get("secs", [0])[-20:]) if tk5.get("secs") else 0
            sk1_s = np.mean(sk1.get("secs", [0])[-20:]) if sk1.get("secs") else 0
            print(
                f"  [{idx+1}/{len(pairs)}] "
                f"tgt_K5_secs={np.mean(tk5_s):.3f} "
                f"src_K1_secs={np.mean(sk1_s):.3f} "
                f"| {speed:.1f}pair/s ETA {eta:.0f}s",
                flush=True,
            )

    # =========================================
    # Results table
    # =========================================
    print(f"\n{'='*100}")
    print(f"{'config':<20} {'SECS':>22} {'CER':>22} {'F0_corr':>22} {'Leakage':>22}")
    print(f"{'-'*110}")

    summary = {}
    for name in CONFIG_ORDER:
        if name not in results or len(results[name].get("secs", [])) == 0:
            continue
        row = {}
        parts = [f"{name:<20}"]
        for metric in ["secs", "cer", "f0", "leak"]:
            arr = np.array(results[name][metric])
            n = len(arr)
            boot = np.array([
                arr[np.random.choice(n, n, replace=True)].mean()
                for _ in range(500)
            ])
            row[metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)),
            }
            m = arr.mean()
            lo = np.percentile(boot, 2.5)
            hi = np.percentile(boot, 97.5)
            parts.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
        summary[name] = row
        print(f"{parts[0]} {parts[1]:>22} {parts[2]:>22} {parts[3]:>22} {parts[4]:>22}")

    print(f"\n{'='*60}")
    print("--- SECS-Content Tradeoff Summary ---\n")

    sa_cer = summary.get("source_all", {}).get("cer", {}).get("mean", 0)
    print(f"source_all CER baseline: {sa_cer:.3f}")
    print(f"(content degradation threshold: CER <= {sa_cer + 0.05:.3f})\n")

    print(f"{'config':<20} {'SECS':>8} {'CER':>8} {'F0':>8} {'Leak':>8} {'Go?':>6}")
    print(f"{'-'*60}")
    for name in CONFIG_ORDER:
        if name not in summary:
            continue
        s = summary[name]
        secs_m = s["secs"]["mean"]
        cer_m = s["cer"]["mean"]
        f0_m = s["f0"]["mean"]
        lk_m = s["leak"]["mean"]
        go = ""
        if name not in ("source_all", "tgt_all_q"):
            if secs_m >= 0.45 and cer_m <= sa_cer + 0.05:
                go = "GO"
            elif secs_m >= 0.45:
                go = "secs-only"
            elif cer_m <= sa_cer + 0.05:
                go = "content-only"
        print(f"{name:<20} {secs_m:>8.3f} {cer_m:>8.3f} {f0_m:>8.3f} {lk_m:>8.3f} {go:>6}")

    out = {}
    for name, row in summary.items():
        out[name] = {
            m: {"mean": v["mean"], "std": v["std"],
                "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
            for m, v in row.items()
        }
    Path("results").mkdir(exist_ok=True)
    with open("results/phase2a_oracle_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/phase2a_oracle_sweep.json")


if __name__ == "__main__":
    main()
