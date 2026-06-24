"""
Phase 2c: Content-feature DTW diagnostic

Hypothesis: DTW oracle (0.686) vs frame NN (0.20) gap = temporal structure.
Wav2Vec2 hidden states give content-focused features for alignment.

Step 1: same-text Wav2Vec2-DTW (layers 6, 9, 12)
Step 2: compare DTW vs frame NN (same Wav2Vec2 features)
Step 3: measure layer effect

Conversion: q0_hat = q0_source; q1..8 = requantize(z_target_like - q0_source)

Go: same-text W2V2-DTW >= 0.55, CER <= 0.10
"""
import sys, json, time, re
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

W2V2_LAYERS = [6, 9, 12]
CONFIG_ORDER = [
    "dac_dtw",
    *[f"w2v2_l{l}_dtw" for l in W2V2_LAYERS],
    "w2v2_l9_nn",
    "random",
]


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
                pairs.append({"src": sa, "src_wav": wa,
                              "tgt": sb, "tgt_wav": wb, "text_id": tid})
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
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


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
        "openai/whisper-tiny.en").to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def load_wav2vec2():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h").to(DEVICE).eval()
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
    return q_list


@torch.no_grad()
def src_k1_convert(dac, q0_source, z_target_like):
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    z_q = q0_source.clone()
    residual = z_target_like - q0_source
    for d in range(1, n):
        q_out, _, _, _, _ = quantizers[d](residual)
        z_q = z_q + q_out
        residual = residual - q_out
    return z_q


def align_latents_dac(z_s, z_t):
    z_s_np = z_s.squeeze(0).cpu().numpy().T
    z_t_np = z_t.squeeze(0).cpu().numpy().T
    _, path = fastdtw(z_s_np, z_t_np, radius=15)
    T_s = len(z_s_np)
    T_t = len(z_t_np)
    s2t = np.zeros(T_s, dtype=int)
    for s, t in path:
        if s < T_s:
            s2t[s] = min(t, T_t - 1)
    for i in range(1, T_s):
        if s2t[i] == 0:
            s2t[i] = s2t[i - 1]
    z_t_np_aligned = z_t_np[s2t].T
    return torch.from_numpy(z_t_np_aligned).float().unsqueeze(0).to(DEVICE)


@torch.no_grad()
def extract_w2v2_hidden(wav_16k, processor, model):
    inputs = processor(wav_16k, sampling_rate=16000, return_tensors="pt")
    outputs = model.wav2vec2(
        inputs.input_values.to(DEVICE), output_hidden_states=True)
    return outputs.hidden_states


def interp_to_dac(feat, T_dac):
    if feat.shape[0] == T_dac:
        return feat
    x = feat.unsqueeze(0).transpose(1, 2)
    x = F.interpolate(x, size=T_dac, mode="linear", align_corners=False)
    return x.transpose(1, 2).squeeze(0)


def dtw_align_features(feat_s, feat_t, z_t):
    feat_s_np = F.normalize(feat_s, dim=-1).cpu().numpy()
    feat_t_np = F.normalize(feat_t, dim=-1).cpu().numpy()
    _, path = fastdtw(feat_s_np, feat_t_np, radius=15)
    T_s = len(feat_s_np)
    T_t = len(feat_t_np)
    s2t = np.zeros(T_s, dtype=int)
    for s, t in path:
        if s < T_s:
            s2t[s] = min(t, T_t - 1)
    for i in range(1, T_s):
        if s2t[i] == 0:
            s2t[i] = s2t[i - 1]
    z_t_frames = z_t.squeeze(0).T.cpu().numpy()
    z_t_aligned = z_t_frames[s2t].T
    return torch.from_numpy(z_t_aligned).float().unsqueeze(0).to(DEVICE)


def retrieve_nn_w2v2(feat_s, feat_t, z_t):
    sim = F.normalize(feat_s, dim=-1) @ F.normalize(feat_t, dim=-1).T
    best_j = sim.argmax(dim=-1)
    z_t_frames = z_t.squeeze(0).T
    return z_t_frames[best_j].T.unsqueeze(0)


def retrieve_random(z_s, z_t):
    T_s = z_s.shape[2]
    T_t = z_t.shape[2]
    z_t_f = z_t.squeeze(0).T
    idx = torch.randint(0, T_t, (T_s,), device=z_t.device)
    return z_t_f[idx].T.unsqueeze(0)


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
    print("=== Phase 2c: Content-Feature DTW Diagnostic ===\n")

    dac = load_dac()
    print(f"RVQ: {dac.quantizer.n_codebooks} codebooks")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    whisper_proc, whisper_model = load_whisper()
    w2v2_proc, w2v2_model = load_wav2vec2()
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
            T_s = z_s.shape[2]
            T_t = z_t.shape[2]

            wav_s_16k = librosa.resample(
                wav_s.astype(np.float64), orig_sr=DAC_SR,
                target_sr=SECS_SR).astype(np.float32)
            wav_t_16k = librosa.resample(
                wav_t.astype(np.float64), orig_sr=DAC_SR,
                target_sr=SECS_SR).astype(np.float32)
            if len(wav_s_16k) < 8000 or len(wav_t_16k) < 8000:
                continue

            q_s_list = quantize_full(dac, z_s)
            q0_s = q_s_list[0]

            h_s = extract_w2v2_hidden(wav_s_16k, w2v2_proc, w2v2_model)
            h_t = extract_w2v2_hidden(wav_t_16k, w2v2_proc, w2v2_model)

            feat_s_cache = {}
            feat_t_cache = {}
            for layer in W2V2_LAYERS:
                feat_s_cache[layer] = interp_to_dac(h_s[layer].squeeze(0), T_s)
                feat_t_cache[layer] = interp_to_dac(h_t[layer].squeeze(0), T_t)

            z_dict = {}
            z_dict["dac_dtw"] = align_latents_dac(z_s, z_t)
            for layer in W2V2_LAYERS:
                z_dict[f"w2v2_l{layer}_dtw"] = dtw_align_features(
                    feat_s_cache[layer], feat_t_cache[layer], z_t)
            z_dict["w2v2_l9_nn"] = retrieve_nn_w2v2(
                feat_s_cache[9], feat_t_cache[9], z_t)
            z_dict["random"] = retrieve_random(z_s, z_t)

            e_src = secs_embed(secs_model, wav_s_16k)
            e_tgt = secs_embed(secs_model, wav_t_16k)
            f0_src = extract_f0(wav_s_16k)

            audio_batch = []
            valid_configs = []
            for name in CONFIG_ORDER:
                z_target_like = z_dict[name]
                z_q = src_k1_convert(dac, q0_s, z_target_like)
                audio_44k = decode_dac(dac, z_q)
                if len(audio_44k) < DAC_SR * 0.3:
                    continue
                audio_16k = librosa.resample(
                    audio_44k.astype(np.float64), orig_sr=DAC_SR,
                    target_sr=SECS_SR).astype(np.float32)
                if len(audio_16k) < 1600:
                    continue
                audio_batch.append(audio_16k)
                valid_configs.append(name)

            asr_texts = transcribe_batch(
                whisper_proc, whisper_model, audio_batch)

            for ci, name in enumerate(valid_configs):
                audio_16k = audio_batch[ci]
                e_out = secs_embed(secs_model, audio_16k)
                results[name]["secs"].append(cosine(e_tgt, e_out))
                results[name]["leak"].append(cosine(e_src, e_out))
                results[name]["cer"].append(cer(source_text, asr_texts[ci]))
                f0_out = extract_f0(audio_16k)
                results[name]["f0"].append(f0_correlation(f0_src, f0_out))

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            eta = (len(pairs) - idx - 1) / speed
            r = results
            dac_s = np.mean(r.get("dac_dtw", {}).get("secs", [0])[-20:])
            l6_s = np.mean(r.get("w2v2_l6_dtw", {}).get("secs", [0])[-20:])
            l9_s = np.mean(r.get("w2v2_l9_dtw", {}).get("secs", [0])[-20:])
            l12_s = np.mean(r.get("w2v2_l12_dtw", {}).get("secs", [0])[-20:])
            nn_s = np.mean(r.get("w2v2_l9_nn", {}).get("secs", [0])[-20:])
            print(
                f"  [{idx+1}/{len(pairs)}] "
                f"dac={dac_s:.3f} l6={l6_s:.3f} l9={l9_s:.3f} "
                f"l12={l12_s:.3f} nn={nn_s:.3f} "
                f"| {speed:.1f}p/s ETA {eta:.0f}s", flush=True)

    # =========================================
    # Results
    # =========================================
    print(f"\n{'='*115}")
    print(f"{'config':<16} {'SECS':>22} {'CER':>22} {'F0_corr':>22} {'Leakage':>22}")
    print(f"{'-'*115}")

    summary = {}
    for name in CONFIG_ORDER:
        if name not in results or len(results[name].get("secs", [])) == 0:
            continue
        row = {}
        parts = [f"{name:<16}"]
        for metric in ["secs", "cer", "f0", "leak"]:
            arr = np.array(results[name][metric])
            n = len(arr)
            boot = np.array([
                arr[np.random.choice(n, n, replace=True)].mean()
                for _ in range(500)])
            row[metric] = {
                "mean": float(arr.mean()), "std": float(arr.std()),
                "ci_lo": float(np.percentile(boot, 2.5)),
                "ci_hi": float(np.percentile(boot, 97.5)),
            }
            m = arr.mean()
            lo = np.percentile(boot, 2.5)
            hi = np.percentile(boot, 97.5)
            parts.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
        summary[name] = row
        print(f"{parts[0]} {parts[1]:>22} {parts[2]:>22} {parts[3]:>22} {parts[4]:>22}")

    print(f"\n{'='*70}")
    print("--- Analysis ---\n")

    dac_secs = summary.get("dac_dtw", {}).get("secs", {}).get("mean", 0)
    print(f"DTW oracle (DAC latent):  {dac_secs:.3f}")
    print()
    print("Layer sweep (DTW on Wav2Vec2 features):")
    for l in W2V2_LAYERS:
        n = f"w2v2_l{l}_dtw"
        if n in summary:
            s = summary[n]["secs"]["mean"]
            c = summary[n]["cer"]["mean"]
            gap = dac_secs - s
            print(f"  layer {l:>2}:  SECS={s:.3f}  CER={c:.3f}  gap={gap:+.3f}")

    if "w2v2_l9_dtw" in summary and "w2v2_l9_nn" in summary:
        dtw_s = summary["w2v2_l9_dtw"]["secs"]["mean"]
        nn_s = summary["w2v2_l9_nn"]["secs"]["mean"]
        print(f"\nDTW vs NN (layer 9):")
        print(f"  DTW: {dtw_s:.3f}")
        print(f"  NN:  {nn_s:.3f}")
        print(f"  DTW advantage: {dtw_s - nn_s:+.3f}")

    rnd = summary.get("random", {}).get("secs", {}).get("mean", 0)
    print(f"\nRandom baseline: {rnd:.3f}")

    print(f"\n--- Go/No-Go ---")
    best_w2v2 = max(
        (summary.get(f"w2v2_l{l}_dtw", {}).get("secs", {}).get("mean", 0)
         for l in W2V2_LAYERS), default=0)
    best_w2v2_cer = min(
        (summary.get(f"w2v2_l{l}_dtw", {}).get("cer", {}).get("mean", 1)
         for l in W2V2_LAYERS), default=1)
    best_name = max(
        ((summary.get(f"w2v2_l{l}_dtw", {}).get("secs", {}).get("mean", 0), l)
         for l in W2V2_LAYERS), default=(0, 0))[1]

    print(f"Best Wav2Vec2-DTW (layer {best_name}): SECS={best_w2v2:.3f} CER={best_w2v2_cer:.3f}")
    if best_w2v2 >= 0.55 and best_w2v2_cer <= 0.10:
        print("→ GO: content-aware temporal alignment works")
    elif best_w2v2 >= 0.45:
        print(f"→ PARTIAL: {best_w2v2:.3f} < 0.55 but >= 0.45")
    else:
        print(f"→ FAIL: {best_w2v2:.3f} < 0.45")

    out = {}
    for name, row in summary.items():
        out[name] = {
            m: {"mean": v["mean"], "std": v["std"],
                "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
            for m, v in row.items()}
    Path("results").mkdir(exist_ok=True)
    with open("results/phase2c_content_dtw.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/phase2c_content_dtw.json")


if __name__ == "__main__":
    main()
