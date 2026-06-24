"""
Phase 2b: Latent frame retrieval + src_K1 conversion

Core formula:
  q0_hat = q0_source
  q1..8_hat = RVQ_requantize(z_target_like - q0_source)
  y = DAC_decode(Σ q_hat)

Retrieval methods:
  dtw_oracle:  DTW-aligned target latent (upper bound, = src_K1 from Phase 2a)
  w2v2_nn:     Wav2Vec2 CTC phoneme label + content cosine NN
  pca_nn:      PCA(32)+k-means(50) cluster + content cosine NN
  w2v2_topk:   Wav2Vec2 phoneme top-K latent blend
  pca_topk:    PCA cluster top-K latent blend
  random:      random target frame (lower bound)

Go conditions:
  same-phoneme oracle (w2v2_nn) >= 0.55
  unit retrieval (pca_nn)       >= 0.45
  CER                            <= 0.10
  target_sim - source_sim        > 0
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
N_CLUSTERS = 50
PCA_DIM = 32
TOPK = 3
VCTK_WAV = Path("../data/vctk_200")
VCTK_TXT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")

CONFIG_ORDER = [
    "dtw_oracle",
    "w2v2_nn",
    "pca_nn",
    "w2v2_topk",
    "pca_topk",
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
        "openai/whisper-tiny.en"
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return processor, model


def load_wav2vec2():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h"
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
    return q_list


@torch.no_grad()
def src_k1_convert(dac, q0_source, z_target_like):
    """
    q0_source: [1, 1024, T]
    z_target_like: [1, 1024, T]
    Returns z_q [1, 1024, T]
    """
    quantizers = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    z_q = q0_source.clone()
    residual = z_target_like - q0_source
    for d in range(1, n):
        q_out, _, _, _, _ = quantizers[d](residual)
        z_q = z_q + q_out
        residual = residual - q_out
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


@torch.no_grad()
def extract_w2v2_labels(wav_16k, processor, model):
    inputs = processor(wav_16k, sampling_rate=16000, return_tensors="pt")
    logits = model(inputs.input_values.to(DEVICE)).logits
    labels = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return labels


def map_labels_to_dac_frames(labels, T_dac):
    T_src = len(labels)
    idx = np.minimum(np.arange(T_dac) * T_src // max(T_dac, 1), T_src - 1)
    return torch.from_numpy(labels[idx]).long().to(DEVICE)


def fit_pca_clusters(dac, n_samples=200):
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans

    all_wavs = sorted(VCTK_WAV.rglob("*.wav"))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(all_wavs), min(n_samples, len(all_wavs)), replace=False)

    frames = []
    for i in sample_idx:
        wav = load_wav_44k(all_wavs[i])
        if len(wav) < DAC_SR:
            continue
        z = encode_dac(dac, wav)
        frames.append(z.squeeze(0).T.cpu().numpy())
        if len(frames) >= n_samples:
            break

    frames = np.concatenate(frames, axis=0)
    print(f"  PCA fit on {len(frames)} frames from {len(frames)} utts")

    pca = PCA(n_components=PCA_DIM, random_state=42)
    pca.fit(frames)
    frames_pca = pca.transform(frames)

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=42, batch_size=4096, n_init=3
    )
    kmeans.fit(frames_pca)
    return pca, kmeans


def assign_pca_units(z, pca, kmeans):
    z_f = z.squeeze(0).T.cpu().numpy()
    z_pca = pca.transform(z_f)
    units = kmeans.predict(z_pca)
    return torch.from_numpy(units).long().to(DEVICE)


def retrieve_nn(z_s, z_t, units_s, units_t):
    z_s_f = z_s.squeeze(0).T
    z_t_f = z_t.squeeze(0).T

    sim = F.normalize(z_s_f, dim=-1) @ F.normalize(z_t_f, dim=-1).T
    mask = (units_s.unsqueeze(1) == units_t.unsqueeze(0)).float()
    has_match = mask.sum(dim=-1, keepdim=True) > 0
    eff_sim = torch.where(has_match, sim * mask + (1 - mask) * (-2.0), sim)
    best_j = eff_sim.argmax(dim=-1)
    return z_t_f[best_j].T.unsqueeze(0)


def retrieve_topk_blend(z_s, z_t, units_s, units_t, k=TOPK):
    z_s_f = z_s.squeeze(0).T
    z_t_f = z_t.squeeze(0).T
    T_t = z_t_f.shape[0]

    sim = F.normalize(z_s_f, dim=-1) @ F.normalize(z_t_f, dim=-1).T
    mask = (units_s.unsqueeze(1) == units_t.unsqueeze(0)).float()
    has_match = mask.sum(dim=-1, keepdim=True) > 0
    eff_sim = torch.where(has_match, sim * mask + (1 - mask) * (-2.0), sim)

    k_actual = min(k, T_t)
    topk_vals, topk_idx = eff_sim.topk(k_actual, dim=-1)
    weights = F.softmax(topk_vals * 10, dim=-1)
    z_topk = z_t_f[topk_idx]
    z_blended = (z_topk * weights.unsqueeze(-1)).sum(1)
    return z_blended.T.unsqueeze(0)


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
    print("=== Phase 2b: Latent Frame Retrieval + src_K1 ===\n")

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
    print("Models loaded")

    print("Fitting PCA + k-means...")
    pca, kmeans = fit_pca_clusters(dac)
    print(f"  {N_CLUSTERS} clusters on {PCA_DIM}-dim PCA\n")

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

            wav_s_16k = librosa.resample(
                wav_s.astype(np.float64), orig_sr=DAC_SR, target_sr=SECS_SR
            ).astype(np.float32)
            wav_t_16k = librosa.resample(
                wav_t.astype(np.float64), orig_sr=DAC_SR, target_sr=SECS_SR
            ).astype(np.float32)
            if len(wav_s_16k) < 8000 or len(wav_t_16k) < 8000:
                continue

            q_s_list = quantize_full(dac, z_s)
            q0_s = q_s_list[0]

            units_s_pca = assign_pca_units(z_s, pca, kmeans)
            units_t_pca = assign_pca_units(z_t, pca, kmeans)

            labels_s = extract_w2v2_labels(wav_s_16k, w2v2_proc, w2v2_model)
            labels_t = extract_w2v2_labels(wav_t_16k, w2v2_proc, w2v2_model)
            T_s = z_s.shape[2]
            T_t = z_t.shape[2]
            units_s_w2v2 = map_labels_to_dac_frames(labels_s, T_s)
            units_t_w2v2 = map_labels_to_dac_frames(labels_t, T_t)

            z_t_dtw = align_latents(z_s, z_t)

            e_src = secs_embed(secs_model, wav_s_16k)
            e_tgt = secs_embed(secs_model, wav_t_16k)
            f0_src = extract_f0(wav_s_16k)

            z_dict = {}
            z_dict["dtw_oracle"] = z_t_dtw
            z_dict["w2v2_nn"] = retrieve_nn(z_s, z_t, units_s_w2v2, units_t_w2v2)
            z_dict["pca_nn"] = retrieve_nn(z_s, z_t, units_s_pca, units_t_pca)
            z_dict["w2v2_topk"] = retrieve_topk_blend(z_s, z_t, units_s_w2v2, units_t_w2v2)
            z_dict["pca_topk"] = retrieve_topk_blend(z_s, z_t, units_s_pca, units_t_pca)
            z_dict["random"] = retrieve_random(z_s, z_t)

            audio_batch = []
            valid_configs = []
            for name in CONFIG_ORDER:
                z_target_like = z_dict[name]
                z_q = src_k1_convert(dac, q0_s, z_target_like)
                audio_44k = decode_dac(dac, z_q)
                if len(audio_44k) < DAC_SR * 0.3:
                    continue
                audio_16k = librosa.resample(
                    audio_44k.astype(np.float64),
                    orig_sr=DAC_SR, target_sr=SECS_SR,
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
            dtw_s = np.mean(results.get("dtw_oracle", {}).get("secs", [0])[-20:])
            w2v2_s = np.mean(results.get("w2v2_nn", {}).get("secs", [0])[-20:])
            pca_s = np.mean(results.get("pca_nn", {}).get("secs", [0])[-20:])
            rnd_s = np.mean(results.get("random", {}).get("secs", [0])[-20:])
            print(
                f"  [{idx+1}/{len(pairs)}] "
                f"dtw={dtw_s:.3f} w2v2={w2v2_s:.3f} "
                f"pca={pca_s:.3f} rnd={rnd_s:.3f} "
                f"| {speed:.1f}pair/s ETA {eta:.0f}s",
                flush=True,
            )

    # =========================================
    # Results
    # =========================================
    print(f"\n{'='*110}")
    print(f"{'config':<15} {'SECS':>22} {'CER':>22} {'F0_corr':>22} {'Leakage':>22}")
    print(f"{'-'*115}")

    summary = {}
    for name in CONFIG_ORDER:
        if name not in results or len(results[name].get("secs", [])) == 0:
            continue
        row = {}
        parts = [f"{name:<15}"]
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

    print(f"\n{'='*70}")
    print("--- Go/No-Go Assessment ---\n")

    dtw_secs = summary.get("dtw_oracle", {}).get("secs", {}).get("mean", 0)
    w2v2_secs = summary.get("w2v2_nn", {}).get("secs", {}).get("mean", 0)
    pca_secs = summary.get("pca_nn", {}).get("secs", {}).get("mean", 0)
    w2v2_cer = summary.get("w2v2_nn", {}).get("cer", {}).get("mean", 1)
    pca_cer = summary.get("pca_nn", {}).get("cer", {}).get("mean", 1)
    w2v2_leak = summary.get("w2v2_nn", {}).get("leak", {}).get("mean", 1)
    w2v2_secs_m = summary.get("w2v2_nn", {}).get("secs", {}).get("mean", 0)

    print(f"DTW oracle (upper bound):    {dtw_secs:.3f}")
    print(f"Wav2Vec2 NN (phoneme oracle): {w2v2_secs:.3f}  (target >= 0.55)")
    print(f"PCA NN (unit retrieval):      {pca_secs:.3f}  (target >= 0.45)")
    print(f"Random (lower bound):         {summary.get('random', {}).get('secs', {}).get('mean', 0):.3f}")
    print()
    print(f"CER check:")
    print(f"  w2v2_nn CER:  {w2v2_cer:.3f}  (must be <= 0.10)")
    print(f"  pca_nn CER:   {pca_cer:.3f}  (must be <= 0.10)")
    print()
    print(f"Speaker discrimination (target_sim - source_sim > 0):")
    for name in CONFIG_ORDER:
        if name not in summary:
            continue
        s = summary[name]["secs"]["mean"]
        l = summary[name]["leak"]["mean"]
        diff = s - l
        ok = "OK" if diff > 0 else "FAIL"
        print(f"  {name:<15} secs={s:.3f} leak={l:.3f} diff={diff:+.3f} {ok}")

    print(f"\n--- Verdict ---")
    w2v2_pass = w2v2_secs >= 0.55 and w2v2_cer <= 0.10 and (w2v2_secs - w2v2_leak) > 0
    pca_pass = pca_secs >= 0.45 and pca_cer <= 0.10 and (pca_secs - summary["pca_nn"]["leak"]["mean"]) > 0

    if w2v2_pass:
        print("Phoneme oracle: GO (>= 0.55, CER OK, speaker disc OK)")
    else:
        print(f"Phoneme oracle: {w2v2_secs:.3f} < 0.55 or CER/leak fail")

    if pca_pass:
        print("Unit retrieval: GO (>= 0.45, CER OK, speaker disc OK)")
    else:
        print(f"Unit retrieval: {pca_secs:.3f} < 0.45 or CER/leak fail")

    out = {}
    for name, row in summary.items():
        out[name] = {
            m: {"mean": v["mean"], "std": v["std"],
                "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
            for m, v in row.items()
        }
    Path("results").mkdir(exist_ok=True)
    with open("results/phase2b_latent_retrieval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/phase2b_latent_retrieval.json")


if __name__ == "__main__":
    main()
