"""
Phase 2d: Cross-text subsequence DTW

Core formula (unchanged from Phase 2c):
  q0_hat = q0_source
  q1..8_hat = RVQ_requantize(z_target_like - q0_source)

Alignment: subsequence DTW on Wav2Vec2 layer 6 features.
Source broken into chunks; each chunk matched against cross-text enrollment.

Ablations:
  - chunk length: 0.5s / 1.0s / 1.5s / full
  - enrollment length: 10s / 30s / 60s
  - latent smoothing post-DTW

Go: cross-text >= 0.50, CER <= 0.10, target_sim - source_sim > 0
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
DAC_FPS = 86.13
W2V2_LAYER = 6
SMOOTH_WINDOW = 5
VCTK_WAV = Path("../data/vctk_200")
VCTK_TXT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")

CONFIGS = [
    ("same_text", None, None, False),
    ("cross_c05_e30", 43, 2584, False),
    ("cross_c10_e30", 86, 2584, False),
    ("cross_c15_e30", 129, 2584, False),
    ("cross_c10_e10", 86, 861, False),
    ("cross_c10_e60", 86, 5168, False),
    ("cross_full_e30", None, 2584, False),
    ("cross_c10_e30_sm", 86, 2584, True),
    ("random", None, None, False),
]
CONFIG_NAMES = [c[0] for c in CONFIGS]


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
    p = VCTK_TXT / speaker / f"{speaker}_{text_id}.txt"
    if p.exists():
        return p.read_text().strip()
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
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n] / m


def load_wav_44k(p):
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def extract_f0(wav_16k):
    w = wav_16k.astype(np.float64)
    if len(w) < 512:
        return np.zeros(1)
    try:
        f0, _, _ = pw.wav2world(w, 16000, frame_period=5.0)
        return f0
    except Exception:
        return np.zeros(1)


def f0_corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    v = (a > 1.0) & (b > 1.0)
    if v.sum() < 10:
        return 0.0
    la, lb = np.log(a[v]), np.log(b[v])
    if la.std() < 1e-6 or lb.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(la, lb)[0, 1])


def load_dac():
    from transformers import AutoModel
    dac = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    return dac


def load_whisper():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en").to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return proc, m


def load_wav2vec2():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    proc = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    m = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h").to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return proc, m


@torch.no_grad()
def encode_dac(dac, wav):
    x = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    return dac.encoder(x)


@torch.no_grad()
def decode_dac(dac, z):
    return dac.decoder(z).squeeze().cpu().numpy()


@torch.no_grad()
def quantize_full(dac, z):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    residual = z.clone()
    q_list = []
    for d in range(n):
        q_out, _, _, _, _ = qs[d](residual)
        q_list.append(q_out)
        residual = residual - q_out
    return q_list


@torch.no_grad()
def src_k1_convert(dac, q0_s, z_target_like):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    z_q = q0_s.clone()
    residual = z_target_like - q0_s
    for d in range(1, n):
        q_out, _, _, _, _ = qs[d](residual)
        z_q = z_q + q_out
        residual = residual - q_out
    return z_q


@torch.no_grad()
def extract_w2v2_layer(wav_16k, proc, model, layer):
    inputs = proc(wav_16k, sampling_rate=16000, return_tensors="pt")
    out = model.wav2vec2(
        inputs.input_values.to(DEVICE), output_hidden_states=True)
    return out.hidden_states[layer].squeeze(0)


def interp_to_dac(feat, T_dac):
    if feat.shape[0] == T_dac:
        return feat
    x = feat.unsqueeze(0).transpose(1, 2)
    x = F.interpolate(x, size=T_dac, mode="linear", align_corners=False)
    return x.transpose(1, 2).squeeze(0)


def dtw_global_mapping(feat_s_np, feat_t_np):
    fs = feat_s_np / (np.linalg.norm(feat_s_np, axis=1, keepdims=True) + 1e-8)
    ft = feat_t_np / (np.linalg.norm(feat_t_np, axis=1, keepdims=True) + 1e-8)
    _, path = fastdtw(fs, ft, radius=15)
    T_s, T_t = len(fs), len(ft)
    s2t = np.zeros(T_s, dtype=np.int64)
    for s, t in path:
        if s < T_s:
            s2t[s] = min(t, T_t - 1)
    for i in range(1, T_s):
        if s2t[i] == 0:
            s2t[i] = s2t[i - 1]
    return s2t


def subseq_dtw(query, reference):
    T_q, T_r = len(query), len(reference)
    q_n = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    r_n = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)
    C = 1.0 - q_n @ r_n.T

    INF = 1e18
    D = np.full((T_q, T_r), INF)
    D[0] = C[0]

    for i in range(1, T_q):
        diag = np.full(T_r, INF)
        diag[1:] = D[i - 1, :-1]
        vert = D[i - 1]
        row = C[i] + np.minimum(diag, vert)
        for j in range(1, T_r):
            v = C[i, j] + row[j - 1]
            if v < row[j]:
                row[j] = v
        D[i] = row

    best_end = int(np.argmin(D[T_q - 1]))
    path = []
    i, j = T_q - 1, best_end
    path.append((i, j))
    while i > 0 or j > 0:
        dv = D[i - 1, j - 1] if i > 0 and j > 0 else INF
        vv = D[i - 1, j] if i > 0 else INF
        hv = D[i, j - 1] if j > 0 else INF
        m = min(dv, vv, hv)
        if m == dv:
            i, j = i - 1, j - 1
        elif m == vv:
            i = i - 1
        else:
            j = j - 1
        path.append((i, j))
    path.reverse()

    mapping = np.zeros(T_q, dtype=np.int64)
    for pi, pj in path:
        if 0 <= pi < T_q:
            mapping[pi] = pj
    return mapping


def build_z_from_mapping(mapping, z_frames_np):
    z_sel = z_frames_np[mapping]
    return torch.from_numpy(z_sel.T).float().unsqueeze(0).to(DEVICE)


def smooth_z(z, window=SMOOTH_WINDOW):
    if window <= 1:
        return z
    C = z.shape[1]
    T = z.shape[2]
    kernel = torch.ones(C, 1, window, device=z.device) / window
    pad = (window - 1) // 2
    return F.conv1d(z, kernel, padding=pad, groups=C)[:, :, :T]


def chunked_subseq_align(feat_s_np, enroll_feat_np, chunk_frames):
    T_s = len(feat_s_np)
    mapping = np.zeros(T_s, dtype=np.int64)
    for start in range(0, T_s, chunk_frames):
        end = min(start + chunk_frames, T_s)
        chunk = feat_s_np[start:end]
        m = subseq_dtw(chunk, enroll_feat_np)
        mapping[start:end] = m
    return mapping


_enroll_cache = {}


def get_enrollment(dac, w2v2_proc, w2v2_model, speaker, exclude_text):
    key = (speaker, exclude_text)
    if key in _enroll_cache:
        return _enroll_cache[key]

    speaker_dir = VCTK_WAV / speaker
    utts = sorted(speaker_dir.glob("*.wav"))
    utts = [u for u in utts if u.stem.split("_")[1] != exclude_text]

    feat_list, z_list = [], []
    total_sec = 0.0
    for u in utts:
        if total_sec >= 70:
            break
        wav = load_wav_44k(u)
        if len(wav) < DAC_SR:
            continue
        z = encode_dac(dac, wav)
        T_dac = z.shape[2]
        wav_16k = librosa.resample(
            wav.astype(np.float64), orig_sr=DAC_SR,
            target_sr=SECS_SR).astype(np.float32)
        h = extract_w2v2_layer(wav_16k, w2v2_proc, w2v2_model, W2V2_LAYER)
        feat = interp_to_dac(h, T_dac).cpu().numpy()
        feat_list.append(feat)
        z_list.append(z.squeeze(0).T.cpu().numpy())
        total_sec += T_dac / DAC_FPS

    feat_all = np.concatenate(feat_list, axis=0) if feat_list else np.zeros((1, 768))
    z_all = np.concatenate(z_list, axis=0) if z_list else np.zeros((1, 1024))
    _enroll_cache[key] = (feat_all, z_all)
    return feat_all, z_all


@torch.no_grad()
def transcribe_batch(proc, model, audio_list):
    max_len = max(len(a) for a in audio_list)
    max_len = min(max_len, 16000 * 30)
    padded = np.zeros((len(audio_list), max_len), dtype=np.float32)
    for i, a in enumerate(audio_list):
        c = a[:max_len]
        padded[i, :len(c)] = c
    inputs = proc(padded, sampling_rate=16000, return_tensors="pt")
    feats = inputs.input_features.to(DEVICE)
    ids = model.generate(feats, max_new_tokens=80)
    texts = proc.batch_decode(ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def secs_embed(model, wav_16k):
    w = torch.from_numpy(wav_16k.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model.encode_batch(w).squeeze(0)


def cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()


def main():
    print("=== Phase 2d: Cross-text Subsequence DTW ===\n")

    dac = load_dac()
    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )
    w_proc, w_model = load_whisper()
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
            T_s = z_s.shape[2]
            q0_s = quantize_full(dac, z_s)[0]

            wav_s_16k = librosa.resample(
                wav_s.astype(np.float64), orig_sr=DAC_SR,
                target_sr=SECS_SR).astype(np.float32)
            wav_t_16k = librosa.resample(
                wav_t.astype(np.float64), orig_sr=DAC_SR,
                target_sr=SECS_SR).astype(np.float32)
            if len(wav_s_16k) < 8000 or len(wav_t_16k) < 8000:
                continue

            feat_s = interp_to_dac(
                extract_w2v2_layer(wav_s_16k, w2v2_proc, w2v2_model, W2V2_LAYER),
                T_s,
            ).cpu().numpy()

            z_t = encode_dac(dac, wav_t)
            T_t = z_t.shape[2]
            feat_t = interp_to_dac(
                extract_w2v2_layer(wav_t_16k, w2v2_proc, w2v2_model, W2V2_LAYER),
                T_t,
            ).cpu().numpy()
            z_t_frames = z_t.squeeze(0).T.cpu().numpy()

            enroll_feat, enroll_z = get_enrollment(
                dac, w2v2_proc, w2v2_model, p["tgt"], p["text_id"])

            e_src = secs_embed(secs_model, wav_s_16k)
            e_tgt = secs_embed(secs_model, wav_t_16k)
            f0_src = extract_f0(wav_s_16k)

            z_dict = {}

            for name, chunk_f, enroll_f, do_smooth in CONFIGS:
                if name == "random":
                    idx_rand = np.random.randint(0, len(enroll_z), T_s)
                    z_dict[name] = build_z_from_mapping(idx_rand, enroll_z)
                    continue

                if name == "same_text":
                    mp = dtw_global_mapping(feat_s, feat_t)
                    z_dict[name] = build_z_from_mapping(mp, z_t_frames)
                    continue

                ef = min(enroll_f, len(enroll_feat))
                enroll_feat_trunc = enroll_feat[:ef]
                enroll_z_trunc = enroll_z[:ef]

                if chunk_f is None:
                    mp = subseq_dtw(feat_s, enroll_feat_trunc)
                else:
                    mp = chunked_subseq_align(
                        feat_s, enroll_feat_trunc, chunk_f)

                z_target = build_z_from_mapping(mp, enroll_z_trunc)
                if do_smooth:
                    z_target = smooth_z(z_target)
                z_dict[name] = z_target

            audio_batch = []
            valid = []
            for name, _, _, _ in CONFIGS:
                z_q = src_k1_convert(dac, q0_s, z_dict[name])
                audio_44k = decode_dac(dac, z_q)
                if len(audio_44k) < DAC_SR * 0.3:
                    continue
                audio_16k = librosa.resample(
                    audio_44k.astype(np.float64), orig_sr=DAC_SR,
                    target_sr=SECS_SR).astype(np.float32)
                if len(audio_16k) < 1600:
                    continue
                audio_batch.append(audio_16k)
                valid.append(name)

            asr_texts = transcribe_batch(w_proc, w_model, audio_batch)

            for ci, name in enumerate(valid):
                a16 = audio_batch[ci]
                e_out = secs_embed(secs_model, a16)
                results[name]["secs"].append(cosine(e_tgt, e_out))
                results[name]["leak"].append(cosine(e_src, e_out))
                results[name]["cer"].append(cer(source_text, asr_texts[ci]))
                results[name]["f0"].append(f0_corr(f0_src, extract_f0(a16)))

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            sp = (idx + 1) / el
            eta = (len(pairs) - idx - 1) / sp
            r = results
            st = np.mean(r.get("same_text", {}).get("secs", [0])[-20:])
            c10 = np.mean(r.get("cross_c10_e30", {}).get("secs", [0])[-20:])
            rnd = np.mean(r.get("random", {}).get("secs", [0])[-20:])
            print(
                f"  [{idx+1}/{len(pairs)}] "
                f"same={st:.3f} cross_c10={c10:.3f} rnd={rnd:.3f} "
                f"| {sp:.1f}p/s ETA {eta:.0f}s", flush=True)

    # =========================================
    print(f"\n{'='*115}")
    print(f"{'config':<22} {'SECS':>22} {'CER':>22} {'F0_corr':>22} {'Leakage':>22}")
    print(f"{'-'*115}")

    summary = {}
    for name in CONFIG_NAMES:
        if name not in results or not results[name].get("secs"):
            continue
        row = {}
        parts = [f"{name:<22}"]
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
            parts.append(
                f"{m:.3f} [{np.percentile(boot, 2.5):.3f},{np.percentile(boot, 97.5):.3f}]")
        summary[name] = row
        print(f"{parts[0]} {parts[1]:>22} {parts[2]:>22} {parts[3]:>22} {parts[4]:>22}")

    print(f"\n{'='*70}")
    print("--- Analysis ---\n")

    st = summary.get("same_text", {}).get("secs", {}).get("mean", 0)
    print(f"Same-text oracle:  {st:.3f}")
    print()

    print(f"{'config':<22} {'SECS':>8} {'CER':>8} {'diff':>8} {'Go?':>6}")
    print(f"{'-'*55}")
    for name in CONFIG_NAMES:
        if name not in summary or name in ("same_text", "random"):
            continue
        s = summary[name]
        secs_m = s["secs"]["mean"]
        cer_m = s["cer"]["mean"]
        leak_m = s["leak"]["mean"]
        diff = secs_m - leak_m
        go = "GO" if secs_m >= 0.50 and cer_m <= 0.10 and diff > 0 else ""
        print(f"{name:<22} {secs_m:>8.3f} {cer_m:>8.3f} {diff:>+8.3f} {go:>6}")

    rnd = summary.get("random", {}).get("secs", {}).get("mean", 0)
    print(f"\n{'random':<22} {rnd:>8.3f}")

    best_cross = max(
        (summary.get(n, {}).get("secs", {}).get("mean", 0)
         for n, _, _, _ in CONFIGS
         if n not in ("same_text", "random")), default=0)
    print(f"\nBest cross-text: {best_cross:.3f}")
    if best_cross >= 0.50:
        print("→ GO: cross-text CONCEPT v2 established")
    elif best_cross >= 0.45:
        print(f"→ PARTIAL: need phoneme-balanced enrollment")
    else:
        print(f"→ CHALLENGING: cross-text alignment needs work")

    out = {}
    for name, row in summary.items():
        out[name] = {m: {"mean": v["mean"], "std": v["std"],
                         "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
                     for m, v in row.items()}
    Path("results").mkdir(exist_ok=True)
    with open("results/phase2d_cross_text.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/phase2d_cross_text.json")


if __name__ == "__main__":
    main()
