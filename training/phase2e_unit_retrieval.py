"""
Phase 2e: Cross-text unit-indexed retrieval (A diagnostic + B1 oracle)

A: Wav2Vec2-NN (top1 / topk3 / topk3+continuity)
   Go: SECS >= 0.45, CER <= 0.12. Stop if fail.

B1: Unit-indexed retrieval
   Units: Wav2Vec2 CTC labels (~30), k-means clusters (50, 100) on layer 6
   Within-unit matching: Wav2Vec2 layer 6 cosine
   Go: SECS >= 0.50, CER <= 0.12

Conversion (all configs):
  q0_hat = q0_source
  q1..8_hat = RVQ_requantize(z_target_like - q0_source)
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
DAC_FPS = 86.13
N_PAIRS = 200
W2V2_LAYER = 6
ENROLL_SEC = 30
VCTK_WAV = Path("../data/vctk_200")
VCTK_TXT = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/txt")
CLUSTER_SIZES = [50, 100]

CONFIG_ORDER = [
    "same_text",
    "cross_nn_top1",
    "cross_nn_topk3",
    "cross_nn_cont",
    "cross_ctc_nn",
    "cross_ctc_topk3",
    "cross_cls50_nn",
    "cross_cls100_nn",
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
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb,
                              "tgt_wav": wb, "text_id": tid})
                if len(pairs) >= n:
                    return pairs
    return pairs


def load_vctk_text(speaker, text_id):
    p = VCTK_TXT / speaker / f"{speaker}_{text_id}.txt"
    return p.read_text().strip() if p.exists() else ""


def normalize_text(t):
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t)


def cer(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 1.0
    m, n = len(r), len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1,
                           dp[i-1][j-1]+(0 if r[i-1]==h[j-1] else 1))
    return dp[m][n] / m


def load_wav_44k(p):
    wav, sr = sf.read(str(p), dtype="float32")
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != DAC_SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=DAC_SR)
    return wav.astype(np.float32)


def extract_f0(w):
    w = w.astype(np.float64)
    if len(w) < 512:
        return np.zeros(1)
    try:
        return pw.wav2world(w, 16000, frame_period=5.0)[0]
    except Exception:
        return np.zeros(1)


def f0_corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    v = (a > 1) & (b > 1)
    if v.sum() < 10:
        return 0.0
    la, lb = np.log(a[v]), np.log(b[v])
    if la.std() < 1e-6 or lb.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(la, lb)[0, 1])


def load_dac():
    from transformers import AutoModel
    d = AutoModel.from_pretrained("descript/dac_44khz").to(DEVICE).eval()
    for p in d.parameters():
        p.requires_grad_(False)
    return d


def load_whisper():
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    pr = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    m = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny.en").to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return pr, m


def load_w2v2():
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    pr = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    m = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-base-960h").to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return pr, m


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
    res = z.clone()
    ql = []
    for d in range(n):
        q, _, _, _, _ = qs[d](res)
        ql.append(q)
        res = res - q
    return ql


@torch.no_grad()
def src_k1(dac, q0, z_tl):
    qs = dac.quantizer.quantizers
    n = dac.quantizer.n_codebooks
    zq = q0.clone()
    res = z_tl - q0
    for d in range(1, n):
        q, _, _, _, _ = qs[d](res)
        zq = zq + q
        res = res - q
    return zq


@torch.no_grad()
def extract_w2v2_all(wav_16k, proc, model, T_dac):
    inp = proc(wav_16k, sampling_rate=16000, return_tensors="pt")
    iv = inp.input_values.to(DEVICE)
    logits = model(iv).logits
    ctc = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    hs = model.wav2vec2(iv, output_hidden_states=True).hidden_states
    h = hs[W2V2_LAYER].squeeze(0)
    h_dac = interp_to_dac(h, T_dac).cpu().numpy()
    Tw = len(ctc)
    idx = np.minimum(np.arange(T_dac) * Tw // max(T_dac, 1), Tw - 1)
    ctc_dac = ctc[idx]
    return h_dac, ctc_dac


def interp_to_dac(feat, T_dac):
    if feat.shape[0] == T_dac:
        return feat
    x = feat.unsqueeze(0).transpose(1, 2)
    x = F.interpolate(x, size=T_dac, mode="linear", align_corners=False)
    return x.transpose(1, 2).squeeze(0)


def dtw_global_map(fs_np, ft_np):
    fs = fs_np / (np.linalg.norm(fs_np, axis=1, keepdims=True) + 1e-8)
    ft = ft_np / (np.linalg.norm(ft_np, axis=1, keepdims=True) + 1e-8)
    _, path = fastdtw(fs, ft, radius=15)
    Ts, Tt = len(fs), len(ft)
    m = np.zeros(Ts, dtype=np.int64)
    for s, t in path:
        if s < Ts:
            m[s] = min(t, Tt - 1)
    for i in range(1, Ts):
        if m[i] == 0:
            m[i] = m[i - 1]
    return m


def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def retrieve_nn_top1(feat_s, e_feat, e_z):
    sim = norm_rows(feat_s) @ norm_rows(e_feat).T
    return e_z[np.argmax(sim, axis=1)]


def retrieve_nn_topk(feat_s, e_feat, e_z, k=3):
    sim = norm_rows(feat_s) @ norm_rows(e_feat).T
    Ts = len(feat_s)
    k = min(k, e_feat.shape[0])
    idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(sim, idx, axis=1)
    w = np.exp(vals * 10)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    zk = e_z[idx]
    return (zk * w[:, :, None]).sum(axis=1)


def retrieve_nn_cont(feat_s, e_feat, e_z, k=3, penalty=0.3):
    sim = norm_rows(feat_s) @ norm_rows(e_feat).T
    Ts, Te = sim.shape
    k = min(k, Te)
    tidx = np.argpartition(sim, -k, axis=1)[:, -k:]
    tval = np.take_along_axis(sim, tidx, axis=1)
    dp = np.zeros((Ts, k))
    bp = np.zeros((Ts, k), dtype=int)
    dp[0] = tval[0]
    for i in range(1, Ts):
        for ki in range(k):
            ji = tidx[i, ki]
            trans = dp[i - 1] - penalty * np.abs(tidx[i - 1] - ji) / Te
            bp[i, ki] = int(np.argmax(trans))
            dp[i, ki] = tval[i, ki] + trans[bp[i, ki]]
    sel = int(np.argmax(dp[-1]))
    mp = np.zeros(Ts, dtype=int)
    for i in range(Ts - 1, -1, -1):
        mp[i] = tidx[i, sel]
        if i > 0:
            sel = bp[i, sel]
    return e_z[mp]


def retrieve_unit_nn(feat_s, u_s, e_feat, e_u, e_z):
    sim = norm_rows(feat_s) @ norm_rows(e_feat).T
    mask = (u_s[:, None] == e_u[None, :]).astype(np.float32)
    has = mask.sum(axis=1) > 0
    ms = np.where(mask > 0, sim, -1e6)
    ub = np.argmax(ms, axis=1)
    gb = np.argmax(sim, axis=1)
    best = np.where(has, ub, gb)
    return e_z[best]


def retrieve_unit_topk(feat_s, u_s, e_feat, e_u, e_z, k=3):
    sim = norm_rows(feat_s) @ norm_rows(e_feat).T
    mask = (u_s[:, None] == e_u[None, :]).astype(np.float32)
    has = mask.sum(axis=1) > 0
    ms = np.where(mask > 0, sim, -1e6)
    Ts = len(feat_s)
    k = min(k, e_feat.shape[0])
    idx = np.argpartition(ms, -k, axis=1)[:, -k:]
    vals = np.take_along_axis(ms, idx, axis=1)
    w = np.exp(np.maximum(vals, -20) * 10)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    zk = e_z[idx]
    blended = (zk * w[:, :, None]).sum(axis=1)
    gb = np.argmax(sim, axis=1)
    return np.where(has[:, None], blended, e_z[gb])


def build_z(z_sel_np):
    return torch.from_numpy(z_sel_np.T).float().unsqueeze(0).to(DEVICE)


_cache = {}


def get_enrollment(dac, proc, model, speaker, excl_text, km_dict):
    key = (speaker, excl_text)
    if key in _cache:
        return _cache[key]
    spk_dir = VCTK_WAV / speaker
    utts = sorted(spk_dir.glob("*.wav"))
    utts = [u for u in utts if u.stem.split("_")[1] != excl_text]
    fl, zl, cl = [], [], []
    cls_l = {n: [] for n in km_dict}
    tot = 0.0
    for u in utts:
        if tot >= ENROLL_SEC:
            break
        wav = load_wav_44k(u)
        if len(wav) < DAC_SR:
            continue
        z = encode_dac(dac, wav)
        Td = z.shape[2]
        w16 = librosa.resample(
            wav.astype(np.float64), orig_sr=DAC_SR,
            target_sr=SECS_SR).astype(np.float32)
        feat, ctc = extract_w2v2_all(w16, proc, model, Td)
        fl.append(feat)
        zl.append(z.squeeze(0).T.cpu().numpy())
        cl.append(ctc)
        for n, km in km_dict.items():
            cls_l[n].append(km.predict(feat))
        tot += Td / DAC_FPS
    e = {"feat": np.concatenate(fl), "z": np.concatenate(zl),
         "ctc": np.concatenate(cl)}
    for n in km_dict:
        e[f"cls{n}"] = np.concatenate(cls_l[n])
    _cache[key] = e
    return e


@torch.no_grad()
def transcribe_batch(pr, m, audios):
    ml = min(max(len(a) for a in audios), 16000 * 30)
    pad = np.zeros((len(audios), ml), dtype=np.float32)
    for i, a in enumerate(audios):
        c = a[:ml]
        pad[i, :len(c)] = c
    inp = pr(pad, sampling_rate=16000, return_tensors="pt")
    ids = m.generate(inp.input_features.to(DEVICE), max_new_tokens=80)
    return [t.strip() for t in pr.batch_decode(ids, skip_special_tokens=True)]


def secs_emb(m, w):
    w = torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return m.encode_batch(w).squeeze(0)


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()


def fit_clusters(proc, model, n_clusters):
    from sklearn.cluster import MiniBatchKMeans
    wavs = sorted(VCTK_WAV.rglob("*.wav"))
    rng = np.random.RandomState(42)
    s = rng.choice(len(wavs), min(200, len(wavs)), replace=False)
    frames = []
    for i in s:
        wav = load_wav_44k(wavs[i])
        if len(wav) < DAC_SR:
            continue
        w16 = librosa.resample(
            wav.astype(np.float64), orig_sr=DAC_SR,
            target_sr=SECS_SR).astype(np.float32)
        z = encode_dac(dac_global, wav)
        Td = z.shape[2]
        h, _ = extract_w2v2_all(w16, proc, model, Td)
        frames.append(h)
        if len(frames) >= 200:
            break
    frames = np.concatenate(frames)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                         batch_size=4096, n_init=3)
    km.fit(frames)
    return km


dac_global = None


def main():
    global dac_global
    print("=== Phase 2e: Unit-Indexed Retrieval (A + B1) ===\n")

    dac = load_dac()
    dac_global = dac
    from speechbrain.inference.speaker import EncoderClassifier
    sm = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)})
    wpr, wm = load_whisper()
    pr, m = load_w2v2()
    print("Models loaded")

    print("Fitting clusters...")
    km_dict = {}
    for nc in CLUSTER_SIZES:
        km_dict[nc] = fit_clusters(pr, m, nc)
        print(f"  {nc} clusters fitted")
    print()

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
            txt = load_vctk_text(p["src"], p["text_id"])
            if not txt:
                continue

            z_s = encode_dac(dac, wav_s)
            Ts = z_s.shape[2]
            q0 = quantize_full(dac, z_s)[0]

            w16s = librosa.resample(wav_s.astype(np.float64),
                                    orig_sr=DAC_SR, target_sr=SECS_SR).astype(np.float32)
            w16t = librosa.resample(wav_t.astype(np.float64),
                                    orig_sr=DAC_SR, target_sr=SECS_SR).astype(np.float32)
            if len(w16s) < 8000 or len(w16t) < 8000:
                continue

            feat_s, ctc_s = extract_w2v2_all(w16s, pr, m, Ts)
            cls_s = {n: km.predict(feat_s) for n, km in km_dict.items()}

            z_t = encode_dac(dac, wav_t)
            Tt = z_t.shape[2]
            feat_t, _ = extract_w2v2_all(w16t, pr, m, Tt)
            z_tf = z_t.squeeze(0).T.cpu().numpy()

            enr = get_enrollment(dac, pr, m, p["tgt"], p["text_id"], km_dict)

            e_src = secs_emb(sm, w16s)
            e_tgt = secs_emb(sm, w16t)
            f0s = extract_f0(w16s)

            zd = {}
            mp = dtw_global_map(feat_s, feat_t)
            zd["same_text"] = build_z(z_tf[mp])
            zd["cross_nn_top1"] = build_z(retrieve_nn_top1(feat_s, enr["feat"], enr["z"]))
            zd["cross_nn_topk3"] = build_z(retrieve_nn_topk(feat_s, enr["feat"], enr["z"]))
            zd["cross_nn_cont"] = build_z(retrieve_nn_cont(feat_s, enr["feat"], enr["z"]))
            zd["cross_ctc_nn"] = build_z(retrieve_unit_nn(feat_s, ctc_s, enr["feat"], enr["ctc"], enr["z"]))
            zd["cross_ctc_topk3"] = build_z(retrieve_unit_topk(feat_s, ctc_s, enr["feat"], enr["ctc"], enr["z"]))
            zd["cross_cls50_nn"] = build_z(retrieve_unit_nn(feat_s, cls_s[50], enr["feat"], enr["cls50"], enr["z"]))
            zd["cross_cls100_nn"] = build_z(retrieve_unit_nn(feat_s, cls_s[100], enr["feat"], enr["cls100"], enr["z"]))
            rnd_idx = np.random.randint(0, len(enr["z"]), Ts)
            zd["random"] = build_z(enr["z"][rnd_idx])

            ab = []
            valid = []
            for name in CONFIG_ORDER:
                zq = src_k1(dac, q0, zd[name])
                a44 = decode_dac(dac, zq)
                if len(a44) < DAC_SR * 0.3:
                    continue
                a16 = librosa.resample(a44.astype(np.float64),
                                       orig_sr=DAC_SR, target_sr=SECS_SR).astype(np.float32)
                if len(a16) < 1600:
                    continue
                ab.append(a16)
                valid.append(name)

            asr = transcribe_batch(wpr, wm, ab)

            for ci, name in enumerate(valid):
                a = ab[ci]
                eo = secs_emb(sm, a)
                results[name]["secs"].append(cos(e_tgt, eo))
                results[name]["leak"].append(cos(e_src, eo))
                results[name]["cer"].append(cer(txt, asr[ci]))
                results[name]["f0"].append(f0_corr(f0s, extract_f0(a)))

        except Exception as e:
            print(f"  SKIP pair {idx}: {e}")
            continue

        if (idx + 1) % 20 == 0:
            el = time.time() - t0
            sp = (idx + 1) / el
            eta = (len(pairs) - idx - 1) / sp
            r = results
            st = np.mean(r.get("same_text", {}).get("secs", [0])[-20:])
            nn1 = np.mean(r.get("cross_nn_top1", {}).get("secs", [0])[-20:])
            cnn = np.mean(r.get("cross_ctc_nn", {}).get("secs", [0])[-20:])
            c50 = np.mean(r.get("cross_cls50_nn", {}).get("secs", [0])[-20:])
            print(f"  [{idx+1}/{len(pairs)}] same={st:.3f} nn={nn1:.3f} "
                  f"ctc_nn={cnn:.3f} cls50={c50:.3f} "
                  f"| {sp:.1f}p/s ETA {eta:.0f}s", flush=True)

    # Results
    print(f"\n{'='*115}")
    print(f"{'config':<20} {'SECS':>22} {'CER':>22} {'F0_corr':>22} {'Leakage':>22}")
    print(f"{'-'*115}")

    summary = {}
    for name in CONFIG_ORDER:
        if name not in results or not results[name].get("secs"):
            continue
        row = {}
        parts = [f"{name:<20}"]
        for met in ["secs", "cer", "f0", "leak"]:
            arr = np.array(results[name][met])
            n = len(arr)
            boot = np.array([arr[np.random.choice(n, n, replace=True)].mean()
                             for _ in range(500)])
            row[met] = {"mean": float(arr.mean()), "std": float(arr.std()),
                        "ci_lo": float(np.percentile(boot, 2.5)),
                        "ci_hi": float(np.percentile(boot, 97.5))}
            parts.append(f"{arr.mean():.3f} [{np.percentile(boot,2.5):.3f},"
                         f"{np.percentile(boot,97.5):.3f}]")
        summary[name] = row
        print(f"{parts[0]} {parts[1]:>22} {parts[2]:>22} {parts[3]:>22} {parts[4]:>22}")

    print(f"\n{'='*70}")
    print("--- Go/No-Go ---\n")
    st = summary.get("same_text", {}).get("secs", {}).get("mean", 0)
    print(f"Same-text oracle: {st:.3f}\n")

    print(f"{'config':<20} {'SECS':>8} {'CER':>8} {'diff':>8} {'Go?':>8}")
    print(f"{'-'*55}")
    for name in CONFIG_ORDER:
        if name not in summary or name in ("same_text",):
            continue
        s = summary[name]
        sm_ = s["secs"]["mean"]
        cm = s["cer"]["mean"]
        lm = s["leak"]["mean"]
        diff = sm_ - lm
        go = ""
        if name == "random":
            pass
        elif sm_ >= 0.50 and cm <= 0.12 and diff > 0:
            go = "GO"
        elif sm_ >= 0.45 and cm <= 0.12:
            go = "marginal"
        print(f"{name:<20} {sm_:>8.3f} {cm:>8.3f} {diff:>+8.3f} {go:>8}")

    best_cross = max(
        (summary.get(n, {}).get("secs", {}).get("mean", 0)
         for n in CONFIG_ORDER if n not in ("same_text", "random")), default=0)
    print(f"\nBest cross-text: {best_cross:.3f}")
    if best_cross >= 0.50:
        print("→ GO: unit-indexed cross-text works")
    elif best_cross >= 0.45:
        print("→ PARTIAL: phoneme-balanced enrollment may help")
    else:
        print("→ CHALLENGING: consider same-text mode (C)")

    out = {n: {m: {"mean": v["mean"], "std": v["std"],
                   "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]}
               for m, v in r.items()} for n, r in summary.items()}
    Path("results").mkdir(exist_ok=True)
    with open("results/phase2e_unit_retrieval.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: results/phase2e_unit_retrieval.json")


if __name__ == "__main__":
    main()
