"""
E2+E3: F0 register key + mcep smoothing

E1でViterbi平滑化が逆効果と判明 → key識別力がボトルネック

E2: F0 registerをkeyに追加
  - hard partition: 同じregister bin内のみ検索
  - soft: 隣接binも候補

E3: E2 + 出力mcepの軽いmedian smoothing
"""
import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000; FRAME_PERIOD = 5.0; FFTL = 2048; ALPHA = 0.410
MC_ORDER = 24; MC_DIM = 25
VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")
N_PAIRS = 20


def analyze_wav(wav_path):
    wav, sr = sf.read(str(wav_path), dtype="float32")
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
    wav = wav.astype(np.float64)
    f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
    f0 = world.stonemask(wav, f0, t, SR)
    sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
    ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)
    mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
    return {"f0": f0.astype(np.float32), "mc": mc.astype(np.float32), "ap": ap}


def synth(f0, mc, ap):
    mc64 = np.ascontiguousarray(mc, dtype=np.float64)
    sp = sptk.mc2sp(mc64, ALPHA, FFTL)
    ap64 = np.ascontiguousarray(ap, dtype=np.float64)
    f064 = np.ascontiguousarray(f0, dtype=np.float64)
    return world.synthesize(f064, sp, ap64, SR, frame_period=FRAME_PERIOD).astype(np.float32)


def shift_f0(f0, tgt_mean):
    voiced = f0[f0 > 0]
    if len(voiced) == 0: return f0.astype(np.float64)
    src_mean = float(np.exp(np.mean(np.log(voiced))))
    return np.where(f0 > 0, f0 * tgt_mean / src_mean, 0).astype(np.float64)


def find_pairs(n=20):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    used = set()
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb or sa in used or sb in used: continue
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                used.add(sa); used.add(sb)
                if len(pairs) >= n: return pairs
    return pairs


def compute_speaker_mean(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    return np.concatenate([np.load(f)["mc"] for f in files], axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    bank_mc = np.concatenate([np.load(f)["mc"] for f in files], axis=0).astype(np.float32)
    bank_f0 = np.concatenate([np.load(f)["f0"] for f in files], axis=0).astype(np.float32)
    return bank_mc, bank_f0


def f0_to_register(f0, n_bins=4):
    T = len(f0)
    reg = np.zeros(T, dtype=np.int32)
    voiced = f0 > 0
    if voiced.any():
        log_f0_v = np.log(f0[voiced])
        edges = np.linspace(log_f0_v.min(), log_f0_v.max() + 0.01, n_bins + 1)[1:-1]
        reg[voiced] = np.digitize(log_f0_v, edges) + 1
    return reg


def f0_to_register_global(f0, n_bins=4):
    T = len(f0)
    reg = np.zeros(T, dtype=np.int32)
    voiced = f0 > 0
    if voiced.any():
        edges = np.linspace(np.log(80), np.log(400), n_bins + 1)[1:-1]
        reg[voiced] = np.digitize(np.log(f0[voiced]), edges) + 1
    return reg


def build_context_key(mc, spk_mean, ctx=8, weights=None):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


def retrieve(query_keys, bank_keys, bank_values, k=3, temp=1.0):
    tree = cKDTree(bank_keys)
    dist, idx = tree.query(query_keys, k=k)
    if k == 1:
        dist = dist[:, None]; idx = idx[:, None]
    weights = np.exp(-dist / (temp + 1e-10))
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-10)
    return np.einsum('nk,nkd->nd', weights, bank_values[idx])


def retrieve_register(query_keys, query_reg, bank_keys, bank_reg, bank_values,
                      k=3, temp=1.0, allow_adjacent=True):
    T = len(query_keys)
    mc_dim = bank_values.shape[1]
    result = np.zeros((T, mc_dim), dtype=np.float32)

    unique_regs = sorted(set(bank_reg))
    reg_trees = {}
    for r in unique_regs:
        mask = bank_reg == r
        n = mask.sum()
        if n == 0: continue
        idx_map = np.where(mask)[0]
        reg_trees[r] = (cKDTree(bank_keys[mask]), idx_map, n)

    fallback_tree = cKDTree(bank_keys) if len(bank_keys) > 0 else None
    fallback_idx = np.arange(len(bank_keys))

    for t in range(T):
        qr = query_reg[t]

        candidates = []
        if qr in reg_trees:
            candidates.append(qr)
        if allow_adjacent:
            for adj in [qr - 1, qr + 1]:
                if adj in reg_trees:
                    candidates.append(adj)

        if not candidates:
            if fallback_tree is not None:
                d, ix = fallback_tree.query(query_keys[t], k=min(k, len(fallback_idx)))
                d = np.atleast_1d(d); ix = np.atleast_1d(ix)
                w = np.exp(-d / (temp + 1e-10)); w = w / (w.sum() + 1e-10)
                result[t] = (bank_values[fallback_idx[ix]] * w[:, None]).sum(axis=0)
            continue

        all_d = []; all_v = []
        for r in candidates:
            tree, idx_map, n = reg_trees[r]
            kk = min(k, n)
            d, ix = tree.query(query_keys[t], k=kk)
            d = np.atleast_1d(d); ix = np.atleast_1d(ix)
            all_d.append(d)
            all_v.append(bank_values[idx_map[ix]])

        all_d = np.concatenate(all_d)
        all_v = np.concatenate(all_v, axis=0)

        kk = min(k, len(all_d))
        top = np.argsort(all_d)[:kk]
        d_top = all_d[top]
        v_top = all_v[top]
        w = np.exp(-d_top / (temp + 1e-10)); w = w / (w.sum() + 1e-10)
        result[t] = (v_top * w[:, None]).sum(axis=0)

    return result


def main():
    DEVICE = torch.device("cuda")
    print("=== E2+E3: F0 Register Key + Smoothing ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    fratio = np.load("data/mc_fratio_weights.npy")
    inv_fratio = (1.0 / (fratio + 1e-6)).astype(np.float32)
    inv_fratio = inv_fratio / inv_fratio.mean()

    pairs = find_pairs(N_PAIRS)
    print(f"Pairs: {len(pairs)}\n")

    configs = [
        ("baseline",            False, False, False, 0),
        ("median3",             False, False, False, 3),
        ("median5",             False, False, False, 5),
        ("reg_hard",            True,  False, False, 0),
        ("reg_adj",             True,  True,  False, 0),
        ("reg_hard_med3",       True,  False, False, 3),
        ("reg_adj_med3",        True,  True,  False, 3),
    ]

    results = defaultdict(list)
    speaker_means = {}

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])
        mc_s = feat_s["mc"]; f0_s = feat_s["f0"]; ap_s = feat_s["ap"]
        f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        for spk in [p["src"], p["tgt"]]:
            if spk not in speaker_means:
                speaker_means[spk] = compute_speaker_mean(spk)
        src_mean = speaker_means[p["src"]]
        tgt_mean = speaker_means[p["tgt"]]

        bank_mc, bank_f0 = build_bank(p["tgt"], p["text"], n_utts=10)

        src_keys = build_context_key(mc_s, src_mean, 8, inv_fratio)
        bank_keys = build_context_key(bank_mc, tgt_mean, 8, inv_fratio)

        src_reg = f0_to_register_global(f0_s, n_bins=4)
        bank_reg = f0_to_register_global(bank_f0, n_bins=4)

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for name, use_reg, adj, _, med_size in configs:
                if use_reg:
                    mc_pred = retrieve_register(
                        src_keys, src_reg, bank_keys, bank_reg, bank_mc,
                        k=3, temp=1.0, allow_adjacent=adj)
                else:
                    mc_pred = retrieve(src_keys, bank_keys, bank_mc, k=3, temp=1.0)

                if med_size > 0:
                    mc_pred = median_filter(mc_pred, size=(med_size, 1))

                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])
                sim = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
                results[name].append(sim)

        if (idx+1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] " + " | ".join(
                f"{n}={np.mean(results[n][-5:]):.3f}" for n, *_ in configs[:4]), flush=True)

    print(f"\n{'='*55}")
    print(f"{'config':<20} {'mean':>8} {'std':>8} {'vs base':>8}")
    print(f"{'-'*50}")
    base = np.mean(results["baseline"])
    for name, *_ in configs:
        arr = np.array(results[name])
        print(f"{name:<20} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.mean()-base:>+8.4f}")

    best_name = max(results.keys(), key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_name])
    print(f"\n最高: {best_name} = {best_score:.4f}")
    if best_score >= 0.48:
        print("→ E3 Go条件 (>= 0.48) クリア!")
    elif best_score >= 0.45:
        print("→ 0.48にほぼ到達")
    else:
        print("→ 0.48に届かず")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/e2_e3_register.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
