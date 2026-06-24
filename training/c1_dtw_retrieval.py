"""
C-1: Per-utterance DTW retrieval

フレーム独立kNN (0.43) の限界 → DTW系列整列で正確な対応を抽出

各bank発話とsource個別にDTW → path取得
→ 各source frameに最も近いbank frameを対応付け
→ DTWOracle(0.71)に近いpath品質を目指す

バリエーション:
  1. per-utt DTW + best-utt selection
  2. per-utt DTW + cost-weighted blend
  3. per-utt DTW + ctx8 key での距離計算
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn.functional as F
import librosa
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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


def compute_speaker_stats(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    all_mc = np.concatenate([np.load(f)["mc"] for f in files], axis=0)
    return all_mc.mean(axis=0)


def build_bank_utterances(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    files = [f for f in files if exclude_text not in f.name][:n_utts]
    utts = []
    for f in files:
        d = np.load(f)
        utts.append({"mc": d["mc"].astype(np.float32), "f0": d["f0"].astype(np.float32),
                      "name": f.stem})
    return utts


def per_utt_dtw_retrieve(mc_src, src_key, bank_utts, bank_keys, weights_fr,
                          selection="best", blend_k=3):
    """
    各bank発話とDTW整列 → best frame選択

    selection:
      "best": 最小コストの発話から取得
      "blend": 発話間でコスト加重ブレンド
    """
    T = len(mc_src)
    mc_dim = mc_src.shape[1]
    result = np.zeros((T, mc_dim), dtype=np.float32)

    candidates = []  # (T, mc_dim) per utterance
    costs = []       # (T,) per utterance

    for utt, bkey in zip(bank_utts, bank_keys):
        bank_mc = utt["mc"]
        if len(bank_mc) < 10:
            continue

        dist, path = fastdtw(src_key, bkey, radius=10, dist=euclidean)

        src_to_bank = np.full(T, -1, dtype=int)
        for s, b in path:
            if s < T:
                src_to_bank[s] = b
        for i in range(1, T):
            if src_to_bank[i] == -1:
                src_to_bank[i] = src_to_bank[i-1]

        cand = bank_mc[src_to_bank.clip(0, len(bank_mc)-1)]

        frame_dist = np.sqrt(((mc_src - cand)**2).sum(axis=1))

        candidates.append(cand)
        costs.append(frame_dist)

    if not candidates:
        return result

    candidates = np.stack(candidates)  # (n_utt, T, mc_dim)
    costs = np.stack(costs)            # (n_utt, T)

    if selection == "best":
        best_idx = np.argmin(costs, axis=0)  # (T,)
        result = candidates[best_idx, np.arange(T)]
    elif selection == "blend":
        k = min(blend_k, len(candidates))
        sorted_idx = np.argsort(costs, axis=0)  # (n_utt, T)
        topk_idx = sorted_idx[:k]               # (k, T)
        topk_costs = costs[topk_idx, np.arange(T)]  # (k, T)
        topk_cands = candidates[topk_idx, np.arange(T)]  # (k, T, mc_dim)

        weights = np.exp(-topk_costs / (topk_costs.mean() + 1e-10))
        weights = weights / (weights.sum(axis=0, keepdims=True) + 1e-10)
        result = np.einsum('kt,ktd->td', weights, topk_cands)

    return result


def main():
    DEVICE = torch.device("cuda")
    print("=== C-1: Per-utterance DTW Retrieval ===\n")

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
        ("dtw_mean_best",    "mean",     "best",  5),
        ("dtw_ifr_best",     "ifr",      "best",  5),
        ("dtw_ifr_blend3",   "ifr",      "blend", 5),
    ]

    results = defaultdict(list)
    speaker_means = {}
    t0 = time.time()

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
                speaker_means[spk] = compute_speaker_stats(spk)
        src_mean = speaker_means[p["src"]]
        tgt_mean = speaker_means[p["tgt"]]

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            for name, norm, sel, n_bank in configs:
                bank_utts = build_bank_utterances(p["tgt"], p["text"], n_utts=n_bank)

                if norm == "mean":
                    src_key = (mc_s - src_mean).astype(np.float32)
                    bank_keys = [(utt["mc"] - tgt_mean).astype(np.float32) for utt in bank_utts]
                elif norm == "ifr":
                    src_key = ((mc_s - src_mean) * inv_fratio[None, :]).astype(np.float32)
                    bank_keys = [((utt["mc"] - tgt_mean) * inv_fratio[None, :]).astype(np.float32)
                                 for utt in bank_utts]

                mc_pred = per_utt_dtw_retrieve(mc_s, src_key, bank_utts, bank_keys,
                                                inv_fratio, selection=sel, blend_k=3)
                mc_pred = mc_pred[:T].astype(np.float32)
                wav_out = synth(f0_shifted[:T], mc_pred, ap_s[:T])
                sim = F.cosine_similarity(e_tgt, emb(wav_out), dim=-1).item()
                results[name].append(sim)

        elapsed = time.time() - t0
        eta = elapsed / (idx+1) * (len(pairs) - idx - 1)
        if (idx+1) % 5 == 0:
            print(f"  [{idx+1}/{len(pairs)}] " + " | ".join(
                f"{n}={np.mean(results[n][-5:]):.3f}" for n, *_ in configs[:4])
                + f" ETA {eta:.0f}s", flush=True)

    print(f"\n{'='*60}")
    print(f"{'config':<22} {'mean':>8} {'std':>8}")
    print(f"{'-'*42}")
    for name, *_ in configs:
        arr = np.array(results[name])
        print(f"{name:<22} {arr.mean():>8.4f} {arr.std():>8.4f}")

    print(f"\n参考: frame kNN (ctx8_ifr_k3) = 0.427")
    print(f"参考: DTW Oracle = 0.706")

    best_name = max(results.keys(), key=lambda k: np.mean(results[k]))
    best_score = np.mean(results[best_name])
    print(f"\n最高: {best_name} = {best_score:.4f}")
    if best_score >= 0.48:
        print("→ C Go条件 (>= 0.48) クリア!")
    elif best_score >= 0.45:
        print("→ 0.48にほぼ到達")
    else:
        print("→ 0.48に届かず")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/c1_dtw_retrieval.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
