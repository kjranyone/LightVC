"""
大規模評価: 200ペア × bank size sweep × bootstrap CI

config:
  1. ctx8_invfr_k3  (現best)
  2. oracle_rerank  (候補内上限)
  3. dtw_oracle     (絶対上限、bank size非依存)

bank sizes: 5, 10, 25, 50, 100

paired bootstrap CI (1000 resamples)
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
from scipy.spatial import cKDTree
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000; FRAME_PERIOD = 5.0; FFTL = 2048; ALPHA = 0.410
MC_ORDER = 24; MC_DIM = 25
VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")
N_PAIRS = 200
BANK_SIZES = [5, 10, 25, 50, 100]
N_CAND = 20
CTX = 8


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


def find_pairs(n=200):
    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))
    pairs = []
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb: continue
                pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                if len(pairs) >= n: return pairs
    return pairs


def compute_speaker_mean(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    return np.concatenate([np.load(f)["mc"] for f in files], axis=0).mean(axis=0)


def build_context_key(mc, spk_mean, ctx=8, weights=None):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


_bank_cache = {}
def get_bank_keys(spk_id, exclude_text, n_utts, spk_mean, inv_fratio):
    cache_key = (spk_id, n_utts)
    if cache_key in _bank_cache:
        return _bank_cache[cache_key]

    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))
    if exclude_text:
        files = [f for f in files if exclude_text not in f.name]
    files = files[:n_utts]

    bank_mc = np.concatenate([np.load(f)["mc"] for f in files], axis=0).astype(np.float32)
    bank_keys = build_context_key(bank_mc, spk_mean, CTX, inv_fratio)
    tree = cKDTree(bank_keys)

    if len(_bank_cache) > 10:
        _bank_cache.clear()
    _bank_cache[cache_key] = (bank_mc, bank_keys, tree)
    return bank_mc, bank_keys, tree


def paired_bootstrap(scores_dict, n_bootstrap=1000, ci=95):
    n = len(next(iter(scores_dict.values())))
    boot_means = {k: [] for k in scores_dict}
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        for k, v in scores_dict.items():
            boot_means[k].append(np.array(v)[idx].mean())
    results = {}
    lo_pct = (100 - ci) / 2
    hi_pct = 100 - lo_pct
    for k in scores_dict:
        arr = np.array(boot_means[k])
        results[k] = {
            "mean": float(arr.mean()),
            "ci_lo": float(np.percentile(arr, lo_pct)),
            "ci_hi": float(np.percentile(arr, hi_pct)),
        }
    return results


def main():
    DEVICE = torch.device("cuda")
    print("=== 大規模評価: 200ペア × bank sweep ===\n")

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
    print(f"ペア数: {len(pairs)}")

    speaker_means = {}
    all_scores = defaultdict(lambda: defaultdict(list))

    t0 = time.time()

    for idx, p in enumerate(pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])
        mc_s = feat_s["mc"]; f0_s = feat_s["f0"]; ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]; f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        for spk in [p["src"], p["tgt"]]:
            if spk not in speaker_means:
                speaker_means[spk] = compute_speaker_mean(spk)
        src_mean = speaker_means[p["src"]]
        tgt_mean = speaker_means[p["tgt"]]

        src_keys = build_context_key(mc_s, src_mean, CTX, inv_fratio)

        dist_dtw, path_dtw = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path_dtw:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR:
            wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        wavs_to_eval = {}
        for n_bank in BANK_SIZES:
            bank_mc, bank_keys, tree = get_bank_keys(
                p["tgt"], p["text"], n_bank, tgt_mean, inv_fratio)
            dist_knn, idx_knn = tree.query(src_keys, k=min(N_CAND, len(bank_mc)))

            w3 = np.exp(-dist_knn[:, :3])
            w3 = w3 / w3.sum(axis=1, keepdims=True)
            mc_ctx = np.einsum('nk,nkd->nd', w3, bank_mc[idx_knn[:, :3]])
            wavs_to_eval[f"ctx8_b{n_bank}"] = synth(
                f0_shifted[:T], mc_ctx[:T].astype(np.float32), ap_s[:T])

            cand_dist = np.sqrt(((bank_mc[idx_knn] - mc_t_aligned[:T, None])**2).sum(axis=2))
            best3 = np.argsort(cand_dist, axis=1)[:, :3]
            w_or = np.exp(-np.take_along_axis(cand_dist, best3, axis=1))
            w_or = w_or / w_or.sum(axis=1, keepdims=True)
            mc_or_g = np.take_along_axis(bank_mc[idx_knn], best3[:, :, None], axis=1)
            mc_or = (mc_or_g * w_or[:, :, None]).sum(axis=1)
            wavs_to_eval[f"oracle_b{n_bank}"] = synth(
                f0_shifted[:T], mc_or[:T].astype(np.float32), ap_s[:T])

        wav_dtw = synth(f0_shifted[:T], mc_t_aligned[:T].astype(np.float32), ap_s[:T])
        wavs_to_eval["dtw_oracle"] = wav_dtw[:len(f0_shifted)]

        with torch.no_grad():
            tgt_t = torch.from_numpy(wav_tgt.astype(np.float32)).unsqueeze(0).to(DEVICE)
            if tgt_t.shape[1] < 8000:
                print(f"  [{idx+1}] SKIP: target too short ({tgt_t.shape[1]} samples)")
                continue
            e_tgt = secs_model.encode_batch(tgt_t).squeeze(0)

            for name, wav in wavs_to_eval.items():
                wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0).to(DEVICE)
                if wav_t.shape[1] < 8000:
                    all_scores[name]["secs"].append(0.0)
                    all_scores[name]["spk_pair"].append(f"{p['src']}→{p['tgt']}")
                    continue
                e = secs_model.encode_batch(wav_t).squeeze(0)
                sim = F.cosine_similarity(e_tgt, e, dim=-1).item()
                all_scores[name]["secs"].append(sim)
                all_scores[name]["spk_pair"].append(f"{p['src']}→{p['tgt']}")

        elapsed = time.time() - t0
        if (idx+1) % 20 == 0:
            ctx10 = np.mean(all_scores["ctx8_b10"]["secs"][-20:])
            or10 = np.mean(all_scores["oracle_b10"]["secs"][-20:])
            speed = (idx+1) / elapsed
            eta = (len(pairs) - idx - 1) / speed
            print(f"  [{idx+1}/{len(pairs)}] ctx8_b10={ctx10:.3f} "
                  f"oracle_b10={or10:.3f} | {speed:.1f}pair/s ETA {eta:.0f}s", flush=True)

    print(f"\n全ペア完了: {time.time()-t0:.0f}s\n")

    print(f"{'='*80}")
    print(f"{'config':<20} {'mean':>8} {'CI_lo':>8} {'CI_hi':>8} {'CI_width':>8} {'n':>5}")
    print(f"{'-'*60}")

    config_order = []
    for n_bank in BANK_SIZES:
        config_order.append(f"ctx8_b{n_bank}")
    for n_bank in BANK_SIZES:
        config_order.append(f"oracle_b{n_bank}")
    config_order.append("dtw_oracle")

    bootstrap_results = {}
    for name in config_order:
        scores = all_scores[name]["secs"]
        arr = np.array(scores)
        bs = paired_bootstrap({"secs": scores})
        m = bs["secs"]["mean"]
        lo = bs["secs"]["ci_lo"]
        hi = bs["secs"]["ci_hi"]
        width = hi - lo
        print(f"{name:<20} {m:>8.4f} {lo:>8.4f} {hi:>8.4f} {width:>8.4f} {len(scores):>5}")
        bootstrap_results[name] = bs["secs"]

    print(f"\n--- bank size trend ---")
    print(f"{'bank_size':>10} {'ctx8':>10} {'oracle':>10} {'gap':>10}")
    print(f"{'-'*45}")
    for n_bank in BANK_SIZES:
        c = bootstrap_results[f"ctx8_b{n_bank}"]["mean"]
        o = bootstrap_results[f"oracle_b{n_bank}"]["mean"]
        print(f"{n_bank:>10} {c:>10.4f} {o:>10.4f} {o-c:>10.4f}")

    dtw_m = bootstrap_results["dtw_oracle"]["mean"]
    print(f"{'dtw_oracle':>10} {'':>10} {'':>10} {dtw_m:>10.4f}")

    print(f"\n--- 判定 ---")
    ctx100 = bootstrap_results["ctx8_b100"]["mean"]
    ctx5 = bootstrap_results["ctx8_b5"]["mean"]
    or100 = bootstrap_results["oracle_b100"]["mean"]
    or5 = bootstrap_results["oracle_b5"]["mean"]

    if ctx100 - ctx5 > 0.02:
        print("bank増加でretrieval上昇 → coverage不足")
    elif or100 - or5 > 0.02 and ctx100 - ctx5 < 0.01:
        print("bank増加でoracle上昇、retrieval停滞 → 候補増えたがranking/key限界")
    else:
        print("bank増加でどちらも停滞 → 手法/特徴量の限界濃厚")

    out = {
        name: {"mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
               "raw_scores": all_scores[name]["secs"]}
        for name, v in zip(config_order, [bootstrap_results[n] for n in config_order])
    }
    with open("results/large_eval_200.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n保存: results/large_eval_200.json")


if __name__ == "__main__":
    main()
