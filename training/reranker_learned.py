"""
Learned pairwise reranker

top-20候補の順位付けを学習する。mcep回帰なし、純粋なranking。

score(s, c) = MLP(src_mcep, cand_mcep, diff, d_content)

Loss: pairwise margin ranking
  L = mean(max(0, margin - score(s, pos) + score(s, neg)))

pos = oracle best (DTW targetに最も近い候補)
neg = 他候補（hard negative優先）
"""
import sys, json, time, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import pyworld as world
import pysptk as sptk
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.spatial import cKDTree
from fastdtw import fastdtw

sys.path.insert(0, str(Path(__file__).parent))

SR = 16000; FRAME_PERIOD = 5.0; FFTL = 2048; ALPHA = 0.410
MC_ORDER = 24; MC_DIM = 25
VCTK_WAV = Path("../data/vctk_200")
MC_CACHE = Path("data/mc_cache")
SF_PAIRS = Path("data/sf_pairs")
CTX = 8; N_CAND = 20


class CandidateReranker(nn.Module):
    def __init__(self, mc_dim=MC_DIM, hidden=64):
        super().__init__()
        inp = mc_dim * 3 + 2
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, src_mc, cand_mc, d_content, f0_diff):
        diff = src_mc - cand_mc
        x = torch.cat([src_mc, cand_mc, diff, d_content.unsqueeze(-1), f0_diff.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


def compute_speaker_mean(spk_id, n_utts=30):
    spk_dir = MC_CACHE / spk_id
    files = sorted(spk_dir.glob("*.npz"))[:n_utts]
    return np.concatenate([np.load(f)["mc"] for f in files], axis=0).mean(axis=0)


def build_bank(tgt_spk, exclude_text, n_utts=10):
    spk_dir = MC_CACHE / tgt_spk
    files = sorted(spk_dir.glob("*.npz"))
    if exclude_text:
        files = [f for f in files if exclude_text not in f.name]
    files = files[:n_utts]
    bank_mc = np.concatenate([np.load(f)["mc"] for f in files], axis=0).astype(np.float32)
    bank_f0 = np.concatenate([np.load(f)["f0"] for f in files], axis=0).astype(np.float32)
    return bank_mc, bank_f0


def build_context_key(mc, spk_mean, ctx=8, weights=None):
    T = len(mc)
    mc_norm = (mc - spk_mean) * weights[None, :] if weights is not None else mc - spk_mean
    if ctx > 0:
        padded = np.pad(mc_norm, ((ctx, ctx), (0, 0)), mode="edge")
        return np.stack([padded[i:i+T] for i in range(2*ctx+1)], axis=-1).reshape(T, -1)
    return mc_norm


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


def precompute_candidates(pair_data, speaker_means, fratio_inv, n_cand=N_CAND):
    mc_src = pair_data["mc_src"].astype(np.float32)
    mc_tgt = pair_data["mc_tgt"].astype(np.float32)
    f0_src = pair_data["f0_src"].astype(np.float32)
    spk_tgt = str(pair_data["spk_tgt"])
    T = len(mc_src)

    tgt_mean = speaker_means.get(spk_tgt, compute_speaker_mean(spk_tgt))
    speaker_means[spk_tgt] = tgt_mean
    src_spk = str(pair_data["spk_src"])
    src_mean = speaker_means.get(src_spk, compute_speaker_mean(src_spk))
    speaker_means[src_spk] = src_mean

    bank_mc, bank_f0 = build_bank(spk_tgt, "", n_utts=10)

    src_keys = build_context_key(mc_src, src_mean, CTX, fratio_inv)
    bank_keys = build_context_key(bank_mc, tgt_mean, CTX, fratio_inv)

    tree = cKDTree(bank_keys)
    dist_knn, idx_knn = tree.query(src_keys, k=n_cand)

    cand_dist_to_tgt = np.sqrt(((bank_mc[idx_knn] - mc_tgt[:, None, :])**2).sum(axis=2))
    oracle_rank = np.argsort(cand_dist_to_tgt, axis=1)
    oracle_best = oracle_rank[:, 0]

    return {
        "mc_src": mc_src[:T],
        "f0_src": f0_src[:T],
        "cands_mc": bank_mc[idx_knn][:T],
        "cands_f0": bank_f0[idx_knn][:T],
        "d_content": dist_knn[:T],
        "oracle_best": oracle_best[:T],
        "oracle_rank": oracle_rank[:T],
        "cand_dist_to_tgt": cand_dist_to_tgt[:T],
    }


def train(args):
    DEVICE = torch.device("cuda")
    print("=== Learned Pairwise Reranker ===\n")

    fratio = np.load("data/mc_fratio_weights.npy")
    inv_fratio = (1.0 / (fratio + 1e-6)).astype(np.float32)
    inv_fratio = inv_fratio / inv_fratio.mean()

    print("Precomputing training candidates...")
    pair_files = sorted(SF_PAIRS.glob("pair_*.npz"))
    random.shuffle(pair_files)
    train_files = pair_files[:args.n_train]
    print(f"  Train pairs: {len(train_files)}")

    speaker_means = {}
    train_data = []
    t0 = time.time()
    for i, pf in enumerate(train_files):
        try:
            pair_data = np.load(pf)
            data = precompute_candidates(pair_data, speaker_means, inv_fratio)
            train_data.append(data)
        except Exception as e:
            print(f"  SKIP {pf.name}: {e}")
            continue
        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(train_files)}] {time.time()-t0:.0f}s", flush=True)

    print(f"  Precompute done in {time.time()-t0:.0f}s")

    model = CandidateReranker(hidden=args.hidden).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nTraining {args.max_steps} steps...\n")

    step = 0
    log_interval = 500
    t0 = time.time()

    while step < args.max_steps:
        random.shuffle(train_data)
        for data in train_data:
            if step >= args.max_steps: break
            T = len(data["mc_src"])
            mc_src = torch.from_numpy(data["mc_src"]).float().to(DEVICE)
            f0_src = torch.from_numpy(data["f0_src"]).float().to(DEVICE)
            cands_mc = torch.from_numpy(data["cands_mc"]).float().to(DEVICE)
            cands_f0 = torch.from_numpy(data["cands_f0"]).float().to(DEVICE)
            d_content = torch.from_numpy(data["d_content"]).float().to(DEVICE)
            oracle_best = torch.from_numpy(data["oracle_best"]).long().to(DEVICE)

            N = cands_mc.shape[1]

            pos_idx = oracle_best
            pos_mc = cands_mc[torch.arange(T), pos_idx]
            pos_f0 = cands_f0[torch.arange(T), pos_idx]
            pos_d = d_content[torch.arange(T), pos_idx]

            neg_idx = torch.randint(0, N, (T,), device=DEVICE)
            same = neg_idx == pos_idx
            neg_idx[same] = (neg_idx[same] + 1) % N
            neg_mc = cands_mc[torch.arange(T), neg_idx]
            neg_f0 = cands_f0[torch.arange(T), neg_idx]
            neg_d = d_content[torch.arange(T), neg_idx]

            src_flat = mc_src
            f0_diff_pos = (f0_src - pos_f0).abs().log1p()
            f0_diff_neg = (f0_src - neg_f0).abs().log1p()

            pos_score = model(src_flat, pos_mc, pos_d, f0_diff_pos)
            neg_score = model(src_flat, neg_mc, neg_d, f0_diff_neg)

            loss = F.relu(args.margin - pos_score + neg_score).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            if step % log_interval == 0:
                acc = (pos_score > neg_score).float().mean().item()
                speed = step / (time.time() - t0)
                print(f"step {step}/{args.max_steps} | loss={loss.item():.4f} "
                      f"acc={acc:.3f} | {speed:.1f}step/s", flush=True)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": {"hidden": args.hidden}},
               output_dir / "reranker.pt")
    print(f"\nSaved: {output_dir / 'reranker.pt'}")


def evaluate(args):
    DEVICE = torch.device("cuda")
    print("\n=== Evaluating Learned Reranker ===\n")

    from speechbrain.inference.speaker import EncoderClassifier
    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(DEVICE)},
    )

    fratio = np.load("data/mc_fratio_weights.npy")
    inv_fratio = (1.0 / (fratio + 1e-6)).astype(np.float32)
    inv_fratio = inv_fratio / inv_fratio.mean()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = CandidateReranker(hidden=ckpt["config"]["hidden"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pairs = find_pairs(20)
    speaker_means = {}

    results = defaultdict(list)

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

        bank_mc, bank_f0 = build_bank(p["tgt"], p["text"], n_utts=10)
        src_keys = build_context_key(mc_s, src_mean, CTX, inv_fratio)
        bank_keys = build_context_key(bank_mc, tgt_mean, CTX, inv_fratio)

        tree = cKDTree(bank_keys)
        dist_knn, idx_knn = tree.query(src_keys, k=N_CAND)

        dist_dtw, path_dtw = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path_dtw:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        cands_mc = bank_mc[idx_knn]
        cands_f0 = bank_f0[idx_knn]

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(DEVICE)).squeeze(0)
            e_tgt = emb(wav_tgt)

            # baseline (kNN top-3 blend)
            w3 = np.exp(-dist_knn[:, :3])
            w3 = w3 / w3.sum(axis=1, keepdims=True)
            mc_base = np.einsum('nk,nkd->nd', w3, cands_mc[:, :3])
            wav_base = synth(f0_shifted[:T], mc_base[:T].astype(np.float32), ap_s[:T])
            sim_base = F.cosine_similarity(e_tgt, emb(wav_base), dim=-1).item()
            results["baseline"].append(sim_base)

            # learned reranker
            mc_src_t = torch.from_numpy(mc_s[:T]).float().to(DEVICE)
            f0_src_t = torch.from_numpy(f0_s[:T]).float().to(DEVICE)
            scores_all = []
            for k in range(N_CAND):
                cand_t = torch.from_numpy(cands_mc[:T, k]).float().to(DEVICE)
                d_t = torch.from_numpy(dist_knn[:T, k]).float().to(DEVICE)
                f0_diff = (f0_src_t - torch.from_numpy(cands_f0[:T, k]).float().to(DEVICE)).abs().log1p()
                s = model(mc_src_t, cand_t, d_t, f0_diff)
                scores_all.append(s)

            scores = torch.stack(scores_all, dim=1)  # (T, N_CAND)

            top3_idx = scores.topk(3, dim=1).indices  # (T, 3)
            top3_scores = scores.topk(3, dim=1).values  # (T, 3)
            w = F.softmax(top3_scores, dim=1)  # (T, 3)

            cands_mc_t = torch.from_numpy(cands_mc[:T]).float().to(DEVICE)
            idx_expanded = top3_idx.unsqueeze(-1).expand(-1, -1, MC_DIM)
            mc_top3 = torch.gather(cands_mc_t, 1, idx_expanded)
            mc_reranked = (w.unsqueeze(-1) * mc_top3).sum(dim=1)

            wav_rr = synth(f0_shifted[:T], mc_reranked.cpu().numpy()[:T].astype(np.float32), ap_s[:T])
            sim_rr = F.cosine_similarity(e_tgt, emb(wav_rr), dim=-1).item()
            results["reranked"].append(sim_rr)

            # oracle rerank (ceiling)
            cand_dist = np.sqrt(((cands_mc[:T] - mc_t_aligned[:T, None])**2).sum(axis=2))
            best3 = np.argsort(cand_dist, axis=1)[:, :3]
            w_or = np.exp(-np.take_along_axis(cand_dist, best3, axis=1))
            w_or = w_or / w_or.sum(axis=1, keepdims=True)
            mc_or_gathered = np.take_along_axis(cands_mc[:T], best3[:, :, None], axis=1)
            mc_or = (mc_or_gathered * w_or[:, :, None]).sum(axis=1)
            wav_or = synth(f0_shifted[:T], mc_or.astype(np.float32), ap_s[:T])
            sim_or = F.cosine_similarity(e_tgt, emb(wav_or), dim=-1).item()
            results["oracle"].append(sim_or)

        print(f"  [{idx+1}/{len(pairs)}] base={sim_base:.3f} "
              f"rr={sim_rr:.3f} oracle={sim_or:.3f}", flush=True)

    print(f"\n{'='*50}")
    print(f"{'config':<15} {'mean':>8} {'std':>8}")
    print(f"{'-'*35}")
    for name in ["baseline", "reranked", "oracle"]:
        arr = np.array(results[name])
        print(f"{name:<15} {arr.mean():>8.4f} {arr.std():>8.4f}")

    rr_score = np.mean(results["reranked"])
    print(f"\nReranked = {rr_score:.4f}")
    if rr_score >= 0.48:
        print("→ Go条件 (>= 0.48) クリア!")
    elif rr_score >= 0.46:
        print("→ heuristic Go条件クリア、tuningで伸びるかも")
    elif rr_score >= 0.43:
        print("→ baseline同等、改善限界かも")
    else:
        print("→ baseline下回る → C路線へ")

    out = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))}
           for name, v in results.items()}
    with open("results/reranker_learned.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train")
    train_p.add_argument("--output", default="checkpoints/reranker")
    train_p.add_argument("--n_train", type=int, default=500)
    train_p.add_argument("--max_steps", type=int, default=20000)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--hidden", type=int, default=64)
    train_p.add_argument("--margin", type=float, default=0.5)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", default="checkpoints/reranker/reranker.pt")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
        args.checkpoint = str(Path(args.output) / "reranker.pt")
        evaluate(args)
    elif args.cmd == "eval":
        evaluate(args)
