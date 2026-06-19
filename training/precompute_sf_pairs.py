"""
Pre-compute DTW-aligned mel-cepstrum pairs for SF-VC training.

For each text group, generate N random speaker pairs.
DTW align source mc to target mc.
Save aligned pair as .npz.

Output: data/sf_pairs/pair_XXXXXX.npz
  mc_src: source mel-cepstrum [T, 25]
  mc_tgt: DTW-aligned target mel-cepstrum [T, 25]
  f0_src: source F0 [T]
  codeap_tgt: aligned target aperiodicity [T, 1]
  spk_src: source speaker ID (string)
  spk_tgt: target speaker ID (string)
"""
import sys, os, time, random, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from fastdtw import fastdtw

SR = 16000
MC_DIM = 25
VCTK_MC_CACHE = Path("data/mc_cache")
OUTPUT_DIR = Path("data/sf_pairs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_per_text", type=int, default=30)
    parser.add_argument("--max_total", type=int, default=10000)
    args = parser.parse_args()

    print("Indexing text groups...")
    text_groups = defaultdict(list)
    for spk_dir in sorted(VCTK_MC_CACHE.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for npz_path in spk_dir.glob("*.npz"):
            stem = npz_path.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                text_id = parts[1]
                text_groups[text_id].append((spk, str(npz_path)))

    valid_texts = {t: utts for t, utts in text_groups.items() if len(utts) >= 2}
    print(f"  {len(valid_texts)} text groups with ≥2 speakers")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pair_idx = 0
    t0 = time.time()
    total_possible = sum(min(len(utts) * (len(utts) - 1) // 2, args.pairs_per_text) for utts in valid_texts.values())
    target_count = min(args.max_total, total_possible)
    print(f"Target: {target_count} pairs ({args.pairs_per_text}/text)")

    for text_id, utts in sorted(valid_texts.items()):
        if pair_idx >= target_count:
            break

        n_pairs = min(args.pairs_per_text, len(utts) * (len(utts) - 1) // 2)
        pairs_made = 0
        attempts = 0
        while pairs_made < n_pairs and attempts < n_pairs * 3 and pair_idx < target_count:
            attempts += 1
            src_idx, tgt_idx = random.sample(range(len(utts)), 2)
            src_spk, src_path = utts[src_idx]
            tgt_spk, tgt_path = utts[tgt_idx]

            try:
                src_data = np.load(src_path)
                tgt_data = np.load(tgt_path)

                mc_src = src_data["mc"]
                mc_tgt = tgt_data["mc"]

                if len(mc_src) < 20 or len(mc_tgt) < 20:
                    continue
                if len(mc_src) > 600 or len(mc_tgt) > 600:
                    mc_src = mc_src[:600]
                    mc_tgt = mc_tgt[:600]

                dist, path = fastdtw(mc_src, mc_tgt, radius=20)

                src_map = np.zeros(len(mc_src), dtype=int)
                last = 0
                for s_idx, t_idx in path:
                    if s_idx < len(mc_src) and t_idx < len(mc_tgt):
                        src_map[s_idx] = t_idx
                        last = t_idx

                for i in range(1, len(src_map)):
                    if src_map[i] == 0:
                        src_map[i] = src_map[i - 1]

                mc_tgt_aligned = mc_tgt[src_map]
                codeap_aligned = tgt_data["codeap"][src_map]
                f0_src = src_data["f0"][:len(mc_src)]

                out_path = OUTPUT_DIR / f"pair_{pair_idx:06d}.npz"
                np.savez(out_path,
                         mc_src=mc_src.astype(np.float32),
                         mc_tgt=mc_tgt_aligned.astype(np.float32),
                         f0_src=f0_src.astype(np.float32),
                         codeap_tgt=codeap_aligned.astype(np.float32),
                         spk_src=src_spk,
                         spk_tgt=tgt_spk)

                pair_idx += 1
                pairs_made += 1
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        if (pair_idx % 500) < n_pairs:
            elapsed = time.time() - t0
            rate = pair_idx / max(elapsed, 0.1)
            eta = (target_count - pair_idx) / max(rate, 0.1)
            print(f"  {pair_idx}/{target_count} ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {pair_idx} pairs in {elapsed:.0f}s ({pair_idx/elapsed:.1f}/s)")
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
