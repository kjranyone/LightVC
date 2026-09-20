"""Build a reproducible multi-speaker female held set for Z0-attr / Z1 eval.

Reviewer flagged n=24 = 4 speakers as insufficient (speaker-level bootstrap needs
many speakers). Sample N_SPK female_real speakers deterministically (sorted), K
utts each that (i) have f0, (ii) have an existing wav, (iii) 2-9 s. Persist the
feat-path manifest so eval is reproducible and speaker-level bootstrap is valid.
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/female_real_feat")
    ap.add_argument("--n-spk", type=int, default=30)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=9.0)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--out", default="../results/interpretable_vc/held_multispk.txt")
    args = ap.parse_args()

    root = Path(args.root)
    spks = sorted([p for p in root.iterdir() if p.is_dir()])
    picked, kept_spk = [], 0
    for s in spks:
        if kept_spk >= args.n_spk:
            break
        utts = sorted(s.glob("*.pt"))
        chosen = []
        for f in utts:
            if len(chosen) >= args.k:
                break
            try:
                d = torch.load(f, weights_only=False)
            except Exception:
                continue
            if "f0" not in d:
                continue
            wp = d.get("path")
            if not wp or not Path(wp).exists():
                continue
            n = len(d["f0"])
            sec = n * args.hop / args.sr
            if not (args.min_sec <= sec <= args.max_sec):
                continue
            chosen.append(str(f.resolve()))
        if len(chosen) == args.k:
            picked += chosen
            kept_spk += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(picked) + "\n")
    print(f"held: {kept_spk} speakers x {args.k} = {len(picked)} utts -> {out}")


if __name__ == "__main__":
    main()
