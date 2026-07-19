from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import librosa
from speechbrain.inference.speaker import EncoderClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--out", default="../data/ecapa_emb.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir="hf_models/spkrec-ecapa",
                                       run_opts={"device": args.device})
    files = list(Path(args.feat).rglob("*.pt"))
    out = {}
    for i, f in enumerate(files):
        try:
            d = torch.load(f, weights_only=False)
            x, _ = librosa.load(d["path"], sr=16000, mono=True)
            e = m.encode_batch(torch.from_numpy(x).float().unsqueeze(0)).squeeze().detach()
            e = e / (e.norm() + 1e-6)
            out[str(d["path"])] = e.cpu().float().numpy()
        except Exception:
            continue
        if i % 300 == 0:
            print(f"{i}/{len(files)}", flush=True)
    torch.save(out, args.out)
    print(f"ecapa emb for {len(out)} utts -> {args.out}")


if __name__ == "__main__":
    main()
