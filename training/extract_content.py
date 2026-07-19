from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import librosa
from transformers import HubertModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CV_SR = 16000


def load_contentvec() -> HubertModel:
    m = HubertModel.from_pretrained("lengyue233/content-vec-best")
    return m.to(DEV).eval()


@torch.no_grad()
def content_of(model: HubertModel, wav16: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(wav16).float().view(1, -1).to(DEV)
    h = model(x).last_hidden_state.squeeze(0)
    return h.half().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--limit-per-spk", type=int, default=0)
    ap.add_argument("--min-sec", type=float, default=1.0)
    args = ap.parse_args()

    model = load_contentvec()
    out_root = Path(args.out)
    total = 0
    for d in args.dirs:
        spk_dir = Path(d)
        spk = spk_dir.name
        wavs = sorted(spk_dir.glob(args.glob))
        if args.limit_per_spk:
            wavs = wavs[: args.limit_per_spk]
        dst = out_root / spk
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for w in wavs:
            y, _ = librosa.load(str(w), sr=CV_SR, mono=True)
            if len(y) < args.min_sec * CV_SR:
                continue
            c = content_of(model, y)
            torch.save({"content": c, "path": str(w.resolve()), "speaker": spk,
                        "style": w.stem, "dur": round(len(y) / CV_SR, 2)},
                       dst / f"{w.stem}.pt")
            n += 1
            total += 1
        print(f"  {spk}: {n} utts", flush=True)
    print(f"done: {total} utts -> {out_root}")


if __name__ == "__main__":
    main()
