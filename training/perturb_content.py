from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import librosa
import pyworld
from transformers import HubertModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CV_SR = 16000
FP = 5.0


def load_cv() -> HubertModel:
    return HubertModel.from_pretrained("lengyue233/content-vec-best").to(DEV).eval()


def warp_formant(sp: np.ndarray, ratio: float) -> np.ndarray:
    n = sp.shape[1]
    grid = np.arange(n)
    src = np.clip(grid / ratio, 0, n - 1)
    return np.stack([np.interp(src, grid, sp[i]) for i in range(sp.shape[0])])


def perturb(x16: np.ndarray, pitch_ratio: float, formant_ratio: float) -> np.ndarray:
    x = x16.astype(np.float64)
    f0, t = pyworld.harvest(x, CV_SR, f0_floor=65, f0_ceil=1000, frame_period=FP)
    f0 = pyworld.stonemask(x, f0, t, CV_SR)
    sp = pyworld.cheaptrick(x, f0, t, CV_SR)
    ap = pyworld.d4c(x, f0, t, CV_SR)
    sp = warp_formant(sp, formant_ratio)
    y = pyworld.synthesize(f0 * pitch_ratio, sp, ap, CV_SR, FP)
    return y.astype(np.float32)


@torch.no_grad()
def content_of(cv: HubertModel, wav16: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(wav16)).float().view(1, -1).to(DEV)
    return cv(x).last_hidden_state.squeeze(0).half().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--n-aug", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rng = np.random.RandomState(1234)
    cv = load_cv()
    files = sorted(Path(args.feat).rglob("*.pt"))
    print(f"perturb-content for {len(files)} feats, n_aug={args.n_aug}")
    done = 0
    for f in files:
        d = torch.load(f, weights_only=False)
        if "content_pert" in d and not args.overwrite:
            done += 1
            continue
        try:
            y16, _ = librosa.load(d["path"], sr=CV_SR, mono=True)
        except Exception:
            continue
        augs = []
        for _ in range(args.n_aug):
            pr = float(rng.uniform(0.50, 1.15))
            fr = float(rng.uniform(0.78, 1.25))
            yp = perturb(y16, pr, fr)
            augs.append(content_of(cv, yp))
        d["content_pert"] = augs
        torch.save(d, f)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(files)}", flush=True)
    print(f"done: {done}")


if __name__ == "__main__":
    main()
