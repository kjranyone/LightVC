from __future__ import annotations

import argparse
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import librosa
import pyworld
import torch

SR = 16000
STYLES = ["neutral", "soft", "warm", "breathy", "low_tension",
          "cute_high", "intimate_close", "young_bright"]


def style_name(s: str) -> str:
    p = s.split("_", 1)
    return p[1] if len(p) > 1 else s


def formant_track(x: np.ndarray, order: int = 12) -> np.ndarray:
    win = int(0.025 * SR)
    hop = int(SR * 0.01)
    out = []
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win] * np.hanning(win)
        pre = np.append(fr[0], fr[1:] - 0.97 * fr[:-1])
        try:
            a = librosa.lpc(pre.astype(np.float32), order=order)
            r = np.roots(a)
            r = r[np.imag(r) > 0]
            ang = np.sort(np.arctan2(np.imag(r), np.real(r)) * SR / (2 * np.pi))
            ang = ang[ang > 90]
            out.append(ang[:3] if len(ang) >= 3 else np.pad(ang, (0, 3 - len(ang)), constant_values=np.nan))
        except Exception:
            out.append(np.array([np.nan, np.nan, np.nan]))
    return np.array(out)


def work(fp: str):
    try:
        d = torch.load(fp, weights_only=False)
        st = style_name(d["style"])
        if st not in STYLES:
            return None
        x, _ = librosa.load(d["path"], sr=SR, mono=True)
        ft = formant_track(x.astype(np.float64))
        if len(ft) < 24 or not np.isfinite(ft).any():
            return None
        ft = np.nan_to_num(ft, nan=np.nanmedian(ft)).astype(np.float32)
        cont = d["content"].float().numpy().astype(np.float16)
        return {"content": cont, "formant": ft, "style": st, "speaker": d["speaker"]}
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/c76a325d-9c57-4dc0-bd41-4abd61a25a89/scratchpad/formant_cache.pt")
    args = ap.parse_args()
    random.seed(0)
    files = [str(f) for f in Path(args.feat).rglob("*.pt")]
    random.shuffle(files)
    files = files[: args.n * 3]
    with Pool(12) as p:
        res = [r for r in p.map(work, files) if r is not None][: args.n]
    torch.save(res, args.out)
    print(f"cached {len(res)} utts -> {args.out}")


if __name__ == "__main__":
    main()
