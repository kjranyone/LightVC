"""Full-corpus encoder -> content/f0/energy feat (matches female_real_feat/rcav_feat
format exactly, verified 2026-07-20). Per CLAUDE.md: encode NEW corpora fully, no
small subsets. content=ContentVec(16k), f0=pyworld harvest+stonemask(44.1k,HOP512),
energy=non-overlap HOP512 frame RMS (corr 1.000 vs stored). Shardable for parallel.

Usage (4 shards in parallel):
  for i in 0 1 2 3; do uv run python encode_feat.py --corpus ../data/female_tts_corpus \
      --out ../data/female_tts_feat --shard $i/4 & done
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np, torch, librosa, pyworld, soundfile as sf
from transformers import HubertModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR, HOP, CV_SR = 44100, 512, 16000


def load_cv():
    return HubertModel.from_pretrained("lengyue233/content-vec-best").to(DEV).eval()


@torch.no_grad()
def content_of(cv, w16):
    x = torch.from_numpy(np.ascontiguousarray(w16)).float().view(1, -1).to(DEV)
    return cv(x).last_hidden_state.squeeze(0).half().cpu()


def f0_of(w44, n):
    w64 = w44.astype(np.float64)
    f0, t = pyworld.harvest(w64, SR, f0_floor=65, f0_ceil=1000, frame_period=HOP / SR * 1000)
    f0 = pyworld.stonemask(w64, f0, t, SR).astype(np.float32)
    return f0[:n] if len(f0) >= n else np.pad(f0, (0, n - len(f0)))


def energy_of(w44, n):
    m = len(w44) // HOP
    e = np.sqrt((w44[: m * HOP].reshape(m, HOP) ** 2).mean(-1)).astype(np.float32)
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--limit-per-spk", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    si, sn = (int(x) for x in args.shard.split("/"))
    spks = sorted(d for d in Path(args.corpus).iterdir() if d.is_dir())
    spks = [s for k, s in enumerate(spks) if k % sn == si]
    cv = load_cv()
    out_root = Path(args.out)
    total = skip = 0
    for sd in spks:
        wavs = sorted(sd.glob(args.glob))
        if args.limit_per_spk:
            wavs = wavs[: args.limit_per_spk]
        dst = out_root / sd.name
        dst.mkdir(parents=True, exist_ok=True)
        for w in wavs:
            op = dst / f"{w.stem}.pt"
            if op.exists() and not args.overwrite:
                skip += 1; continue
            try:
                w16, _ = librosa.load(str(w), sr=CV_SR, mono=True)
                if len(w16) < args.min_sec * CV_SR:
                    continue
                w44, sr = sf.read(str(w), dtype="float32")
                if w44.ndim > 1:
                    w44 = w44.mean(1)
                if sr != SR:
                    w44 = librosa.resample(w44, orig_sr=sr, target_sr=SR)
                n = len(w44) // HOP
                if n < 4:
                    continue
                d = {"content": content_of(cv, w16),
                     "f0": torch.from_numpy(f0_of(w44, n)),
                     "energy": torch.from_numpy(energy_of(w44, n)),
                     "speaker": sd.name, "path": str(w.resolve()),
                     "style": w.stem, "dur": round(len(w16) / CV_SR, 2)}
                torch.save(d, op)
                total += 1
            except Exception as e:
                print(f"  skip {w.name}: {str(e)[:50]}", flush=True)
        print(f"[shard {si}/{sn}] {sd.name}: {total} enc / {skip} skip", flush=True)
    print(f"[shard {si}/{sn}] done: {total} encoded, {skip} skipped -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
