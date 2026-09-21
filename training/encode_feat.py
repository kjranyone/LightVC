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
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--limit-per-spk", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", default=None,
                    help="既存featキャッシュの検証のみ実行(生成はしない)")
    args = ap.parse_args()
    if args.verify:
        raise SystemExit(verify_cache(args.verify))
    if not args.corpus or not args.out:
        raise SystemExit("--corpus/--out required (or use --verify)")
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




def verify_cache(out_dir: str, n: int = 40, seed: int = 0) -> int:
    """保存featと元音声のサンプル相関プローブ(f0破損85%事故の恒久対策)。

    新規feat生成後に必ず実行する:
      uv run python encode_feat.py --verify ../data/<new_feat>
    f0輪郭相関の中央値<0.9 または energy相関<0.99 なら非ゼロexit。
    """
    import sys as _sys
    import random as _r
    import pyworld as _pw
    root = Path(out_dir)
    rng = _r.Random(seed)
    fs = [q for q in root.rglob("*.pt")]
    rng.shuffle(fs)
    fs = fs[:n]
    cs, es = [], []
    for f in fs:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            w44, _ = librosa.load(d["path"], sr=44100, mono=True)
            w64 = w44.astype("float64")
            f0, t = _pw.harvest(w64, 44100, f0_floor=65, f0_ceil=1000,
                                frame_period=512 / 44100 * 1000)
            fr = _pw.stonemask(w64, f0, t, 44100)
            m = len(w44) // 512
            e = np.sqrt((w44[: m * 512].reshape(m, 512) ** 2).mean(-1))
            st_f, st_e = d["f0"].numpy(), d["energy"].numpy()
            nf = min(len(fr), len(st_f))
            mf = (fr[:nf] > 60) & (st_f[:nf] > 60)
            if mf.sum() >= 30:
                cs.append(float(np.corrcoef(fr[:nf][mf], st_f[:nf][mf])[0, 1]))
            ne = min(len(e), len(st_e))
            if ne >= 30:
                es.append(float(np.corrcoef(e[:ne], st_e[:ne])[0, 1]))
        except Exception:
            continue
    cm = float(np.median(cs)) if cs else 0.0
    em = float(np.median(es)) if es else 0.0
    ok = cm >= 0.9 and em >= 0.99
    print(f"verify {root}: n={len(cs)} f0-corr median {cm:.3f} "
          f"energy-corr median {em:.3f} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
