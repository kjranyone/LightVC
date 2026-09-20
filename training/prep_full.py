"""Full-corpus preprocessing for the deployable generator (mel + f0 only).

CLAUDE.md: 本番学習は必ずフルコーパスを使う -- female-dataset (2776 real
speakers) and the irodori TTS corpus (669 speakers), both in full. Every run in
this session used 80 utterances (6.7 minutes, 0.087% of what is available), and
PESQ 2.45 is what that buys. The refiner did not care about data volume because
its prior already carried the information; a GENERATOR asked to build a waveform
from f0 and mel has to learn voice variety itself, and 6.7 minutes cannot teach
that.

Only mel and f0 are extracted. The generator's prior is f0 sinusoids, so the
expensive parts of rddsp.analyze -- ZFF/GCI detection, least-squares harmonic
amplitudes, complex cepstrum -- are not needed. Measured on this machine:

    mel + f0    0.147 s/utt  ->  3.7 h single-core for 91570 utterances
    full rddsp  1.719 s/utt  -> 43.7 h

Sharded to disk because the corpus does not fit in memory: 91570 utterances of
5 s at 44.1 kHz is 40 GB of waveform alone. Each shard stores the waveform as
int16 (half the size, and PESQ-irrelevant at this depth), the 80-band log-mel,
and the f0 track.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ONE thread per worker. Without this each of 11 workers spawns 12 BLAS/torch
# threads on a 12-core box -- 132 threads thrashing. Measured: 0.050 s/utt pinned
# vs no shard written at all in 53 minutes unpinned.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import librosa
import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_neural import mel_of

ROOT = Path("/home/kojirotanaka/kjranyone/LightVC")
OUT = ROOT / "data/full_prep"
SOURCES = [ROOT / "female-dataset", ROOT / "data/female_tts_corpus"]
SECONDS = 5
PER_SHARD = 500


def utt_list() -> list[tuple[str, Path]]:
    """Every wav, tagged with its speaker, sorted so shards are reproducible."""
    out = []
    for src in SOURCES:
        if not src.exists():
            continue
        tag = src.name
        for spk in sorted(p for p in src.iterdir() if p.is_dir()):
            for w in sorted(spk.glob("*.wav")):
                out.append((f"{tag}/{spk.name}", w))
    return out


def one(path: Path):
    x, _ = librosa.load(str(path), sr=R.SR, mono=True)
    if len(x) < R.SR:                       # under a second is not a training item
        return None
    gt = torch.tensor(x[: R.SR * SECONDS])
    if float(gt.abs().max()) < 1e-4:
        return None
    try:
        f0, _voi = R.harmonic_sum_f0(gt)
    except Exception:
        return None
    if not bool((f0 > 50).any()):            # no voiced frame -> nothing for the prior
        return None
    mel = mel_of(gt)
    return dict(w=(gt.clamp(-1, 1) * 32767).to(torch.int16),
                mel=mel.to(torch.float16), f0=f0.to(torch.float16))


def main() -> None:
    shard_id = int(sys.argv[1])
    n_shard = int(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    items = utt_list()
    mine = [(s, p) for i, (s, p) in enumerate(items) if i % n_shard == shard_id]
    print(f"  worker {shard_id}/{n_shard}: {len(mine)} of {len(items)} utterances",
          flush=True)
    buf, spk, nsh, t0 = [], [], 0, time.time()
    for i, (s, p) in enumerate(mine):
        d = one(p)
        if d is None:
            continue
        d["spk"] = s
        buf.append(d)
        if len(buf) >= PER_SHARD:
            torch.save(buf, OUT / f"sh_{shard_id:02d}_{nsh:04d}.pt")
            nsh += 1
            buf = []
            el = time.time() - t0
            print(f"  worker {shard_id}: {i+1}/{len(mine)} "
                  f"({100*(i+1)/len(mine):.1f}%) {el/60:.1f} min "
                  f"eta {el/(i+1)*(len(mine)-i-1)/60:.0f} min", flush=True)
    if buf:
        torch.save(buf, OUT / f"sh_{shard_id:02d}_{nsh:04d}.pt")
    print(f"  worker {shard_id} done: {nsh + (1 if buf else 0)} shards, "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
