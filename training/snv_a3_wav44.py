"""A3(diag): E1 試作のコーパスの 44.1 kHz の波形を、BigVGAN の log-mel のフレーム数 × 512 の長さに切って float16 で置く(自前の因果の演奏の信号 E の入力)。
出力 = data/snv_e1_feat/wav44/e1/<声>/<t>.npy。mel のフレーム数 = 長さ // 512 を全件 assert する。
    cd training && uv run --no-sync python snv_a3_wav44.py
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/snv_e1_feat"
SR = 44100
HOP = 512


def job(p: Path) -> int:
    q = FEAT / f"wav44/e1/{p.parent.name}/{p.stem}.npy"
    n = np.load(FEAT / f"mel/e1/{p.parent.name}/{p.stem}.npy", mmap_mode="r").shape[1]
    if q.exists():
        return n
    y = librosa.load(str(p), sr=SR, mono=True)[0]
    assert len(y) // HOP == n, f"{p}: 長さ {len(y)} // 512 != mel {n}"
    q.parent.mkdir(parents=True, exist_ok=True)
    np.save(q, y[: n * HOP].astype(np.float16))
    return n


def main() -> int:
    ps = sorted((ROOT / "data/snv_e1").glob("e*/t*.wav"))
    with ProcessPoolExecutor(16) as ex:
        ns = list(ex.map(job, ps, chunksize=16))
    print("done", len(ps), "frames", sum(ns), "hours", sum(ns) * HOP / SR / 3600, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
