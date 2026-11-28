"""段 P97(paper_snv): 描画器の学び直しの音(data/snv_e1・data/snv_e1nv の 48 kHz・PCM16)を、可逆の FLAC で data/snv_p97_wav/{e1,e1nv}/<声>/<名前>.flac に置く(Colab への転送の量を減らす・標本は元とビット一致)。
    cd training && uv run --no-sync python snv_p97_pack.py
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "data/snv_p97_wav"


def one(job: tuple[str, str]) -> int:
    src, dst = Path(job[0]), Path(job[1])
    x, sr = sf.read(str(src), dtype="int16")
    assert sr == 48000 and x.ndim == 1, (src, sr, x.shape)
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.flac")
        sf.write(str(tmp), x, sr, format="FLAC", subtype="PCM_16")
        tmp.rename(dst)
    y, _ = sf.read(str(dst), dtype="int16")
    assert np.array_equal(x, y), dst
    return len(x)


def main() -> int:
    jobs = [(str(w), str(DST / "e1" / w.parent.name / (w.stem + ".flac"))) for w in sorted((ROOT / "data/snv_e1").glob("*/t*.wav"))]
    jobs += [(str(w), str(DST / "e1nv" / w.parent.name / (w.stem + ".flac"))) for w in sorted((ROOT / "data/snv_e1nv").glob("*/*.wav"))]
    with ProcessPoolExecutor(16) as ex:
        n = list(ex.map(one, jobs, chunksize=32))
    (DST / "manifest.json").write_text(json.dumps({"files": len(jobs), "samples": int(sum(n)), "hours": round(sum(n) / 48000 / 3600, 2), "from": ["data/snv_e1", "data/snv_e1nv"], "format": "FLAC PCM_16 48 kHz(元と標本がビット一致)"}, ensure_ascii=False, indent=1))
    print("files", len(jobs), "hours", round(sum(n) / 48000 / 3600, 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
