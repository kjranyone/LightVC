"""D260(学習なし・results/conv_p0/d260_prereg.yaml): D256 の Δ(目標の表の数で重みづけた平均 − 操作者の表の同じ平均・c1..c24)と同じ音色の差を実女声の STFT の振幅に掛け(SRCD)、新しい前段 + rvoc3hi で写し合成する(CSRCD)。
    uv run python d260_deltaeq.py --s <dev の親>
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.signal as ss
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d257_eq import BIN, SR, job, load48
from d259_eqcopy import COPY


def tab_mean(d: Path) -> np.ndarray:
    j = json.loads((d / "enroll.json").read_text())
    k = len(j["counts"])
    T = np.fromfile(d / "enroll.bin", "<f4").reshape(-1, k)
    c = np.asarray(j["counts"], np.float64)
    return (T * c[None]).sum(1) / max(c.sum(), 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    a = ap.parse_args()
    import nvoc as N
    import pae as PA
    fb, dct = PA._np_consts()
    fc = (fb * np.arange(fb.shape[1])[None]).sum(1) / np.maximum(fb.sum(1), 1e-9) * (N.SR / 2) / (fb.shape[1] - 1)
    F = Path(a.s) / "devff"
    od = F / "d260"
    od.mkdir(exist_ok=True)
    jobs = []
    for f in sorted((F / "d56").glob("p00*_SRC.wav")):
        pi, ti = int(f.name[1:5]), int(f.name[7:10])
        k = f.stem[:-4]
        delta = tab_mean(F / "d85_work" / f"en_{ti:03d}") - tab_mean(F / "d85_work" / f"op_{pi:04d}")
        dl = dct[1:25].T @ delta[:24]
        x = load48(f)
        fq, _, Z = ss.stft(x, SR, nperseg=2048, noverlap=2048 - 240)
        g = np.exp(np.interp(fq, fc, dl))
        _, y = ss.istft(Z * g[:, None], SR, nperseg=2048, noverlap=2048 - 240)
        y = y[: len(x)]
        sf.write(od / f"{k}_SRCD.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), SR)
        op = F / "d85_work" / f"op_{pi:04d}"
        jobs.append([BIN, "artic", "--enroll", op, "--operator-enroll", op] + COPY + ["-i", od / f"{k}_SRCD.wav", "-o", od / f"{k}_CSRCD.wav"])
    with Pool(8) as pl:
        list(pl.imap_unordered(job, jobs))
    print("written", len(jobs), "delta logamp range example", round(float(dl.min()), 2), round(float(dl.max()), 2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
