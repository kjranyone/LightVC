import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_ct as CT
import snv_p0_front as P

SR = P.SR


def envelope(x: np.ndarray, f0: np.ndarray, mode: str = "hann3", f_sw: float = 1e9, cap_ms: float = 1e12) -> np.ndarray:
    return P.envelope_from_amp(CT.amp(x, f0, mode, f_sw, cap_ms))


def _selftest() -> None:
    import librosa
    import torch

    x = librosa.load(str(P.ROOT / "data/snv_e1/em480/t03.wav"), sr=SR, mono=True)[0].astype(np.float32)
    x = x[: P.n_frames(len(x)) * P.H]
    f0 = P.f0_track(torch.from_numpy(x))[0].numpy().astype(np.float64)
    a, b = P.envelope(x, f0), envelope(x, f0)
    d = np.abs(a - b)
    print("pyworld との差(c0..c24): 平均", round(float(d.mean()), 5), "最大", round(float(d.max()), 4), "c0 の標準偏差", round(float(a[0].std()), 3))
    assert d.mean() < 0.01 and d.max() < 0.5, (d.mean(), d.max())
    s = envelope(x, f0, "sine2")
    print("sine2 と hann3 の差の平均", round(float(np.abs(s - a).mean()), 4))
    print("p100_ct selftest OK")


if __name__ == "__main__":
    _selftest()
