import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, str(Path(__file__).parent))
from p97_probe import IN, ROOT, SPS

PR = ROOT / "results/snv_p98/probe"
SR = 48000
BANDS = [(2, 8), (8, 16), (16, 32), (32, 64), (64, 120)]
LP = butter(4, 150.0, "lowpass", fs=SR, output="sos")


def modspec(x: np.ndarray) -> np.ndarray:
    e = np.sqrt(np.maximum(sosfiltfilt(LP, x.astype(np.float64) ** 2), 1e-12))[::48]
    l = 20 * np.log10(e + 1e-6)
    act = l > l.max() - 35
    seg, out = 500, []
    for i in range(0, len(l) - seg, seg // 2):
        if act[i : i + seg].mean() > 0.9:
            f, p = welch(l[i : i + seg] - l[i : i + seg].mean(), fs=1000, nperseg=250)
            out.append([p[(f >= a) & (f < b)].sum() * (f[1] - f[0]) for a, b in BANDS])
    return np.sqrt(np.mean(out, 0)) if out else np.full(len(BANDS), np.nan)


def main() -> None:
    names = sys.argv[1].split(",")
    rd = lambda p: np.fromfile(p, np.float32)
    print("帯域(Hz)の対数の輪郭の変調の RMS(dB)", BANDS)
    print("目標の声(Irodori)", np.round(np.nanmean([modspec(rd(p)) for p in sorted((IN / "tg").glob("*.f32"))], 0), 2))
    print("源(本物の男声)", np.round(np.nanmean([modspec(rd(IN / s / "in.f32")) for s in SPS], 0), 2))
    for n in names:
        print(n, np.round(np.nanmean([modspec(sf.read(PR / n / f"{s}.wav", dtype="float32")[0]) for s in SPS], 0), 2))


if __name__ == "__main__":
    main()
