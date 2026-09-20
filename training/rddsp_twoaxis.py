"""Re-examine the DSP core's shipping knobs on BOTH axes.

Every knob in rddsp.py was set by maximising PESQ. 12.21'' shows PESQ is nearly
blind to the band-level error the listener named -- a 0.32 dB improvement scores
+0.001. So the shipping settings were chosen by a criterion that could not see
one of the two things that matter, and there is no reason to expect them to be
near-optimal on the other.

Sweep one knob at a time and report PESQ and BLE together. The rule from the
degeneracy work applies unchanged: the grid MUST contain the current value, or
a setting can pass by not being looked at.

Tuning is done on TRAINING speakers; the 12 held-out speakers are not touched
here, so anything this finds can still be confirmed honestly afterwards.

Every configuration is passed through the rate ledger (rddsp_bankcheck) before
its score is reported. A setting that raises the declared rate is not a quality
improvement, it is transfer -- 6.3 exists precisely so that this sweep cannot
walk into the FAIL region the way the PESQ-only sweep did.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import score_one
from rddsp_gpu import build
from rddsp_bandmetric import ble

NUTT = 8

KNOBS = {
    "FLOOR_GAIN": [0.85, 1.0, 1.15, 1.3],          # current 1.0
    "FLOOR_K":    [0.15, 0.25, 0.4, 0.6],          # current 0.25
    "MVF_SCALE":  [0.2, 0.3, 0.45, 0.6],           # current 0.3
    "MVF_FLOOR":  [1500.0, 2000.0, 3000.0, 4000.0],  # current 2000
    "NOISE_SMOOTH": [0, 80, 160, 320],             # current 160
}


def gate_ok() -> str:
    """The rate ledger, per candidate. check_current returns (ok, white_snr,
    harmonic_margin); only the first is a gate -- the white-noise numbers are
    diagnostic and 2.4 declares them inadmissible for the harmonic branch."""
    from rddsp_bankcheck import check_current
    ok, w, m = check_current()
    nb = R.NOISE_SMOOTH if R.NOISE_SMOOTH else (R.NOISE_NFFT // 2 + 1)
    rate = nb * (R.SR / (R.NOISE_NFFT // R.NOISE_HOPDIV)) / R.SR
    return f"{'PASS' if ok else 'FAIL'} (noise rate {rate:.2f}x)"


def measure(items) -> tuple[float, float]:
    ps, bs = [], []
    for it in items:
        gt = it["gt"]
        try:
            y, _, _, _ = R.resynthesize(gt)
        except Exception:
            return float("nan"), float("nan")
        n = min(gt.shape[-1], y.shape[-1])
        ps.append(score_one(gt[:n], y[:n]))
        bs.append(ble(y[:n], gt[:n]))
    return float(np.mean(ps)), float(np.mean(bs))


def main() -> None:
    tr, _ = build(80, 12)
    items = tr[:NUTT]
    base_p, base_b = measure(items)
    print(f"  tuning on {len(items)} TRAINING speakers (held-out set untouched)")
    print(f"  shipping settings: PESQ {base_p:.4f}   BLE {base_b:.3f} dB   "
          f"gate {gate_ok()}\n", flush=True)
    print(f"  {'knob':<14} {'value':>8} {'PESQ':>8} {'dPESQ':>8} {'BLE':>7} {'dBLE':>8}  gate")

    for knob, values in KNOBS.items():
        cur = getattr(R, knob)
        for v in values:
            setattr(R, knob, v)
            p, b = measure(items)
            mark = " <- current" if v == cur else ""
            print(f"  {knob:<14} {str(v):>8} {p:8.4f} {p-base_p:+8.4f} {b:7.3f} "
                  f"{b-base_b:+8.3f}  {gate_ok()}{mark}", flush=True)
        setattr(R, knob, cur)

    print("\n  Read the two deltas together. A row that gains PESQ and loses BLE is")
    print("  the trap the PESQ-only sweep walked into; a row that loses a little")
    print("  PESQ and gains a lot of BLE is what the ear asked for.")


if __name__ == "__main__":
    main()
