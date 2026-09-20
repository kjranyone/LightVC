"""Sweep module-level knobs of rddsp through the full resynthesize() path.

Usage: uv run python rddsp_sweep.py FLOOR_GAIN 1.0 0.7 0.5 0.25 0.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
from rddsp_loop import Cache


def cast(v: str):
    if v in ("True", "False"):
        return v == "True"
    if v == "None":
        return None
    try:
        return int(v) if "." not in v and "e" not in v else float(v)
    except ValueError:
        return v


def main() -> None:
    name = sys.argv[1]
    vals = [cast(v) for v in sys.argv[2:]]
    c = Cache()
    orig = getattr(R, name)
    base = c.score()
    print(f"baseline ({name}={orig!r}) {base:.4f}")
    for v in vals:
        setattr(R, name, v)
        try:
            s = c.score()
        except Exception as e:  # noqa: BLE001
            print(f"  {name}={v!r:>10} FAILED {type(e).__name__}: {e}")
            continue
        print(f"  {name}={v!r:>10} {s:.4f}  {s-base:+.4f}")
    setattr(R, name, orig)


if __name__ == "__main__":
    main()
