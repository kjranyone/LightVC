"""D380: 入力の mp3 の圧縮が変換を悪くするかの切り分け(外郎売.mp3 = 78 kbps の局所の遠さ・D378 の原因の候補)。
dev(TTS の男声 → 女声 60 組)の入力 d56/p00*_SRC.wav を mp3(libmp3lame・mono・--kbps)に圧縮して戻した組 dev_mp3 を作る(目標・操作者の登録・devset は dev と同じ)。
    uv run python d380_mp3set.py --s <dev の親> --kbps 80
    uv run python d274_evaldump.py --s <dev の親> --set dev_mp3 --out <dev の親>/e2e
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--kbps", type=int, default=80)
    ap.add_argument("--name", default="dev_mp3")
    a = ap.parse_args()
    S = Path(a.s)
    src, dst = S / "dev", S / a.name
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "d56").mkdir(parents=True)
    for d in ("d37", "d85_work"):
        (dst / d).symlink_to(src / d)
    shutil.copy(src / "devset.json", dst / "devset.json")
    shutil.copy(src / "excl.txt", dst / "excl.txt")
    n = 0
    for f in sorted((src / "d56").glob("p00*_SRC.wav")):
        x, sr = sf.read(f, dtype="float32")
        enc = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-c:a", "libmp3lame", "-b:a", f"{a.kbps}k", "-f", "mp3", "-"],
                             input=x.astype("<f4").tobytes(), capture_output=True, check=True).stdout
        dec = subprocess.run(["ffmpeg", "-v", "error", "-f", "mp3", "-i", "-", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], input=enc, capture_output=True, check=True).stdout
        y = np.frombuffer(dec, "<f4")
        m = min(len(x), len(y))
        sf.write(dst / "d56" / f.name, y[:len(x)] if len(y) >= len(x) else np.r_[y, np.zeros(len(x) - len(y), np.float32)], sr)
        n += 1
    print("encoded", n, "files at", a.kbps, "kbps →", dst, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
