"""D376: オーナー提供の実男声の朗読(外郎売.mp3・1 人・4 分 18 秒・48 kHz の mp3 78 kbps)から実男声 → 女声の評価の組 devreal を作る(評価専用・学習に使わない)。
mono 48 kHz に直し、−35 dB(最大比)より静かな 0.25 s 以上の区間で区切り、6〜10 s の発話にまとめる。先頭から 22 s 以上を操作者の登録の参照(Rust artic-enroll)・残りを評価の入力に。
目標 = dev の 30 人(各 2 本 = 60 組・同じ目標に同じ入力が重ならないよう順に割り当て)= TTS の男声入力の dev の指標とそのまま比べられる。dev の参照・中心・登録を写す(d326_vctkset.py と同じ並び)。
    uv run python d376_realset.py --mp3 ../外郎売.mp3 --s <dev の親> --out <dev の親>/devreal
    uv run python d274_evaldump.py --s <dev の親> --set devreal --out <dev の親>/e2e
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN

ROOT = Path(__file__).resolve().parent.parent


def segments(x: np.ndarray, sr: int = 48000) -> list:
    hop = sr // 100
    n = len(x) // hop
    e = 10 * np.log10(np.array([np.mean(x[i * hop:(i + 1) * hop] ** 2) for i in range(n)]) + 1e-12)
    loud = e > e.max() - 35.0
    cuts, run = [], 0
    for i in range(n):
        run = run + 1 if not loud[i] else 0
        if run == 25:
            cuts.append(i - 12)
    bounds = [0] + cuts + [n]
    segs = []
    for s, t in zip(bounds[:-1], bounds[1:]):
        idx = np.flatnonzero(loud[s:t])
        if len(idx):
            segs.append((s + idx[0], s + idx[-1] + 1))
    return [(a * hop, b * hop) for a, b in segs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3", required=True)
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_sec", type=float, default=22.0)
    ap.add_argument("--min_sec", type=float, default=6.0)
    ap.add_argument("--max_sec", type=float, default=10.0)
    a = ap.parse_args()
    S, O = Path(a.s), Path(a.out)
    dev = S / "dev"
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (O / d).mkdir(parents=True, exist_ok=True)
    for f in sorted((dev / "d37").glob("cen_*.wav")):
        shutil.copy(f, O / "d37" / f.name)
    for f in sorted((dev / "d37/_work").glob("ref_*.wav")):
        shutil.copy(f, O / "d37/_work" / f.name)
    for e in sorted((dev / "d85_work").glob("en_*")):
        if not (O / "d85_work" / e.name).exists():
            shutil.copytree(e, O / "d85_work" / e.name)
    shutil.copy(dev / "excl.txt", O / "excl.txt")
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", a.mp3, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, "<f4").astype(np.float32)
    sf.write(O / "d56/_work" / "uiro_full.wav", x, 48000)
    segs = segments(x)
    pad = int(0.1 * 48000)
    pieces = [x[max(0, s - pad):min(len(x), t + pad)] for s, t in segs]
    ref, tl, i = [], 0.0, 0
    while tl < a.ref_sec and i < len(pieces):
        ref.append(pieces[i])
        tl += len(pieces[i]) / 48000
        i += 1
    chunks, cur = [], []
    for p in pieces[i:]:
        cur.append(p)
        dur = sum(len(q) for q in cur) / 48000
        if dur >= a.min_sec:
            chunks.append(np.concatenate(cur)[: int(a.max_sec * 48000)])
            cur = []
    print("pieces", len(pieces), "ref", round(tl, 1), "s from", i, "pieces | eval chunks", len(chunks), "lengths", [round(len(c) / 48000, 1) for c in chunks], flush=True)
    refw = O / "d56/_work" / "ref_uiro.wav"
    sf.write(refw, np.concatenate(ref).astype(np.float32), 48000)
    spd = O / "d85_work" / "spk_uiro"
    if not (spd / "enroll.json").exists():
        subprocess.run([str(BIN), "artic-enroll", "-r", str(refw), "-o", str(spd)], cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, check=True, capture_output=True)
    log = []
    for j in range(60):
        ti = j // 2
        ch = (j * 7) % len(chunks)
        pi = j
        sf.write(O / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", chunks[ch].astype(np.float32), 48000)
        opd = O / "d85_work" / f"op_{pi:04d}"
        if not opd.exists():
            opd.symlink_to(spd.name)
        log.append({"pi": pi, "ti": ti, "chunk": ch, "spk": "uiro", "target": targets[ti]})
    (O / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "real_male_ja_uiro", "source": "外郎売.mp3(オーナー提供・評価専用)"}, ensure_ascii=False, indent=1))
    print("pairs", len(log), "distinct chunks", len({r["chunk"] for r in log}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
