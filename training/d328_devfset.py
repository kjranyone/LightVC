"""D328 の組 devf: dev と同じ目標・同じ組の番号 pi で、dev の男声 mNNN が読んだ文(used の順)を同じ TTS の女声 fNNN(data/female_tts_v4l_matched)が読む。
元の声 8 秒・操作者の参照 = 錨 t10 と used を除く名前順の残りから 22 s 以上。置き場所は dev と同じ構成(d274_evaldump・d280 がそのまま使える)。
    uv run python d328_devfset.py --s <dev の親> --out <dev の親>/devf
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
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
CORPUS = ROOT / "data/female_tts_v4l_matched"


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-400:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
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
    ds = json.loads((dev / "devset.json").read_text())
    man = json.loads((CORPUS / "manifest.json").read_text())
    log = []
    for r in ds["pairs"]:
        pi, ti, spk = r["pi"], r["ti"], "f" + r["spk"][1:]
        files = man["speakers"][spk]["files"]
        xs = [load48(CORPUS / spk / f"{k}.wav") for k in r["used"]]
        x = np.concatenate(xs)
        if len(x) < 8 * 48000:
            print("short", pi, spk, round(len(x) / 48000, 2), flush=True)
        sf.write(O / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", x[: 8 * 48000].astype(np.float32), 48000)
        opd = O / "d85_work" / f"op_{pi:04d}"
        if not (opd / "enroll.json").exists():
            rest = [k for k in sorted(files) if k not in r["used"] and k != "t10"]
            ys, tot = [], 0.0
            for k in rest:
                if tot >= 22.0:
                    break
                y = load48(CORPUS / spk / f"{k}.wav")
                ys.append(y)
                tot += len(y) / 48000
            assert tot >= 20.0, f"{spk} の操作者の参照が足りない {tot:.1f}s"
            opw = O / "d56/_work" / f"op_{pi:04d}.wav"
            sf.write(opw, np.concatenate(ys).astype(np.float32), 48000)
            run([BIN, "artic-enroll", "-r", opw, "-o", opd])
        log.append({"pi": pi, "ti": ti, "spk": spk, "kind": "F", "used": r["used"], "target": r["target"]})
    (O / "devset.json").write_text(json.dumps({"targets": ds["targets"], "pairs": log, "kind": "female_tts_v4l_matched"}, ensure_ascii=False, indent=1))
    print("pairs", len(log), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
