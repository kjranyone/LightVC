"""D265 の組(results/conv_p0/d265_prereg.yaml): 目標 = dev の 30 人・操作者 = 女声 TTS v3(data/female_tts_corpus・60 人・p0000〜)と男声 TTS v3(data/male_tts_corpus・47 人を文を替えて 60 組・p0100〜)。平静の読みだけ・元の声 8 秒・操作者の参照 = 残りの平静の読み(20 秒以上)。
    uv run python d265_ttsset.py --s <dev の親> --out <dev の親>/devtts
"""
from __future__ import annotations

import argparse
import json
import random
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
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    log = []
    rng = random.Random(265)
    fsp = sorted(d.name for d in (ROOT / "data/female_tts_corpus").iterdir() if d.is_dir())
    rng.shuffle(fsp)
    msp = sorted(d.name for d in (ROOT / "data/male_tts_corpus").iterdir() if d.is_dir())
    for kind, base, spks, p0 in (("FT", ROOT / "data/female_tts_corpus", fsp, 0), ("MT", ROOT / "data/male_tts_corpus", msp, 100)):
        for j in range(60):
            ti = j // 2
            spk = spks[j % len(spks)]
            ws = sorted((base / spk).glob("t*_neutral.wav"))
            off = 0 if j < len(spks) else 2
            ws = ws[off:] + ws[:off]
            src, tl, rest = [], 0.0, []
            for w in ws:
                if tl < 8.0:
                    src.append(w); tl += sf.info(str(w)).duration
                else:
                    rest.append(w)
            rl = sum(sf.info(str(w)).duration for w in rest)
            if tl < 8.0 or rl < 20.0:
                print("skip", kind, spk, round(tl, 1), round(rl, 1), flush=True)
                continue
            pi = p0 + j
            sf.write(O / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", np.concatenate([load48(w) for w in src])[: 8 * 48000].astype(np.float32), 48000)
            opw = O / "d56/_work" / f"op_{pi:04d}.wav"
            sf.write(opw, np.concatenate([load48(w) for w in rest]).astype(np.float32), 48000)
            opd = O / "d85_work" / f"op_{pi:04d}"
            if not (opd / "enroll.json").exists():
                run([BIN, "artic-enroll", "-r", opw, "-o", opd])
            log.append({"pi": pi, "ti": ti, "kind": kind, "spk": spk, "src": [w.name for w in src], "target": targets[ti]})
    (O / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "tts_to_female"}, ensure_ascii=False, indent=1))
    print("FT", sum(r["kind"] == "FT" for r in log), "MT", sum(r["kind"] == "MT" for r in log), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
