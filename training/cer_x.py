"""R-X の fooling canary: 変換音の CER（source 男声の書き起こしを参照）。

    uv run python cer_x.py --arms cartA=../results/diag_cartA/conv cipt=../results/diag_cipt/conv
各 arm ディレクトリは eval_x --dump の出力（<spk>_<stem>.wav）。
source は data/male_feat の対応 wav。whisper-base、言語は gt から自動検出を共通適用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_vc_g import wav_path_of
from eval_cer import cer

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=dir 形式")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arms)

    import whisper
    model = whisper.load_model("base")

    first = sorted(Path(next(iter(arms.values()))).glob("*.wav"))
    if not first:
        sys.exit("変換 wav が無い")
    stems = [p.stem for p in first]

    def src_wav(stem: str) -> Path:
        spk, ut = stem.split("_", 2)[0] + "_" + stem.split("_", 2)[1], stem.split("_", 2)[2]
        f = ROOT / "data/male_feat" / spk / f"{ut}.pt"
        d = torch.load(f, map_location="cpu", weights_only=False)
        return wav_path_of(d)

    refs, lang = {}, None
    for st in stems:
        r = model.transcribe(str(src_wav(st)), fp16=False, temperature=0.0)
        if lang is None:
            lang = r["language"]
            print(f"  言語検出: {lang}")
        refs[st] = r["text"]

    out = {}
    for name, d in arms.items():
        cs = []
        for st in stems:
            p = Path(d) / f"{st}.wav"
            if not p.exists():
                print(f"  ⚠ {name}: {p.name} が無い")
                continue
            hyp = model.transcribe(str(p), language=lang, fp16=False, temperature=0.0)["text"]
            c = cer(refs[st], hyp)
            cs.append(c)
            print(f"  {name} {st}: CER {c:.3f}", flush=True)
        out[name] = sum(cs) / max(len(cs), 1)
    print()
    for name, v in out.items():
        print(f"  {name}: mean CER {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
