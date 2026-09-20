"""CER canary v2（2026-08-21 事前登録の新計器・diag_v2f_gmel 腕から適用）。

平均→**median**（whisper のループ幻覚 1 件で平均が任意に壊れる実測への対処）、
`condition_on_previous_text=False`（ループ幻覚の既知緩和）。

    uv run python cer_x2.py --arms old=<dir> new=<dir>
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_vc_g import wav_path_of

ROOT = Path(__file__).resolve().parent.parent


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip(), hyp.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[n] / m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arms)
    import whisper
    model = whisper.load_model("base")
    stems = [p.stem for p in sorted(Path(next(iter(arms.values()))).glob("*.wav"))]

    def src_wav(stem: str) -> Path:
        parts = stem.split("_", 2)
        f = ROOT / "data/male_feat" / f"{parts[0]}_{parts[1]}" / f"{parts[2]}.pt"
        d = torch.load(f, map_location="cpu", weights_only=False)
        return wav_path_of(d)

    refs, lang = {}, None
    for st in stems:
        r = model.transcribe(str(src_wav(st)), fp16=False, temperature=0.0,
                             condition_on_previous_text=False)
        lang = lang or r["language"]
        refs[st] = r["text"]

    for name, d in arms.items():
        cs = []
        for st in stems:
            p = Path(d) / f"{st}.wav"
            if not p.exists():
                continue
            hyp = model.transcribe(str(p), language=lang, fp16=False,
                                   temperature=0.0,
                                   condition_on_previous_text=False)["text"]
            c = cer(refs[st], hyp)
            cs.append(c)
            print(f"  {name} {st}: CER {c:.3f}", flush=True)
        print(f"  {name}: median {statistics.median(cs):.4f}  mean {sum(cs)/len(cs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
