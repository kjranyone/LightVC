"""Intelligibility gate for the front-end (E): Whisper CER of each arm vs gt.
Transcribe gt -> reference, transcribe arm -> hypothesis, CER = char edit / ref len.
Low added-CER (arm ~ gt) = intelligible; high = garbage (the 'ASR不能' the ear caught).
Usage: uv run python eval_cer.py <dir> <arm1> <arm2> ...  (gt auto-included as ref)
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import whisper

D = Path(sys.argv[1])
ARMS = sys.argv[2:] or ["ceiling", "e1none", "e1light", "e1heavy"]
stems = sorted({"_".join(Path(p).name.split("_")[:-1]) for p in glob.glob(str(D / "*_gt.wav"))})
model = whisper.load_model("base")


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


def tx(p: Path) -> str:
    if not p.exists():
        return ""
    return model.transcribe(str(p), language="ja", fp16=False)["text"]


print(f"Whisper CER vs gt-transcript (lower=more intelligible) | {len(stems)} utts")
refs = {s: tx(D / f"{s}_gt.wav") for s in stems}
for a in ARMS:
    cs = []
    for s in stems:
        h = tx(D / f"{s}_{a}.wav")
        if refs[s] or h:
            cs.append(cer(refs[s], h))
    if cs:
        print(f"  {a:10s} CER {sum(cs)/len(cs):.3f}  (n={len(cs)})")
# also print gt self-transcript sanity (a couple)
for s in stems[:2]:
    print(f"  [ref {s[:20]}] gt: {refs[s][:50]}")
