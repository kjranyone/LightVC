"""Analytical ear for the mel-gen quality gap (validated 2026-07-20 against the
user's ear: ceiling good >> self rough). Reference PESQ-vs-gt reproduces that
ordering (ceiling ~3.5 >> self ~1.1, alignment-robust); SQUIM-MOS and STOI do
NOT (flat / inverted) and mel-L1 misses it too. Use this to drive the loop
without a listening session every step; the ear stays the final promotion gate.

Reads a render dir with <stem>_gt.wav (or _gtA) + <stem>_<arm>.wav and reports,
per arm: reference PESQ vs gt (primary), SI-SDR + PESQ from torchaudio SQUIM
(no-reference cross-check). Higher PESQ = closer to gt quality.
Usage: uv run python quality_gate.py <dir> [arm1 arm2 ...]
"""
from __future__ import annotations
import sys, glob
from pathlib import Path
import numpy as np
import librosa
from pesq import pesq
from scipy.signal import correlate


def align(ref, deg):
    n = min(len(ref), len(deg)); ref, deg = ref[:n], deg[:n]
    lag = int(np.argmax(np.abs(correlate(deg, ref, mode="full")))) - (len(ref) - 1)
    if lag > 0:
        deg = np.concatenate([deg[lag:], np.zeros(lag, np.float32)])
    elif lag < 0:
        deg = np.concatenate([np.zeros(-lag, np.float32), deg[:lag]])
    return ref, deg


def main():
    D = Path(sys.argv[1])
    gt_tag = "gtA" if glob.glob(str(D / "*_gtA.wav")) else "gt"
    stems = sorted({"_".join(Path(p).name.split("_")[:-1]) for p in glob.glob(str(D / f"*_{gt_tag}.wav"))})
    arms = sys.argv[2:] or sorted({Path(p).name.split("_")[-1][:-4] for p in glob.glob(str(D / "*.wav"))} - {gt_tag})
    import torch, torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    obj = SQUIM_OBJECTIVE.get_model().eval()

    def l16(p):
        w, _ = librosa.load(p, sr=16000); return w.astype(np.float32)

    print(f"quality_gate | {len(stems)} utts | ref={gt_tag} | PESQ-vs-gt = validated ear (ceiling>>self)")
    print(f"{'arm':12s} {'PESQ/gt':>8s} {'sq-PESQ':>8s} {'sq-SISDR':>9s}")
    for a in arms:
        pq, sp, ss = [], [], []
        for s in stems:
            rp, dp = D / f"{s}_{gt_tag}.wav", D / f"{s}_{a}.wav"
            if not dp.exists():
                continue
            ref, deg = align(l16(str(rp)), l16(str(dp)))
            try:
                pq.append(pesq(16000, ref, deg, "wb"))
            except Exception:
                pass
            with torch.no_grad():
                st, pe, si = obj(torch.from_numpy(deg).unsqueeze(0))
            sp.append(pe.item()); ss.append(si.item())
        m = lambda x: np.mean(x) if x else float("nan")
        print(f"{a:12s} {m(pq):8.3f} {m(sp):8.3f} {m(ss):9.2f}")


if __name__ == "__main__":
    main()
