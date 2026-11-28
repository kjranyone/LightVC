"""段 P107 step d(学習なし): 前段の高さの追跡を、きれいな合成の声(E1 の検証の声)でオフラインの pYIN と突き合わせる。
    cd training && uv run --no-sync python snv_diag/p107_track.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    males, females = [v for v in vs if v.startswith("em")][:6], [v for v in vs if v.startswith("ef")][:6]
    res: dict = {}
    for wy in (1200, 600):
        P.W_YIN = wy
        for grp, voices in (("male", males), ("female", females)):
            acc = {k: [] for k in ("v_own", "v_ref", "gross", "up", "down", "own_only", "ref_only", "n")}
            for v in voices:
                for t in ("t03", "t06", "t09"):
                    x = librosa.load(ROOT / f"data/snv_e1/{v}/{t}.wav", sr=P.SR, mono=True)[0].astype(np.float32)
                    x = x[: P.n_frames(len(x)) * P.H]
                    f = P.f0_track(__import__("torch").from_numpy(x))[0].numpy()
                    r, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=P.SR, target_sr=16000), fmin=60, fmax=900, sr=16000, frame_length=1024, hop_length=80, center=True)
                    n = min(len(f), len(r))
                    f, r, vf = f[:n], np.nan_to_num(r[:n]), vf[:n]
                    a, b = f > 0, vf & (r > 0)
                    both = a & b
                    d = np.log2(f[both] / r[both])
                    acc["v_own"].append(a.sum()); acc["v_ref"].append(b.sum()); acc["n"].append(n)
                    acc["gross"].append((np.abs(d) > 0.2).sum()); acc["up"].append((d > 0.5).sum()); acc["down"].append((d < -0.5).sum())
                    acc["own_only"].append((a & ~b).sum()); acc["ref_only"].append((~a & b).sum()); acc.setdefault("both", []).append(both.sum())
            s = {k: int(np.sum(v)) for k, v in acc.items()}
            o = {"voiced_own": round(s["v_own"] / s["n"], 3), "voiced_pyin": round(s["v_ref"] / s["n"], 3), "gross_0.2oct": round(s["gross"] / s["both"], 3), "up_0.5oct": round(s["up"] / s["both"], 3), "down_0.5oct": round(s["down"] / s["both"], 3), "own_only_share": round(s["own_only"] / s["n"], 3), "pyin_only_share": round(s["ref_only"] / s["n"], 3), "voices": voices}
            res[f"W{wy}_{grp}"] = o
            print(f"W_YIN {wy} {grp}", json.dumps(o), flush=True)
    (ROOT / "results/snv_p107/track_e1val.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
