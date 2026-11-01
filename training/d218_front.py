"""D218 の前計算(results/conv_p0/d218_prereg.yaml): 実女声の開発用の組の源と操作者の参照の LIGHTVC_DUMP_FRONT(製品と同じ引数・条件だけ)と、
候補外 8 人(d208 の anchors)の Rust の登録(前計算 d217 の data/realx/pbase/ref/<A>.wav)。
出力 = <s>/d214/{pNNNN_tNNN,opNNNN}_FRONT.f32・--en_out/<A>/。
    uv run python d218_front.py --s <dev> <dev2> <dev3> --en_out <dir> --workers 3
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

from d208_rr_natural import anchors
from d217_pbase_prep import ART, BIN, run

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", nargs="+", required=True)
    ap.add_argument("--en_out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    EN = Path(a.en_out)
    jobs = []
    for A in anchors(8):
        jobs.append(([BIN, "artic-enroll", "-r", ROOT / "data/realx/pbase/ref" / f"{A['target']}.wav", "-o", EN / A["target"]], {}, str(EN / A["target"] / "enroll.json")))
    tmp = []
    for s in a.s:
        S = Path(s)
        rd, work = S / "d214", S / "d85_work"
        seen: set = set()
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            ins = [(f, rd / f"{k}_FRONT")] + ([] if pi in seen else [(S / "d56/_work" / f"op_{pi:04d}.wav", rd / f"op{pi:04d}_FRONT")])
            seen.add(pi)
            for i, o in ins:
                tmp.append(o.with_suffix(".tmp.wav"))
                jobs.append(([BIN, "artic", "--enroll", work / f"en_{ti:03d}", "--operator-enroll", work / f"op_{pi:04d}", *ART, "-i", i, "-o", tmp[-1]],
                             {"LIGHTVC_COND_ONLY": "1", "LIGHTVC_DUMP_FRONT": str(o.with_suffix(".f32"))}, str(o.with_suffix(".f32"))))
    with Pool(a.workers) as pool:
        bad = [r for r in pool.imap_unordered(run, jobs) if r.startswith("FAIL")]
    for t in tmp:
        t.unlink(missing_ok=True)
    print("jobs", len(jobs), "fail", bad[:3], flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
