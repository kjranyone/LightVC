"""gate_ff_v2b の最終の組 devff23x(results/conv_p0/gate_ff_v2b.yaml): devff2(30)と devff3(12)の操作者 42 人の入力のクリップ(8 s)・操作者の参照・登録をそのまま使い、
dev の目標 30 人へ 1 人 3 目標(Random(4951) の巡回)= 126 組。目標の登録・参照は dev のもの。操作者がどの学習の台帳にも無いことを確かめる。
    uv run python d495_ffset23x.py --s <dev の親> --out <dev の親>/devff23x
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=3)
    a = ap.parse_args()
    import train_ppgvoc as PP
    S, F = Path(a.s), Path(a.out)
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (F / d).mkdir(parents=True, exist_ok=True)
    dev = S / "dev"
    for f in sorted((dev / "d37/_work").glob("ref_*.wav")):
        if not (F / "d37/_work" / f.name).exists():
            shutil.copy(f, F / "d37/_work" / f.name)
    for e in sorted((dev / "d85_work").glob("en_*")):
        if not (F / "d85_work" / e.name).exists():
            shutil.copytree(e, F / "d85_work" / e.name)
    targets = json.loads((dev / "devset.json").read_text())["targets"]
    trained = {r["spk"] for r in json.loads(PP.MANIFEST.read_text())["rows"]} | {r["spk"] for r in json.loads((ROOT / "data/rvoc_f0hi/manifest.json").read_text())["rows"]}
    tcycle = list(range(len(targets)))
    random.Random(4951).shuffle(tcycle)
    tq = iter(tcycle * 10)
    log, pi = [], 0
    for sname in ("devff2", "devff3"):
        Ss = S / sname
        for r in json.loads((Ss / "devset.json").read_text())["pairs"]:
            assert r["spk"] not in trained, ("学習の台帳にある操作者", r["spk"])
            k0 = f"p{r['pi']:04d}_t{r['ti']:03d}"
            src = Ss / "d56" / f"{k0}_SRC.wav"
            if not src.exists():
                continue
            for _ in range(a.per):
                ti = next(tq)
                k = f"p{pi:04d}_t{ti:03d}"
                dst = F / "d56" / f"{k}_SRC.wav"
                if not dst.exists():
                    dst.symlink_to(src.resolve())
                ow = F / "d56/_work" / f"op_{pi:04d}.wav"
                if not ow.exists():
                    ow.symlink_to((Ss / "d56/_work" / f"op_{r['pi']:04d}.wav").resolve())
                od = F / "d85_work" / f"op_{pi:04d}"
                if not od.exists():
                    shutil.copytree(Ss / "d85_work" / f"op_{r['pi']:04d}", od)
                log.append({"pi": pi, "ti": ti, "kind": "FF", "spk": r["spk"], "from": f"{sname}/{k0}", "target": targets[ti]})
                pi += 1
    (F / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "female_to_female"}, ensure_ascii=False, indent=1))
    print("pairs", len(log), "operators", len({r["spk"] for r in log}), "targets", len({r["ti"] for r in log}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
