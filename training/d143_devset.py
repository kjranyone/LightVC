"""D143: 開発用の組(調整専用)を評価の組(D37 / D56)と同じ作り方・同じ置き場所の構成で別の場所 <dev> に作る。D136〜D142 で構成の選択を評価の 60 組で繰り返した
(合わせ込みの危険)ので、以後の調整は開発用の組だけで行い、評価の組は最終判定だけに使う(オーナー 10-13『まだ出さない』)。
  目標 = 評価プール(diag_pool.load_pool(150, 3))の 31〜60 人目の実女声(評価の組は 1〜30 人目)・参照 = female-dataset の先頭から 20s 以上(D37 と同じ)・cen = プールの中心
  元の声 = data/male_tts_v4l の同じ 30 人(m000〜m029・目標ごとに 2 人)・評価の組(D56)で使った文と錨 t10 を除いた文から 8s・操作者の参照 = 残りから 22s 以上
置き場所(評価の組と同じ相対構成): <dev>/d37/cen_TTT.wav・<dev>/d37/_work/ref_TTT.wav・<dev>/d56/pNNNN_tNNN_SRC.wav・<dev>/d56/_work/op_PPPP.wav・<dev>/d85_work/{en_TTT, op_PPPP}(Rust artic-enroll)
= d114 / d141 / d109 / d23 / mkcen / d4b を --s <dev> でそのまま使える。
    uv run python d143_devset.py --s <scratchpad> --dev <scratchpad>/dev
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
CORPUS = ROOT / "data/male_tts_v4l"


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--n_tg", type=int, default=30)
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--offset", type=int, default=30, help="プールの何人目から(評価の組 = 0〜29・開発用 = 30〜59・追加の開発用 dev2 = 60〜89)")
    ap.add_argument("--excl", nargs="*", default=[], help="既存の開発用の組 <dev>: その d56/_work/src_*.json の文と目標も除く")
    a = ap.parse_args()
    import diag_pool as P
    S, D = Path(a.s), Path(a.dev)
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (D / d).mkdir(parents=True, exist_ok=True)
    for f in ("excl.txt", "d109_cache.json"):
        if (S / f).exists() and not (D / f).exists():
            shutil.copy(S / f, D / f)
    pool = P.load_pool(150, 3, "cpu")
    man = json.loads((CORPUS / "manifest.json").read_text())
    spks = sorted(s for s, v in man["speakers"].items() if len(v["files"]) >= 30)
    used_eval: dict = {}
    used_tg = set()
    for wd in [S] + [Path(x) for x in a.excl]:
        for j in sorted((wd / "d56/_work").glob("src_*.json")):
            z = json.loads(j.read_text())
            used_eval.setdefault(z["spk"], set()).update(z["used"])
            used_tg.add(z.get("target", ""))
    rng = random.Random(143)
    pi = 0
    log = []
    for ti in range(a.n_tg):
        t = pool[a.offset + ti]
        ws = sorted((ROOT / "female-dataset" / t["spk"]).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        refp = D / "d37/_work" / f"ref_{ti:03d}.wav"
        if not refp.exists():
            sf.write(refp, np.concatenate([load48(w) for w in ref]).astype(np.float32), 48000)
        sf.write(D / "d37" / f"cen_{ti:03d}.wav", t["cen"], 48000)
        en = D / "d85_work" / f"en_{ti:03d}"
        if not (en / "enroll.json").exists():
            run([BIN, "artic-enroll", "-r", refp, "-o", en])
        for j in range(a.per):
            spk = spks[(ti * a.per + j) % len(spks)]
            files = sorted(k for k in man["speakers"][spk]["files"] if k != "t10" and k not in used_eval.get(spk, set()))
            rng.shuffle(files)
            used, xs, tl = [], [], 0.0
            for k in files:
                if tl >= 8.0:
                    break
                x = load48(CORPUS / spk / f"{k}.wav"); xs.append(x); used.append(k); tl += len(x) / 48000
            sf.write(D / "d56" / f"p{pi:04d}_t{ti:03d}_SRC.wav", np.concatenate(xs)[: 8 * 48000].astype(np.float32), 48000)
            opw = D / "d56/_work" / f"op_{pi:04d}.wav"
            opd = D / "d85_work" / f"op_{pi:04d}"
            if not (opd / "enroll.json").exists():
                rest = [k for k in sorted(man["speakers"][spk]["files"]) if k not in used and k != "t10"]
                ys, tot = [], 0.0
                for k in rest:
                    if tot >= 22.0:
                        break
                    y = load48(CORPUS / spk / f"{k}.wav"); ys.append(y); tot += len(y) / 48000
                assert tot >= 20.0, f"{spk} の操作者の参照が足りない {tot:.1f}s"
                sf.write(opw, np.concatenate(ys).astype(np.float32), 48000)
                run([BIN, "artic-enroll", "-r", opw, "-o", opd])
            (D / "d56/_work" / f"src_{pi:04d}.json").write_text(json.dumps({"spk": spk, "used": used, "target": t["spk"]}, ensure_ascii=False))
            log.append({"pi": pi, "ti": ti, "spk": spk, "used": used, "target": t["spk"]})
            pi += 1
        print("target", ti, t["spk"], flush=True)
    eval_targets = [pool[i]["spk"] for i in range(30)]
    assert not ({r["target"] for r in log} & (set(eval_targets) | used_tg)), "開発用の目標が評価の目標・既存の開発用の目標と重なる"
    (D / "devset.json").write_text(json.dumps({"targets": [pool[a.offset + i]["spk"] for i in range(a.n_tg)], "pairs": log, "offset": a.offset, "excl": a.excl}, ensure_ascii=False, indent=1))
    print("done", pi, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
