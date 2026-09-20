"""過去のレビューが当てた変異を**固定集合**として回し、素通り率を出す。

34 巡目の発見: 素通り率は**変異集合を固定しない限り収束指標にならない**。
28→29→32→33→34 の 93% → 100% → 97.5% → 61.7% → 80.7% という推移のうち、
33→34 の上昇は退行ではなく「新しい穴を狙った」ことによる。
∴ 過去の変異を `results/z0/mutation_corpus.json` に貯め、**同じ集合で測る**。

    uv run python corpus_run.py              # 全件
    uv run python corpus_run.py --round 33   # その巡の分だけ
    uv run python corpus_run.py --limit 40   # 先頭 N 件（動作確認用）

**⚠ 素通り率が下がることだけが進捗**。新しいレビューが見つけた変異は
corpus に足す（`--append <json>`）ので、分母は単調増加する。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "results/z0/mutation_corpus.json"
CHECKS = (["uv", "run", "python", "doc_check.py"],
          ["uv", "run", "python", "gate_run.py"],
          ["uv", "run", "python", "gate_mutate.py"])


def _run_checks(env: dict) -> bool:
    """全部 rc=0 なら True（＝素通り）。1 つでも非零なら捕捉。"""
    for cmd in CHECKS:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=ROOT / "training", timeout=2400)
        if r.returncode:
            return False
    return True


def main() -> int:
    import os
    if not CORPUS.exists():
        print(f"  ⚠ {CORPUS} が無い")
        return 1
    muts = json.loads(CORPUS.read_text())
    if "--round" in sys.argv:
        rd = int(sys.argv[sys.argv.index("--round") + 1])
        muts = [m for m in muts if m.get("round") == rd]
    if "--limit" in sys.argv:
        muts = muts[:int(sys.argv[sys.argv.index("--limit") + 1])]

    env = dict(os.environ, DOC_HARNESS_LOCK_HELD="1")
    passed, applied, skipped = [], 0, 0
    for m in muts:
        # ⚠ 3 つのスキーマが混在する（巡ごとにレビュアーが別形式で書いた）。
        #    (file, old, new, count) 形式と ops=[["rep", file, old, new], …] 形式。
        ops = m.get("ops") or ([["rep", m["file"], m["old"], m["new"],
                                 int(m.get("count", 1))]] if "old" in m else [])
        if not ops:
            skipped += 1
            continue
        touched, ok = [], True
        for op in ops:
            if op[0] != "rep":
                ok = False
                break
            _, rel, old, new = op[:4]
            want = int(op[4]) if len(op) > 4 else 1
            f = pathlib.Path(rel) if rel.startswith("/") else ROOT / rel
            if not f.exists():
                ok = False
                break
            src = f.read_text()
            if src.count(old) != want:
                ok = False
                break
            touched.append((f, src))
        if not ok:
            skipped += 1
            for f, src in touched:
                f.write_text(src)
            continue
        for (f, src), op in zip(touched, ops):
            want = int(op[4]) if len(op) > 4 else 1
            f.write_text(src.replace(op[2], op[3], want))
        applied += 1
        try:
            if _run_checks(env):
                passed.append(m)
        finally:
            for f, src in touched:
                f.write_text(src)
    n = applied or 1
    print(f"\n  素通り {len(passed)} / 適用 {applied}"
          f"（{len(passed)/n*100:.1f}%）／ 適用できず {skipped}")
    from collections import Counter
    for c, k in Counter(m.get("cat", "?") for m in passed).most_common(8):
        print(f"    {c}: {k}")
    print("  ⚠ 「適用できず」は変異定義が現行版と合わないもの。"
          "**捕捉と数えない**（数えると文言を変えるだけで率が下がる）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
