"""検査そのものを検査する（**手順書テキスト**が対象）。

⚠ 関門コード（`latency_gate.py` / `pre_launch_check.py`）への変異は
   `gate_mutate.py` が fixture で見る。**ここには置かない**——
   33 巡目にコードを移したあと変異定義が本文を指したままで 6 件が SKIP になり、
   「3 つとも通す」と宣言しながら 1 つが赤いままだった。

**壊しても FAIL しない検査は存在しない検査。**

27 巡目のレビューが変異テストで実証した: 26 巡目に直した 7 件のうち
**6 件に自動検査が 1 つも掛かっていなかった**——明日同じ形で戻しても誰も気づかない。
「直した」と「再発しないようにした」は別物で、後者だけが手順書の価値を上げる。

    uv run python mutate_check.py          # 全変異を試す
    uv run python mutate_check.py --add    # 新しい変異の書き方を表示

各変異は (対象ファイル, 置換前, 置換後, 何の再発を防ぐか) の 4 つ組。
**FAIL しなかった変異＝守られていない修正**として一覧に出す。
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "current/PROCEDURE.md"
APP = ROOT / "current/PROCEDURE_APP.md"

# (file, old, new, 守るもの[, all=True で全置換])
MUTATIONS = [
    (PROC, "| **`mlin`** | **合成（`HOP_S`）** | **`[..., 7:]`**",
     "| **`mlin`** | **合成（`HOP_S`）** | **`[..., 3:]`**",
     "2.3 (0): mlin は合成格子なので除外 7"),
    (PROC, 'rm -f "$PIN"; }', 'rm -f "$PIN" "$SHA"; }',
     "2.1: ピンのキー欠落で sha まで消さない"),
    (PROC, "&& [ -s ../results/$TAG/melframe.txt ] && [ -s ../results/$TAG/melnfft.txt ] \\",
     "\\",
     "4.6: MELFRAME/MELNFFT をガードに入れる"),
    # ---- 22〜26 巡の重大を遡って載せる（退行が毎巡の欠陥の約 4 割を占めるため）
    (PROC, "**⚠ net RTF は全て未再現**", "**net RTF は実測済み**",
     "2.4b: net RTF が未再現である開示", True),
    (PROC, "INCONCLUSIVE を PASS にしない", "INCONCLUSIVE も PASS にする",
     "2.4b-1a: 余裕 < ぶれ幅なら PASS にしない"),
    (PROC, "| 7 | **front-end の頭を零詰めにする**", "| 7 | **front-end の頭は reflect のまま**",
     "2.4a-2 状態 7: front-end の頭を零詰めにする"),
    (PROC, "| 8 | **起動直後は実サンプルだけを解析する**", "| 8 | **起動直後もゼロを含めて解析する**",
     "2.4a-2 状態 8: 起動直後は実サンプルのみ解析"),
    (PROC, "**未来不変性 ネット込み**", "**未来不変性（ネットは見ない）**",
     "5.1: ckpt を通した未来不変性の関門"),
    (PROC, "set -o pipefail   # ⚠ tee で終了コードが潰れる（ship_check の FAIL / ImportError を見落とす）",
     "# pipefail 削除",
     "5.5-C: tee で ship_check の FAIL が潰れない"),
    (PROC, "共通の**減衰係数", "系ごとの**減衰係数",
     "1.2: 盲検の音量整合（試行内共通係数）"),
    (PROC, "- **隣り合う試行では X/Y/Z の割り当てが必ず違う。**", "- **試行ごとに割り当てが違う。**",
     "1.2: 盲検の割り当ては隣接同一を棄却"),
]


def _from_invariants() -> list[tuple]:
    """`doc_check.GATE_INVARIANTS` の各行から変異を自動生成する。

    ⚠ 不変条件を足しただけでは「発火するか」は分からない。
    needle を壊す変異を必ず 1 本作り、FAIL することを確かめて初めて守られたと言える。
    28 巡目に 51 件足したが、そのうち発火を確認したのは手書きの 17 件だけだった。
    """
    sys.path.insert(0, str(ROOT / "training"))
    import doc_check as DC
    out = []
    for name, needle, why in DC.GATE_INVARIANTS:
        f = PROC if name == "PROCEDURE.md" else APP
        src = f.read_text()
        if needle not in src:
            continue
        out.append((f, needle, "＠＠壊した＠＠", f"[inv] {why}", src.count(needle) > 1))
    return out


JOURNAL = ROOT / "results/z0/mutate_journal.json"


def run_checks() -> tuple[int, str]:
    out = []
    rc = 0
    for cmd in (["uv", "run", "python", "doc_check.py"],
                ["uv", "run", "python", "gate_run.py"]):
        import os
        env = dict(os.environ, DOC_HARNESS_LOCK_HELD="1")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                           cwd=ROOT / "training", timeout=1800)
        out.append(r.stdout[-400:])
        rc = rc or r.returncode
    return rc, "\n".join(out)


_LOCK_FH = None


def acquire_lock() -> None:
    """リポジトリ全体の排他ロック。**`results/` の外**に置く。

    29 巡目に同時編集の衝突が 2 回起き、**レビューが他人の変異状態を
    「文書の欠陥」として測った**。`results/z0/.mutate.lock` は
    `gate_run --write` の `rmtree` で消えるので位置も誤りだった。
    """
    global _LOCK_FH
    import fcntl
    import os
    if os.environ.get("DOC_HARNESS_LOCK_HELD") == "1":
        return                      # 親（mutate_check）が既に保持している
    f = ROOT / ".doc_harness.lock"
    _LOCK_FH = f.open("w")
    try:
        fcntl.flock(_LOCK_FH, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("  ⚠ 別の検査／変異が走行中。同時に走らせると"
                         "他人の変異状態を欠陥として測る（29 巡目に 2 回発生）")


def main() -> int:
    acquire_lock()
    # ⚠ 前回が SIGKILL で死んでいたら、ここで文書を回収してから始める。
    if JOURNAL.exists():
        j = json.loads(JOURNAL.read_text())
        Path(j["file"]).write_text(j["orig"])
        JOURNAL.unlink()
        print(f"  ⚠ 前回の走行が中断されていた。{Path(j['file']).name} を復元した")
    if "--add" in sys.argv:
        print(__doc__)
        return 0
    # ⚠ 排他ロック。28 巡目のレビュー中に別インスタンスと同時走行し、
    #   **文書が変異状態のままディスクに残った**（しかも当時の doc_check は FAIL 0 を返した）。
    base_rc, _ = run_checks()
    if base_rc != 0:
        print("  ⚠ 変異前に既に FAIL している。先にそちらを直す")
        return 1
    unguarded = []
    if "--inv" in sys.argv:
        print("  ⚠ --inv は廃止した。needle を削除する変異は check_gate_invariants の")
        print("     needle not in text が必ず真になる**恒真**で、情報量ゼロだった")
        print("     （29 巡目に 0/68 を『全件発火』と誤報告した）。")
        print("     意味を反転する変異（数値を動かす／かつ→または／>→>=）を手で書く。")
        return 1
    muts = MUTATIONS
    for mut in muts:
        f, old, new, what = mut[:4]
        allrep = len(mut) > 4 and mut[4]
        src = f.read_text()
        if not allrep and src.count(old) != 1:
            print(f"  SKIP  {what}\n        （対象が {src.count(old)} 箇所。"
                  f"一意にするか 5 つ目に True を足して全置換にする）")
            unguarded.append(what + "（変異の定義が古い）")
            continue
        before = hashlib.sha256(src.encode()).hexdigest()
        # ⚠ **SIGKILL 対策**（36 巡目の事故）。`finally` は SIGKILL では走らない。
        #    2 分のタイムアウトで殺された本スクリプトが 5.1 の行を反転したまま
        #    ディスクに残し、**その状態を `pins --bless` が正本に焼き込んだ**。
        #    ∴ 変異の前に「元本」を journal に落とし、起動時に必ず回収する。
        JOURNAL.write_text(json.dumps({"file": str(f), "orig": src},
                                      ensure_ascii=False))
        f.write_text(src.replace(old, new) if allrep else src.replace(old, new, 1))
        try:
            rc, _ = run_checks()
        finally:
            f.write_text(src)
            # ⚠ 復元できたことを確かめる。確かめないと「変異したまま残る」を検出できない。
            if hashlib.sha256(f.read_text().encode()).hexdigest() != before:
                sys.exit(f"復元に失敗した: {f}（手で git 相当の復元が要る）")
            JOURNAL.unlink(missing_ok=True)
        if rc == 0:
            print(f"  UNGUARDED  {what}")
            unguarded.append(what)
        else:
            print(f"  guarded    {what}")
    print(f"\n  守られていない修正: {len(unguarded)} / {len(muts)}")
    for u in unguarded:
        print(f"    - {u}")
    print("  ⚠ UNGUARDED は「直したが再発を止めていない」。検査を足すまで直したと言わない。")
    return 1 if unguarded else 0


if __name__ == "__main__":
    raise SystemExit(main())
