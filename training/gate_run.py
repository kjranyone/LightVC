"""手順書の「ゲートブロック」を実際に走らせて、期待どおり通る／落ちるかを見る。

`doc_check.check_bash_blocks` は `bash -n`（構文）しか見ない。直近 3 巡で
「前巡の修正そのものが新しい欠陥の最大の源」になった原因はここで、
**構文は通るが実行すると別の理由で落ちる**修正を繰り返し入れていた。実例:

  - ピンの sha 検査がパス不一致で必ず FAILED open or read（合格枝に到達不能）
  - `held_out_pick.json` の検査がキー不一致で mv に到達しない
  - 台帳の無効化が json.load の後ろで、ファイル欠損時に前回の pass:true が残る
  - `[:, 4:]` が f0（1 次元）に対して IndexError

いずれも 1 回走らせれば分かった。∴ 走らせる。

    uv run python gate_run.py            # 副作用の無いブロックだけ実行
    uv run python gate_run.py --list     # 何を実行し何を除外したかを表示

**⚠ 副作用のあるブロック（学習起動・書き込み・git・cargo）は実行しない。**
除外したことは必ず表示する（黙って減らすと「全部通った」に見える）。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 文書ごとの cwd 規約（各文書の冒頭に明文化されている）
DOCS = {ROOT / "current/PROCEDURE.md": ROOT / "training",
        ROOT / "current/PROCEDURE_APP.md": ROOT}

# ⚠ ブラックリストは原理的に閉じない。29 巡目にこれで実害が出た——
#    `git clone` も `cmake` も列に無かったので、ハーネスが**リポジトリ外に 77 MB を clone し
#    C++ をビルドした**（docstring は「git は実行しない」と宣言していた）。
#    ∴ **ホワイトリスト**にする: 許可した実行子だけを含むブロックしか走らせない。
ALLOWED = {
    "ls", "test", "[", "[[", "echo", "grep", "cat", "head", "tail", "sed", "awk",
    "wc", "cd", "export", "true", "false", "sha256sum", "printf", "sort", "uniq",
    "find", "basename", "dirname", "date", "read", "if", "then", "else", "elif",
    "fi", "for", "do", "done", "while", "case", "esac", "return", "exit", "set",
    "uv", "python3", "python", "nvidia-smi", "md5sum", "diff", "jq",
}
# 旧ブラックリスト（ホワイトリストを通っても、これを含むなら実行しない）
UNSAFE = (
    "nohup", "train_gvoc.py", "train_vc.py", "rm -", "rm-", "mv ", "mkdir",
    "torch.save", "write_text", "json.dump", "_j0.dump", "sha256sum >",
    "git add", "git commit", "git switch", "git checkout", "git stash",
    "cargo build", "cargo run", "cargo install", "signtool", "codesign",
    "render_z0c.py", "annot_gui.py", "listen_gui.py", "tee ",
    "pip ", "uv add", "dev.ps1", "xtask", "lightvc-app --", "-p lightvc-app",
    "sudo", "gui", "--features asio", "cargo test",
    "pkill", "kill ", "killall", "disown", "setsid",
    # ⚠ 成果物を書くものは実行しない。26 巡目にこれを漏らして rtf_front.py を 2 回走らせ、
    #    rtf_front_history.jsonl が伸びて plan_numbers の値が動いた（検査が自分で腐らせた）。
    "rtf_front.py", "prior_stream_ref.py", "plan_numbers.py --json", "z0_abi.py",
)
PLACEHOLDER = re.compile(r"<[^<>\s][^<>]{0,40}>")
# ⚠ リダイレクトだけを拾う。`assert x > 0` の比較演算子に当てない
#   （27 巡目: 規則 1 のゲート本体が「副作用: >」という**偽の理由**で除外されていた）。
REDIRECT = re.compile(r"(?<![0-9<>=!])>>?\s*[\w./$\"'~]")


def blocks(text: str):
    for m in re.finditer(r"```bash\n(.*?)```", text, re.S):
        body = "\n".join(l[2:] if l.startswith("> ") else l
                         for l in m.group(1).split("\n"))
        yield text[:m.start()].count("\n") + 1, body


# 書き込みはするが `results/` に閉じるもの。--write のときだけ実行する。
WRITES_RESULTS = ("mkdir", "torch.save", "write_text", "json.dump", "_j0.dump",
                  " > ", ">>", "mv ", "rm -", "sha256sum >", "z0_abi.py",
                  "rtf_front.py", "prior_stream_ref.py", "plan_numbers.py --json")


def _shell_only(body: str) -> str:
    """heredoc と `python -c "…"` の中身を落とし、**シェル行だけ**にする。

    27 巡目に lookbehind を足したが `) > 0`（`>` の直前が空白）を弾けず、
    **規則 1 のゲート本体が「副作用: リダイレクト」という偽の理由で除外され続けていた**。
    リダイレクトも副作用コマンドもシェル行にしか存在しないので、前処理で落とすのが正しい。
    `#` で始まる行も落とす（コメント中の `| tee` に当たっていた）。
    """
    b = re.sub(r"<<\s*'?(\w+)'?\n.*?^\1\s*$", "true", body, flags=re.S | re.M)
    b = re.sub(r'python3?\s+-c\s+"(?:[^"\\]|\\.)*"', "true", b, flags=re.S)
    b = re.sub(r'(?:uv run )?python3?\s+-\s*<<', "true <<", b)
    return "\n".join(l for l in b.split("\n") if not l.lstrip().startswith("#"))


def _executables(sh: str) -> set[str]:
    """シェル行の**先頭トークン**とパイプ／&&／; の直後を集める。"""
    out = set()
    for ln in sh.split("\n"):
        for seg in re.split(r"\||&&|\|\||;", ln):
            t = seg.strip().split()
            if t:
                out.add(t[0].strip("()$"))
    return {t for t in out if t and not t.startswith(("-", "\"", "'", "#", "$"))}


def classify(body: str, write: bool = False) -> str | None:
    sh = _shell_only(body)
    # ⚠ ホワイトリストを先に当てる。知らない実行子が 1 つでもあれば走らせない。
    unknown = _executables(sh) - ALLOWED
    if unknown:
        return f"未許可の実行子: {sorted(unknown)[:3]}"
    # ⚠ 副作用語は **body 全文** で見る。`python -c "…"` の中の json.dump を
    #    _shell_only が落としてしまい、−1.1c が毎回 latency_ledger.json を壊していた。
    for u in ("json.dump", "_j0.dump", "torch.save", "write_text", "open(", "mkdir"):
        if u in body and not write:
            return f"副作用（python 内）: {u}"
    # ⚠ パス文字列の出現ではなく**書き込み動作**で判定する。
    #    27 巡目までは `contract.toml` の読み取りが出るだけで −1.1c が除外されていた。
    if write and re.search(r"(?:git |cargo |&&\s*cd training)", sh):
        return "results/ の外を触る"
    for u in UNSAFE:
        if write and u in WRITES_RESULTS:
            continue
        if u in sh:
            ln = next((l.strip() for l in sh.split("\n") if u in l), "")
            return f"副作用: {u.strip()}  ← {ln[:60]}"
    if not write:
        m = REDIRECT.search(sh)
        if m:
            ln = sh[:m.start()].split("\n")[-1] + sh[m.start():].split("\n")[0]
            return f"副作用: リダイレクト  ← {ln.strip()[:60]}"
    if PLACEHOLDER.search(body):
        return "プレースホルダあり"
    if "$TAG" in body or "$SEL" in body or "$C" in body:
        return "未定義変数（TAG/SEL/C）"
    return None


def run(body: str, cwd: Path) -> tuple[int, str, str]:
    # ⚠ `-euo pipefail`。29 巡目まで素の `bash -c` で**最後のコマンドの rc しか見ておらず**、
    #    0.1 の `uv sync` を壊しても末尾が `ls | wc -l` なので rc=0 と報告していた。
    #    実効カバレッジは「実行 10」ではなく **中身を検証していたのは 1 本だけ**だった。
    r = subprocess.run(["bash", "-euo", "pipefail", "-c", body], capture_output=True,
                       text=True, cwd=cwd, timeout=900)
    # ⚠ 末尾 1 行だけを返すと、期待メッセージ照合が「最後に出た行」に依存する。
    #    全出力を返し、表示だけ末尾に切る。
    full = (r.stdout + "\n" + r.stderr).strip()
    tail = full.splitlines()
    return r.returncode, (tail[-1][:110] if tail else ""), full


def _snapshot(tmp: Path) -> None:
    """`results/z0` を退避する。書き込みブロックを実行しても正本を壊さないため。

    ⚠ `*.pt` を除外しない——ピンガードは `rm -f "$PIN"` を実行するので、
    除外すると `pin_before/ship_item_12.pt` が復元できずに永久に失われる。
    退避対象を `results/z0` に絞れば数十 MB で収まる。
    """
    import shutil
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(ROOT / "results/z0", tmp, symlinks=True)


def _restore(tmp: Path) -> None:
    import shutil
    dst_root = ROOT / "results/z0"
    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(tmp, dst_root, symlinks=True)


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
    show = "--list" in sys.argv
    write = "--write" in sys.argv     # 書き込みブロックも実行（results/ は退避・復元）
    tmp = Path("/tmp/gate_run_results_backup")
    if write:
        _snapshot(tmp)
        print(f"  results/ を {tmp} に退避した（終了時に復元）")
    ran = skipped = failed = unexpected = 0
    for d, cwd in DOCS.items():
        text = d.read_text(errors="ignore")
        for ln, body in blocks(text):
            why = classify(body, write)
            if why:
                skipped += 1
                if show:
                    print(f"  skip {d.name}:{ln}  ({why})")
                continue
            ran += 1
            rc, tail, full = run(body, cwd)
            # 直前 6 行に宣言があるか
            head = "\n".join(text.split("\n")[max(0, ln - 7): ln - 1])
            m_nz = re.search(r"gate: expect-nonzero(?:\s+/(.+?)/)?", head)
            exp_nz = m_nz is not None
            exp_re = m_nz.group(1) if m_nz and m_nz.group(1) else None
            ok = (rc != 0) if exp_nz else (rc == 0)
            if ok and exp_re and not re.search(exp_re, full):
                # ⚠ 「非零であること」だけでは、関門を常時 abort させても ok になる。
                #    宣言に期待メッセージを持たせ、その理由で止まったことまで見る。
                ok = False
                tail = f"期待メッセージ /{exp_re}/ と違う理由で止まった: {tail}"
            mark = "ok  " if ok else "MISMATCH"
            if rc != 0:
                failed += 1
            if not ok:
                unexpected += 1
            print(f"  {mark} {d.name}:{ln}  rc={rc}"
                  f"{'（expect-nonzero）' if exp_nz else ''}  {tail}")
    if write:
        _restore(tmp)
        print(f"  results/ を復元した")
    # ⚠ カバレッジ床。ブロック先頭に未許可コマンドを足すだけで実行対象が黙って減る
    #    （32 巡目の実測: `mkdir -p /tmp/x &&` を足すと 6 → 5 になるが rc=0 だった）。
    FLOOR = int(os.environ.get("GATE_RUN_FLOOR", "7"))   # ⚠ 実測値。1 下げると規則 1 の ship_check ブロックを落とせた
    if ran < FLOOR:
        unexpected += 1
        print(f"  ⚠ 実行対象が {ran} 本（床 {FLOOR}）。ブロックが黙って除外に落ちている")
    print(f"\n  実行 {ran} / 除外 {skipped} / 非零終了 {failed}")
    print("  ⚠ 非零終了は必ずしも欠陥ではない（前提条件で正しく止まる場合がある）。"
          "**手順書側で `<!-- gate: expect-nonzero 理由 -->` を宣言する。**"
          "宣言と実測が食い違ったときだけ非零で返す。")
    return 1 if unexpected else 0


if __name__ == "__main__":
    raise SystemExit(main())
