"""手順書の整合を機械で検査する。

4 回の改訂で毎回 3〜8 件の新しい欠陥を作った。内訳はほぼ 3 種類しかない:

  1. 参照先のずれ  — 「worktree で作業」と書いた後の手順が元ツリーを cd している、
                     存在しないスクリプトを叩いている、消した run を decided_by に残している
  2. 記号の不一致  — 手順書の定数（cin=851, NBIN=257 等）が実装コードとずれる
  3. 動かないコード — ヒアドキュメントに未定義の名前が残る

どれも読み返しでは見落とす。ship_check.py と同じで、**規則を検査に変える**。

    uv run python doc_check.py
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "current/PROCEDURE.md", ROOT / "current/PROCEDURE_APP.md"]

# 手順書が「これから作る」と宣言してよいもの（存在しなくても FAIL にしない）
TO_BE_CREATED = {
    # 手順の中で作るスクリプト
    "rtf_bench.py", "parity_check.py", "ceiling_check.py", "export_weights.py", "v1d.py",
    "render_ear_$TAG.py", "doc_check.py",
    # 手順の成果物（まだ無くて当然。手順が生成する）
    "VERDICT.md", "knob_contract.json", "kansei_evals.jsonl",
    "target_platform.json", "ledger.json", "deny.toml", "about.toml",
    "installed.json", "NOTICE", "manifest.json",
    "mutate_check.py", "gate_run.py", "ceiling_decision.json", "future_inv_net_", "export_v1d.py", "buffer_warning_check.md", "runtime_contract.json", "baseline.json", "eta.json", "rtf_front.json", "latency_ledger.json", "ci_manual.md", "plan_numbers.json", "measure_io_latency.py", "graph_status.md", "graph_status.json", "hfw.txt", "latency_decision.md", "io_latency_L_io_a.json", "plugin_ids.json", "held_out_pick.json", "abi_notes.txt", "build.rs", "contract.toml", "lib.rs", "annot_gui.py", "license_decisions.md", "model_license.md", "daw_matrix.md", "SHIPPED.md", "v1d_prior.md",
}

# 手順書に書かれた定数 -> 実装側の (モジュール, 属性)
CONSTS = {
    "NFFT_S": ("ship_front", "NFFT_S"), "HOP_S": ("ship_front", "HOP_S"),
    "NFFT_A": ("ship_front", "NFFT_A"), "HOP_A": ("ship_front", "HOP_A"),
    "N_MEL": ("ship_front", "N_MEL"), "KMAX": ("ship_front", "KMAX"),
    "VOI_ABS": ("ship_front", "VOI_ABS"), "MEL_REF": ("ship_front", "MEL_REF"),
    "F0_MIN": ("ship_front", "F0_MIN"), "F0_MAX": ("ship_front", "F0_MAX"),
}
# 実装から導出される値（モジュール属性ではない）
DERIVED = {
    "NBIN": lambda sf: sf.NFFT_S // 2 + 1,
    "cin": lambda sf: 80 + 3 * (sf.NFFT_S // 2 + 1),
}

fails: list[str] = []
warns: list[str] = []


def check_paths(doc: Path, text: str) -> None:
    """本文に出てくるリポジトリ相対パスが実在するか。"""
    for m in re.finditer(r"`([a-z][\w./-]*\.(?:py|rs|json|jsonl|md|toml|pt))`", text):
        p = m.group(1)
        if Path(p).name in TO_BE_CREATED or "<" in p or "$" in p:
            continue
        cands = [ROOT / p, ROOT / "training" / p, ROOT / "current" / p,
                 ROOT / "results/z0" / Path(p).name]
        cands += list((ROOT / "crates").glob(f"*/src/{Path(p).name}"))
        cands += list((ROOT / "crates").glob(f"*/{Path(p).name}"))
        if not any(c.exists() for c in cands):
            fails.append(f"{doc.name}: 参照先が存在しない -> {p}")


def check_commands(doc: Path, text: str) -> None:
    """`uv run python X.py` の X.py が実在するか。"""
    for m in re.finditer(r"uv run python\s+([\w./-]+\.py)", text):
        f = m.group(1)
        if Path(f).name in TO_BE_CREATED or "$" in f:
            continue
        if not (ROOT / "training" / Path(f).name).exists():
            fails.append(f"{doc.name}: 実行対象が存在しない -> {f}")


def check_consts(doc: Path, text: str) -> None:
    """手順書の定数が実装とずれていないか。

    ⚠ 2026-08-10 まで、この検査は正規表現が文書の書き方と噛み合わず
    **両文書で 1 件も照合していなかった**（NFFT_A を 2048 に改竄しても FAIL 0）。
    docstring が「3 種類しかない」欠陥の 2 番目に挙げた当のものが空回りしていた。
    """
    if doc.name != "PROCEDURE.md":                   # APP は写しなので課さない
        return
    sys.path.insert(0, str(ROOT / "training"))
    for name, (mod, attr) in CONSTS.items():
        hits = _const_hits(name, text)
        if not hits:
            warns.append(f"{doc.name}: {name} が本文に現れない（検査が空回りしている）")
            continue
        try:
            real = getattr(__import__(mod), attr)
        except Exception as e:                       # noqa: BLE001
            warns.append(f"{doc.name}: {name} を実装から読めない ({e})")
            continue
        for h in set(hits):
            if abs(float(h) - float(real)) > 1e-9:
                fails.append(f"{doc.name}: {name} が実装と不一致 "
                             f"(手順書 {h} / 実装 {real})")


def _const_hits(name: str, text: str) -> list[str]:
    """`NAME = 1.5` / `NAME` = 1.5 / | `NAME` | **1.5** の 3 形を拾う。**負号込み**。

    ⚠ 先読みで `*` を一律に除くと、太字の閉じ `**` をかけ算と誤認して
      `-1.0254**` が `-1.025` に切り詰まる（桁が 1 つ消えても検査は通る）。
      かけ算として除くのは**空白で挟まれた** ` * ` だけにする。

    ⚠ 2026-08-10 まで 1 形しか拾えず、10 定数中 9 が照合ゼロだった
    （文書は `` `VOI_ABS = 1.1505` `` とバックティックの中に = ごと書く）。
    """
    out = re.findall(rf"`{name}\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)`(?![0-9])(?!\s*[+＋×])(?!\s\*\s)", text)
    # 素の `NAME = 値`。ただし式の一部（`NFFT_A / HOP_A = 4`）と仮定文は除く
    for m in re.finditer(rf"(?<![/×*+＋])\s{name}\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)(?!\s*[0-9+＋×*])",
                         text):
        near = text[m.end(): m.end() + 24]            # 直後だけ見る（行全体は広すぎる）
        if any(k in near for k in ("にして", "にする", "を採る", "なら")):
            continue                                  # 仮定・分岐の記述
        out.append(m.group(1))
    out += re.findall(rf"`{name}`\s*(?:＝|=|は)\s*\**(-?[0-9]+(?:\.[0-9]+)?)(?![0-9])(?!\s*[+＋×])(?!\s\*\s)", text)
    # 表形式は「| `NAME` | **値** |」＝直後のセルだけ（別名を挟む行は拾わない）
    out += re.findall(rf"^\|\s*\**`{name}`\**[^|\n]*\|\s*\**(-?[0-9]+(?:\.[0-9]+)?)\**\s*\|",
                      text, re.M)
    return out


# 文書中の表現 -> plan_numbers の値。**値をここで定義しない**（正本は plan_numbers.py）
# 実測由来の量は走行ごとに ±5% ぶれる。厳密一致を課すと**測定ノイズが FAIL に化ける**
# （2026-08-10 に実際に起きた）。∴ 実測量は許容幅つき、実装定数は厳密一致。
# ⚠ 許容幅は **ぶれ幅から導く**。定数 0.02 を置いていたら、判定余裕 0.007 より粗くて
# 検査が構造的に空回りしていた（22 巡目のレビューが指摘）。ぶれ幅が未知なら厳しい既定。
def _tol() -> float:
    sys.path.insert(0, str(ROOT / "training"))
    try:
        sp = __import__("plan_numbers").RTF_SPREAD
    except Exception:                                 # noqa: BLE001
        return 0.01
    return 0.01 if sp is None else max(round(sp, 4), 0.005)


PLAN_TOL = {k: _tol() for k in ("rtf.front_fused", "rtf.net_budget",
                                "rtf.cand0_front", "rtf.v_only")}
PLAN_TOL["rtf.front_current"] = _tol()
PLAN_TOL["rtf.full_graph"] = 0.0
PLAN_TOL["rtf.v_only_k7"] = 0.0
PLAN_TOL["rtf.full_graph_k7"] = 0.0        # 関門の入力。丸めも許容しない
PLAN_PATTERNS = {
    "rtf.front_fused":  r"front-end[^\n]{0,24}?融合[^\n]{0,10}?\*\*([0-9.]+)\*\*"
                       r"|front-end 計\*?\*?\s*\|[^|\n]*\|\s*\*\*([0-9.]+)\*\*",
    # ⚠ 現行列は 26 巡目まで一度も照合されていなかった（融合列だけを見ていた）
    "rtf.front_current": r"front-end 計\*?\*?\s*\|\s*\*\*([0-9.]+)\*\*"
                        r"|front-end\*?\*? \| 現行 \*\*([0-9.]+)\*\*",
    "rtf.net_budget":   r"net\s*(?:に残る)?予算[^\n]{0,24}\|\s*無し\s*\|\s*\*\*([0-9.]+)\*\*"
                        r"|net\s*(?:に残る)?予算(?:は)?\s*\*\*([0-9.]+)\*\*",
    "rtf.cand0_front":  r"候補 0[^\n]{0,60}?front-end\s*\*?\*?([0-9.]+)",
    "rtf.v_only":       r"V 単体\s*\*?\*?([0-9.]+)",
    # ⚠ full graph は**関門を開ける唯一の入力**なのに rev57 まで照合対象外だった
    # （22 巡目のレビューが指摘）。0.48 と 0.49 が同じ文書に同居していた。
    # 23 巡目: 派生値 k7 が検査外で腐っていた
    "rtf.v_only_k7":    r"参考:? ?k=7 [^\n]{0,14}?([0-9]\.[0-9]+)",
    "rtf.full_graph_k7": r"参考:? ?k=7 [^\n]{0,14}?[0-9]\.[0-9]+ *(?:/|→) *([0-9]\.[0-9]+)",
    "rtf.full_graph":   r"full graph\s*=\s*[0-9.]+\s*\+[^=\n]*=\s*\*?\*?([0-9.]+)"
                        r"|`rtf_target`（暫定）\*?\*?\s*\|\s*\*\*([0-9.]+)\*\*"
                        r"|を足した \*\*([0-9.]+)\*\* なら本命",
    "impl.NBIN":        r"^\|\s*`NBIN`\s*\|\s*\*\*([0-9]+)\*\*",
    "impl.CTX":         r"CTX\s*=\s*\(k_in−1\)\s*\+\s*\(k−1\)×L\s*=\s*[0-9]+\s*\+\s*[0-9]+\s*=\s*\*?\*?([0-9]+)"
                        r"|`CTX`[^\n]{0,20}?→\s*\*\*([0-9]+)\*\*",
    "impl.k_in":        r"`k_in`（入力 `Conv1d`）\s*\|\s*\*\*([0-9]+)\*\*",
    "impl.k":           r"`k`（`ConvNeXtBlock1d`）\s*\|\s*\*\*([0-9]+)\*\*",
    "impl.L":           r"^\|\s*`L`\s*\|\s*\*\*([0-9]+)\*\*",
    "impl.dim":         r"^\|\s*`dim`\s*\|\s*\*\*([0-9]+)\*\*",
    "impl.cin":         r"^\|\s*`cin`\s*\|\s*\*\*([0-9]+)\*\*",
    "impl.input_ring_samples": r"(?:入力リング|input_ring_samples)\s*[=＝]?\s*\*?\*?([0-9]{3,})",
    "impl.excitation_ring_samples": r"(?:励起リング|excitation_ring_samples)\s*[=＝]?\s*\*?\*?([0-9]{3,})",
}


_plan_seen: set = set()


def check_plan_numbers(doc: Path, text: str) -> None:
    """文書中の数値が plan_numbers.py（正本）と一致するか。

    同じ量が節をまたいでリテラルで散在し、1 つ直すと下流が腐る事故が
    16 巡で毎回出た。**値は plan_numbers.py にしか置かない**のが対処で、
    この検査はその規約が守られているかを見る。
    """
    sys.path.insert(0, str(ROOT / "training"))
    try:
        rep = __import__("plan_numbers").report()
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: plan_numbers を読めない ({e})")
        return
    for key, pat in PLAN_PATTERNS.items():
        sec, name = key.split(".")
        want = rep[sec][name]
        for m in re.finditer(pat, text, re.M):
            _plan_seen.add(key)          # 一致はした（候補 0 で除外されても空回りではない）
            if "候補 0" in text[max(0, m.start() - 90): m.start()] and not key.endswith("cand0_front"):
                continue                              # 候補 0 は別構成
            got = next((g for g in m.groups() if g), None)
            if got is None:
                continue
            tol = PLAN_TOL.get(key, 1e-9)
            _plan_seen.add(key)
            if abs(float(got) - float(want)) > tol:
                fails.append(f"{doc.name}: {key} が正本と不一致 "
                             f"(文書 {got} / plan_numbers {want} / 許容 ±{tol})")
    # ⚠ 一度も一致しなかったパターンは「検査が空回りしている」。
    #   rev60 の rtf.*_k7 は本文が「参考: k=7」に変わっただけで無音のまま通っていた。
    if doc.name == DOCS[-1].name:
        for key in PLAN_PATTERNS:
            if key not in _plan_seen:
                warns.append(f"{key} のパターンがどちらの文書にも一致しない"
                             f"（検査が空回りしている）")


def check_orphan_rows(doc: Path, text: str) -> None:
    """区切り行（|---|）を持たない表ブロックを検出する。

    節を差し込むとヘッダと本体が分断され、行が生テキストとして描画される。
    rev39 で中心量（RTF 0.48 の成否）を運ぶ行がこの状態だった。
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and lines[i].rstrip().endswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            block = lines[i:j]
            seps = [k for k, b in enumerate(block) if re.match(r"^\|[\s:|-]+\|$", b)]
            if not seps:      # ⚠ 1 行だけの孤児も見る（28 巡目: 5.1 の高域検査が 1 行で切り離されていた）
                fails.append(f"{doc.name}: L{i+1} 表に区切り行が無い（孤児行）"
                             f" -> {block[0][:50]}")
            elif seps and seps[0] != 1:
                fails.append(f"{doc.name}: L{i+1} 表のヘッダが {seps[0]} 行ある"
                             f"（区切り行の直前 1 行だけがヘッダ）-> {block[0][:44]}")
            elif len(seps) > 1:
                fails.append(f"{doc.name}: L{i+1} 表に区切り行が {len(seps)} 本"
                             f" -> {block[0][:44]}")
            i = j
        else:
            i += 1


# 昇格の可否を決める閾値。実装定数と違って「腐っても動く」ので検査が要る。
# 2026-08-10 の変異テストで、これらは 1 つも検出されなかった。
THRESHOLDS = {
    "N_TRIAL":      (r"export N_TRIAL=([0-9]+)", 10),   # 5.2 の 20 は別文脈
    "Nv 下限":      (r"Nv\s*≥\s*([0-9]+)\s*を確認", 4),
    "delta_ceiling": (r"劣化\s*δ\s*=\s*([0-9.]+)\s*内", 0.15),
    "parity_snr":   (r"SNR\s*≥\s*([0-9]+)\s*dB", 80),
    "rtf_pass":     (r"合格線\s*([0-9.]+)(?!\d)", 0.35),
    "e2e_budget":   (r"設計目標\s*(?:p95\s*<\s*)?([0-9]+)\s*ms", 30),
    "e2e_disq":     (r"p95\s*≥\s*([0-9]+)\s*ms\s*は失格", 50),
}


def check_solution_counts(doc: Path, text: str) -> None:
    """「導出 q での全探索」表の件数が plan_numbers.solutions() と一致するか。

    rev34 で q を導出値にしたとき件数が 30/25/15/10 -> 8/8/6/4 に変わったが、
    撤回前の表が別節に残り、そちらだけ 30/15 のままだった（22 巡目で発覚）。
    件数は導出量なので**文書に書いた瞬間に腐る**。ここで毎回突き合わせる。
    """
    if doc.name != "PROCEDURE_APP.md":
        return
    sys.path.insert(0, str(ROOT / "training"))
    try:
        PN = __import__("plan_numbers")
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: plan_numbers を読めない ({e})")
        return
    head = "| 前提 RTF | 出典 | 導出 q での解 |"
    i = text.find(head)
    if i < 0:
        fails.append(f"{doc.name}: 「導出 q での全探索」表が消えた（件数の正本が無くなる）")
        return
    body = text[i:text.find("\n\n", i)]
    rows = 0
    for m in re.finditer(r"^\|\s*\*?\*?([0-9.]+)\*?\*?\s*\|[^|\n]*\|\s*\*?\*?([0-9]+)\*?\*?\s*\|"
                         r"[^|\n]*\|[^|\n]*\|\s*\*?\*?([0-9—-]+)", body, re.M):
        rt = float(m.group(1))
        rows += 1
        want = len(PN.solutions(rt))
        if int(m.group(2)) != want:
            fails.append(f"{doc.name}: RTF {rt} の導出 q 解が {m.group(2)}（正 {want}）")
        free = m.group(3)
        if free.isdigit() and int(free) != len(PN.solutions(rt, free_q=True)):
            fails.append(f"{doc.name}: RTF {rt} の q 自由件数が {free}"
                         f"（正 {len(PN.solutions(rt, free_q=True))}）")
    if rows < 6:
        fails.append(f"{doc.name}: 導出 q 表の行が {rows} 行しか読めない（表が壊れている）")



def check_rtf_verdict(doc: Path, text: str) -> None:
    """文書が掲げる 2.4b の判定が plan_numbers.rtf_verdict() と一致するか。

    余裕(0.0070)がぶれ幅(0.0073)未満なのに 2 桁に丸めると 0.01 > 0.0073 で
    PASS に見える。**丸めが判定を救う**事故を検出する。
    """
    sys.path.insert(0, str(ROOT / "training"))
    try:
        PN = __import__("plan_numbers")
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: plan_numbers を読めない ({e})")
        return
    _mn, _src = PN.measured_net()
    vd, mg = PN.rtf_verdict(_mn if _mn is not None else PN.RTF_NET_DEFAULT,
                            budget=0.35 if _src == "fullgraph" else None)
    m = re.search(r"net [0-9.]+ vs 予算 [0-9.]+ → \*{0,2}([A-Z]+)", text)
    if not m:                                    # APP は判定表の行で持つ
        m = re.search(r"\|\s*\*\*判定\*\*\s*\|\s*\*\*([A-Z]+)\*\*", text)
    if not m:
        fails.append(f"{doc.name}: RTF の「現時点の判定」が本文に無い（正 {vd}）")
        return
    if m.group(1) != vd:
        fails.append(f"{doc.name}: 2.4b の判定が {m.group(1)}（正 {vd}・"
                     f"余裕 {mg} / ぶれ幅 {PN.RTF_SPREAD}）")
    if doc.name == "PROCEDURE.md" and "INCONCLUSIVE を PASS にしない" not in text:
        fails.append(f"{doc.name}: INCONCLUSIVE を PASS にしない規則が消えた")



def check_cliff(doc: Path, text: str) -> None:
    """本命 io=64,q=2 の崖 RTF と、そこを超えたときの E2E が正しいか。

    崖を 1 でも超えると導出 q が 3 に上がり E2E が 30 ms を割る。
    front-end を最悪値に切り替えたら full graph が 0.48 -> 0.49 に動いた——
    **崖までの余裕は 0.02 -> 0.01 に半減しており、文書が追随しないと
    「まだ余裕がある」と読める。**
    """
    if doc.name != "PROCEDURE_APP.md":
        return
    sys.path.insert(0, str(ROOT / "training"))
    try:
        PN = __import__("plan_numbers")
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: plan_numbers を読めない ({e})")
        return
    cliff = max(r for r in [x / 100 for x in range(30, 100)]
                if PN.q_of(r, 64, 44100) == 2)
    over = PN.q_of(cliff + 0.01, 64, 44100)
    c, h, _ = PN.latency(44100, 64, over)
    m = re.search(r"本命 `io=64, q=2` は `rtf = ([0-9.]+)` が崖", text)
    if not m:
        fails.append(f"{doc.name}: 崖の記述が無い（正 {cliff}）")
        return
    if abs(float(m.group(1)) - cliff) > 1e-9:
        fails.append(f"{doc.name}: 崖が {m.group(1)}（正 {cliff}）")
    e2e = round(c + h, 2)
    if f"{e2e:.2f} ms" not in text:
        fails.append(f"{doc.name}: 崖越えの E2E {e2e:.2f} ms が本文に無い")
    full = PN.report()["rtf"]["full_graph"]
    if abs(full - 0.49) < 1e-9 and "崖 0.50 まで **0.01**" not in text:
        fails.append(f"{doc.name}: full graph {full} なのに崖までの余裕の記述が古い")



def check_json_blocks(doc: Path, text: str) -> None:
    """```json ブロックが json.loads を通るか。

    CI と F.5 が機械で読むと宣言したファイルの雛形に行コメントとカンマ落ちが
    あり、serde_json / json.loads で落ちる状態だった（22 巡目のレビューが指摘）。
    """
    import json as _j
    for k, b in enumerate(re.findall(r"```json\n(.*?)```", text, re.S)):
        b = "\n".join(re.sub(r"^>\s?", "", ln) for ln in b.split("\n"))
        try:
            _j.loads(b)
        except Exception as e:                        # noqa: BLE001
            head = b.split("\n")[0][:50]
            fails.append(f"{doc.name}: json ブロック {k} が壊れている "
                         f"({str(e)[:60]}) 先頭: {head}")



def _ncol(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))          # `\|` はセル内のエスケープ


def check_table_cols(doc: Path, text: str) -> None:
    """表の各行の列数がヘッダと一致するか。

    5.1 の関門表はヘッダ 3 列なのに ceiling 行が 2 列で、
    **「落ちたときの行き先」が物理的に欠落**していた（22 巡目のレビューが指摘）。
    check_orphan_rows は区切り行しか見ておらず、素通りしていた。
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("|") and i + 1 < len(lines) \
                and re.fullmatch(r"\|[\s:|-]+\|", lines[i + 1].strip()):
            ncol = _ncol(lines[i])
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                if _ncol(lines[j]) != ncol:
                    fails.append(f"{doc.name}:{j+1}: 表の列数が {_ncol(lines[j])-1} "
                                 f"（ヘッダは {ncol-1}）。落ちた行き先などが欠落している疑い: "
                                 f"{lines[j][:60]}")
                j += 1
            i = j
        else:
            i += 1


def check_tee_pipefail(doc: Path, text: str) -> None:
    """`| tee` 付きのガードに `set -o pipefail` があるか。

    パイプラインの終了コードは tee の 0 になるので、`|| { exit 1; }` が
    **ship_check の FAIL も ImportError も 1 件も捕まえない**。
    5.5-C はこれで 24 GPU 時間を空ゲートで焼く経路だった。
    """
    for m in re.finditer(r"```bash\n(.*?)```", text, re.S):
        b = m.group(1)
        if "| tee" in b and "||" in b and "set -o pipefail" not in b:
            ln = text[:m.start()].count("\n") + 1
            fails.append(f"{doc.name}:{ln}: `| tee` と `||` があるのに "
                         f"`set -o pipefail` が無い（ガードが常に素通りする）")



def check_formula_sync(doc: Path, text: str) -> None:
    """手順書が再実装している式が plan_numbers と同じ値を返すか。

    −1.1c の検算スクリプトは torch 非依存で走る必要があるので plan_numbers を
    import できない。∴ 式が 2 本になる。本書自身が 3 箇所で「式を 2 本に増やすと
    どちらで実装したかで恒久 FAIL する」と警告しているので、ここで機械照合する。
    """
    if doc.name != "PROCEDURE_APP.md":
        return
    sys.path.insert(0, str(ROOT / "training"))
    try:
        PN = __import__("plan_numbers")
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: plan_numbers を読めない ({e})")
        return
    if "ceil(N − gcd(io × 44100, N × sample_rate) / sample_rate)" not in text \
            and "gcd(io * 44100" not in text and "gcd(io*44100" not in text:
        fails.append(f"{doc.name}: accum の式が本文から消えた（正本 plan_numbers.accum）")
    # ⚠ PDC の式が sinc_len を落としていないか（48kHz・sinc=64 で 4.17ms ずれる）
    if "sinc_len` の関数" not in text and "sinc_len/256" not in text:
        fails.append(f"{doc.name}: fixed_host の式が `sinc_len` 依存を明示していない")
    for sr, io, q in ((48000, 128, 2), (48000, 64, 3)):
        if PN.fixed_host(io, sr, q, sinc=64) == PN.fixed_host(io, sr, q, sinc=256):
            fails.append(f"{doc.name}: plan_numbers.fixed_host が sinc を無視している"
                         f"（{sr}Hz io={io} q={q}）")
    for sr in (44100, 48000):
        for io in (8, 64, 128, 256):
            for q in (1, 2, 3):
                c, h, _ = PN.latency(sr, io, q)
                if c < 0 or h < 0:
                    fails.append(f"{doc.name}: latency({sr},{io},{q}) が負")
    # 本文が掲げる代表値と PN が一致するか
    for m in re.finditer(r"`io=64,\s*q=2`\*?\*?[^|\n]{0,40}?([0-9]+\.[0-9]{2})\s*ms", text):
        want = round(sum(PN.latency(44100, 64, 2)[:2]), 2)
        if abs(float(m.group(1)) - want) > 0.005:
            fails.append(f"{doc.name}: io=64,q=2 の E2E が {m.group(1)} ms（正 {want}）")



def check_key_set(doc: Path, text: str) -> None:
    """`contract.toml` の鍵一覧が C.5-key 以外の場所に書き写されていないか。

    同じ場所が 3 巡連続で割れた（rev57: 4 通り / rev58: 3 通り）。うち 1 通りは
    `contract.toml` に無い `queue_depth_io_blocks` の厳格一致を要求していて実装不能。
    **一覧を 1 箇所に閉じ込めたかを機械で見る**のが唯一の止め方。
    """
    if doc.name != "PROCEDURE_APP.md":
        return
    i = text.find("#### C.5-key")
    if i < 0:
        fails.append(f"{doc.name}: C.5-key の表が消えた（鍵集合の正本が無くなる）")
        return
    j = text.find("\n#### ", i + 10)
    tbl = text[i: j if j > 0 else len(text)]
    banned = ("io_block_samples", "queue_depth_io_blocks", "resamp_samples")
    for b in banned:
        if f"`{b}`" not in tbl:
            fails.append(f"{doc.name}: C.5-key の表が `{b}` を「入れないキー」として"
                         f"明示していない")
    # ⚠ 個数を本文に書かない（rev60 は表 12 行に対し本文が「11 鍵」で割れた）
    for m in re.finditer(r"C\.5-key の[^\n]{0,12}?（?(\d+) ?(?:鍵|キー)", text):
        fails.append(f"{doc.name}: C.5-key の鍵数 {m.group(1)} を本文に書いている"
                     f"（表と割れる。個数ではなく表を指す）")
    # ⚠ 生成物サンプルが禁止キーを含んでいないか
    for mm in re.finditer(r"```json\n(.*?)```", text, re.S):
        b = mm.group(1)
        if "runtime_contract" in text[max(0, mm.start() - 400): mm.start()] \
                or "emit-runtime-contract" in text[max(0, mm.start() - 400): mm.start()]:
            for k in banned:
                if f'"{k}"' in b:
                    fails.append(f"{doc.name}: runtime_contract の雛形が禁止キー "
                                 f"`{k}` を含む（C.5-key に反する）")
    # C.5-key の外で「厳格一致」と鍵名を同じ行に並べていないか
    for k, ln in enumerate(text.split("\n"), 1):
        if i <= sum(len(x) + 1 for x in text.split("\n")[:k - 1]) < (j if j > 0 else len(text)):
            continue
        neg = ("ではなく", "実装不能", "書き写さない", "式一致", "対象外",
               "鍵ではない", "入れない", "置かない")
        # ⚠ 「厳格一致」語が無い形でも割れる（rev59 の D0b は "contract.toml に置け" だった）
        if (("厳格一致" in ln or "`contract.toml`" in ln)
                and "`queue_depth_io_blocks`" in ln
                and not any(x in ln for x in neg)):
            fails.append(f"{doc.name}:{k}: C.5-key の外で "
                         f"`queue_depth_io_blocks` の厳格一致を要求している"
                         f"（`contract.toml` に無い鍵＝実装不能）")



def check_retracted_framing(doc: Path, text: str) -> None:
    """§0.2 が撤回した「先読み 0 の代償」という読み方が残っていないか。

    24 巡目のレビュー: 撤回文を書いた後も 8 箇所に残り、そのうち 1 つは
    24 GPU 時間の行き先（手順 6 ＝ 族の乗り換え）を決める分岐だった。
    撤回文そのもの（§0.2 と逸脱宣言）は残すので、それ以外の行だけを見る。
    """
    if doc.name != "PROCEDURE.md":
        return
    for k, ln in enumerate(text.split("\n"), 1):
        if "先読み 0" not in ln:
            continue
        if any(x in ln for x in ("読まない", "外す", "ではなく", "成立しない",
                                 "だけではない", "疑われているのではない",
                                 "未測定", "何も言えない", "7 件目",
                                 "製品可」ではない",       # §0.2 の撤回本体
                                 "プライム無し・先読み 0",  # 状態機械の性質（別の話）
                                 "front-end の先読み")):
            continue
        fails.append(f"{doc.name}:{k}: §0.2 が撤回した「先読み 0 の代償」が残っている"
                     f"（正: 左寄せ front-end ＋ 窓長 1024 の代償）: {ln.strip()[:50]}")



def check_conventions(doc: Path, text: str) -> None:
    """文書が自分の実行規約を宣言しているか。

    26 巡目: bash ブロックの cwd 規約が無く、リポジトリルートから貼ると
    4 ブロックが ModuleNotFoundError / can't open file で落ちた。
    2 文書で既定 cwd が違う（学習側 training/ ・アプリ側ルート）ので、
    片方だけ直すと混ざる。
    """
    if "を cwd として実行する" not in text:
        fails.append(f"{doc.name}: bash ブロックの作業ディレクトリ規約が宣言されていない")
    if doc.name == "PROCEDURE.md" and "gate_run.py" not in text:
        fails.append(f"{doc.name}: 動的検査 gate_run.py への参照が無い"
                     f"（bash -n だけでは「走らせると落ちる」型を防げない）")



def check_stage_evidence(doc: Path, text: str) -> None:
    """段・出所を名乗るキーが「自己申告」になっていないか。

    27 巡目: `gate_stage` は `rtf_target is not None` だけで measured を名乗れ、
    本書が provisional と呼ぶ値を書くだけで pass:true が出た。
    `d_ahead_source` は出所を残していたのに、同じ台帳の `rtf_target` には無かった。
    **段を名乗るなら、その段である根拠（出所）を同じレコードに残す。**
    """
    if doc.name != "PROCEDURE_APP.md":
        return
    if "'gate_stage'" in text or '"gate_stage"' in text:
        if "rtf_target_source" not in text:
            fails.append(f"{doc.name}: gate_stage があるのに rtf_target_source が無い"
                         f"（段が自己申告になる）")
        if "_stage == 'measured'" not in text:
            fails.append(f"{doc.name}: _ok が gate_stage を見ていない"
                         f"（provisional のまま pass:true が出る）")
    for k in ("io_source", "d_ahead_source", "rtf_target_source"):
        if k in text and f"'{k}'" not in text and f'"{k}"' not in text:
            fails.append(f"{doc.name}: {k} が台帳に書かれていない")



# 手順書のゲートが持つべき文字列。**壊しても FAIL しない検査は存在しない検査**なので、
# 27 巡目に mutate_check.py で「守られていない修正」を洗い出してここに固定した。
GATE_INVARIANTS = [
    ("PROCEDURE.md", "| **`mlin`** | **合成（`HOP_S`）** | **`[..., 7:]`**",
     "2.3 (0) の mlin 除外が合成格子の 7 でない（3 は解析格子の値）"),
    ("PROCEDURE.md", 'rm -f "$PIN"; }',
     "2.1 のピンのキー欠落枝が sha まで消している（改修前の証拠が失われる）"),
    ("PROCEDURE.md", "[ -s ../results/$TAG/melframe.txt ]",
     "4.6 の resume ガードが MELFRAME/MELNFFT を見ていない"),
    # ---- 22〜26 巡の重大（28 巡目に mutate_check で無防備と判明した分）
    ("PROCEDURE.md", "| 7 | **front-end の頭を零詰めにする**",
     "2.4a-2 状態 7 が消えた（reflect の頭は t=0 で先読み 768）"),
    ("PROCEDURE.md", "| 8 | **起動直後は実サンプルだけを解析する**",
     "2.4a-2 状態 8 が消えた（リングのゼロを信号として解析すると頭の f0 が食い違う）"),
    ("PROCEDURE.md", "**未来不変性 ネット込み**",
     "5.1 の ckpt を通した未来不変性の関門が消えた（ship_check は ckpt を読まない）"),
    ("PROCEDURE.md", "共通の**減衰係数",
     "1.2 の盲検が試行内共通係数でなくなった（系ごとだと RMS 整合が壊れ音量で解ける）"),
    ("PROCEDURE.md", "- **隣り合う試行では X/Y/Z の割り当てが必ず違う。**",
     "1.2 の盲検の割り当てが隣接同一を許す（「さっきと同じ」が漏れる）"),
# ---- A1: CLAUDE.md 由来の禁止条項（反転しても検出されなかった）
    ("PROCEDURE.md", "**推論経路に発話全体の統計を置かない**", "規則 2（CLAUDE.md 出荷ゲート）が消えた"),
    ("PROCEDURE.md", "**昇格は耳のみ。**", "規則 4（proxy 単独昇格禁止）が消えた"),
    ("PROCEDURE.md", "中間 step のスナップショットを残す", "規則 7（最終 1 点で判定しない）が消えた"),
        ("PROCEDURE.md", "**同じ TAG で再走行しない。**", "規則 9（上書きで重みを失う）が消えた"),
    ("PROCEDURE.md", "**⚠ 禁止（明文）: V に時間軸をまたぐ正規化層を置かない。**",
     "2.1 の GroupNorm 禁止が消えた（§0.2 の pathB UNBOUNDED を再生産する）"),
    ("PROCEDURE.md", "**加算複素残差にすること（自由位相にしない）。**", "2.1 の自由位相禁止が消えた"),
    ("PROCEDURE.md", "**⚠ 白色雑音で測らない**", "2.4a-2 の白色雑音プローブ禁止が消えた（偽 PASS）"),
    # ---- A2: −1.1c の assert
    # ---- A3: −1.1 の雛形 JSON（1 行書き換えるだけで関門が開く）
    ("PROCEDURE_APP.md", '"rtf_target": null,', "−1.1 雛形の rtf_target が null でない（未実測でゲートが開く）"),
    ("PROCEDURE_APP.md", '"asio_sdk_available": false,', "−1.1 雛形が ASIO 有りを既定にした（WASAPI assert 無効化）"),
    ("PROCEDURE_APP.md", '"owner_accepted_over_30ms": null,', "−1.1 雛形が 30-50ms 受容を既定にした"),
    # ---- A5: 除外量・リング長・K
    ("PROCEDURE.md", "| `f0` | 解析 | **`[..., 7:]`**", "2.3 (0) の f0 除外が 7 でない"),
    ("PROCEDURE.md", "| `mel` | 解析（`HOP_A`） | **`[..., 3:]`**", "2.3 (0) の mel 除外が 3 でない"),
    ("PROCEDURE.md", "| 1 | **入力リング** | **1792 sample**", "2.4a-2 の入力リング長が変わった"),
    ("PROCEDURE.md", "| 5 | **励起リング** | **384 sample**", "2.4a-2 の励起リング長が変わった"),
    # ---- A8: 5.1 の関門構造
    ("PROCEDURE.md", "`unbounded=False` かつ `inconclusive=False`", "5.1 の未来不変性が『または』に緩んだ"),
    ("PROCEDURE.md", "で落ちたら 5.2 に進まない", "5.1 の二値関門が解除された"),
    # ---- A15: 証跡キーの構成依存
    ("PROCEDURE.md", "--steps 200000 --every 2000 --snap 20000", "本走行の --steps/--snap が変わった"),
    ("PROCEDURE.md", "--steps 200000 --every 2000 --snap 8000 --dstart 20000",
     "5.5-C の --snap 8000 が変わった（176000 = 8000×22。20000 だと 176k が採れない）"),
    ("PROCEDURE.md", "**Nv ≥ 4 を必須**", "5.3 の有効試行数の下限が変わった"),
    ("PROCEDURE.md", "**a ≥ Nv/2**", "耳ゲートの合格率が変わった"),
    ("PROCEDURE.md", "再レンダは合計 2 回まで", "1.5 の再レンダ上限が変わった"),
    ("PROCEDURE.md", "1 候補あたりの上限は GPU 時間で数える: 120 h", "6.2 の候補あたり上限（GPU 時間）が変わった"),
    ("PROCEDURE.md", "**実施できた腕が 2 本未満なら族の否定を出さない**", "5.5-A の腕数下限が消えた"),
    # ---- A9: C.3 / C.6 / F.1 の測定条件（全部素通りしていた）
    ("PROCEDURE_APP.md", "最大絶対誤差 ≤ 1e-4", "C.3 の mel 一致許容が緩んだ"),
    ("PROCEDURE_APP.md", "±5 cent", "C.3 の f0 一致許容が緩んだ"),
    ("PROCEDURE_APP.md", "±2 サンプル以内で一致する", "C.6 の PDC 許容が緩んだ"),
    ("PROCEDURE_APP.md", "先頭 **2.0 秒以上を破棄**", "F.1 のウォームアップ破棄が消えた"),
    ("PROCEDURE_APP.md", "4 時間連続", "G.4 の連続稼働時間が縮んだ"),
    # ---- A10: 遅延の正典テーブルの式
    ("PROCEDURE_APP.md", "**`nfft_S − hop_S`**", "項 9 recon の式が変わった（PDC がずれて C.6 が恒久 FAIL）"),
    # ---- A12: 保存先・話者集合
    ("PROCEDURE.md", "すべての ckpt を `../results/<TAG>/` に置く", "ckpt の保存先が変わった（5.1/5.2 のパスが全部外れる）"),
    ("PROCEDURE.md", "Z0C_SPEAKERS=held24", "5.2 の盲検の話者集合が変わった"),
    # ---- C3: 雛形と −1.7 の鍵一覧
    ("PROCEDURE_APP.md", '"resamp_samples": 0,', "−1.1 雛形に resamp_samples が無い（−1.7 が永久に閉じない）"),
]


def _norm(t: str) -> str:
    """表記ゆれを吸収する。`≥/≧`、`−/-`、全角空白。

    29 巡目: `≥` を `≧` に変えるだけ（意味不変）で FAIL する偽陽性があった。
    """
    return (t.replace("≧", "≥").replace("－", "−").replace("\u3000", " "))


_INV_COUNT: dict = {}


def _snapshot_inv_counts() -> None:
    """初回に各 needle の出現数を数えて固定する（`results/z0/inv_counts.json`）。"""
    import json as _j
    f = ROOT / "results/z0/inv_counts.json"
    # ⚠ 無ければ「現在のテキストから数え直す」＝壊れた状態が正になる（32 巡目の実測）。
    #    再生成は `--bless` を明示したときだけ。
    if not f.exists() and "--bless" not in sys.argv:
        fails.append("results/z0/inv_counts.json が無い。壊れた状態に自己修復させない"
                     "（意図して作り直すなら --bless）")
        return
    if f.exists():
        for k, v in _j.loads(f.read_text()).items():
            name, needle = k.split("\u0000", 1)
            _INV_COUNT[(name, needle)] = v
        return
    for name, needle, _ in GATE_INVARIANTS:
        d = ROOT / "current" / name
        _INV_COUNT[(name, needle)] = _norm(d.read_text()).count(_norm(needle))
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_j.dumps({f"{k[0]}\u0000{k[1]}": v for k, v in _INV_COUNT.items()},
                          ensure_ascii=False, indent=1))


def check_gate_invariants(doc: Path, text: str) -> None:
    """ゲートの要となる 1 行が消えていないか。

    「直した」と「再発を止めた」は別物。mutate_check.py が UNGUARDED を出したら
    ここに 1 行足す——それで初めて、明日戻されても気づける。
    """
    if not _INV_COUNT:
        _snapshot_inv_counts()
    for name, needle, why in GATE_INVARIANTS:
        if doc.name != name:
            continue
        nt, nn = _norm(text), _norm(needle)
        got = nt.count(nn)
        if got == 0:
            fails.append(f"{doc.name}: {why}（消えた: {needle[:44]}）")
        elif got != _INV_COUNT.get((name, needle), got):
            # ⚠ 出現数まで固定する。29 巡目: 51 件中 8 件が複数箇所にあり、
            #    片方だけ壊しても needle が残るので素通りした（4.6 の --snap がそれ）。
            fails.append(f"{doc.name}: {why}（出現数 {got}、"
                         f"正 {_INV_COUNT[(name, needle)]}）")



def check_flag_values(doc: Path, text: str) -> None:
    """起動引数の**値**が変わっていないか（GPU 費用と昇格判定に直結する）。"""
    if doc.name != "PROCEDURE.md":
        return
    for flag, want in REQUIRED_FLAG_VALUES.items():
        for m in re.finditer(re.escape(flag) + r"\s+([0-9]+)", text):
            near = text[max(0, m.start() - 90): m.start()]
            if any(k in near for k in ("参考", "旧", "前版", "撤回", "例:", "同じ機構")):
                continue
            if m.group(1) != want:
                fails.append(f"{doc.name}: {flag} が {m.group(1)}（正 {want}）")


def check_export_consts(doc: Path, text: str) -> None:
    """`export DIM=… NLAYER=…` が plan_numbers と一致するか。"""
    if doc.name != "PROCEDURE.md":
        return
    sys.path.insert(0, str(ROOT / "training"))
    try:
        PN = __import__("plan_numbers")
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: plan_numbers を読めない ({e})")
        return
    for m in re.finditer(r"export DIM=([0-9]+) NLAYER=([0-9]+)", text):
        if int(m.group(1)) != PN.DIM or int(m.group(2)) != PN.L_BLK:
            fails.append(f"{doc.name}: export DIM/NLAYER が {m.group(1)}/{m.group(2)}"
                         f"（正 {PN.DIM}/{PN.L_BLK}。2.4b-1 が到達不能と決めた構成で焼かない）")
    if "GAINAUG=1" not in text:
        fails.append(f"{doc.name}: GAINAUG=1（3.5 の凍結値）が消えた")



# 数値の出所を宣言する台帳。`results/z0/number_provenance.json` に持つ。
# **列挙するのは「守るもの」ではなく「守られていないもの」**——29 巡目に
# needle を 51 個並べても素通り率が 100% だったのは、ホワイトリストが
# 4,519 行に対して一般化しないから。burden を逆にする。
PROV_KINDS = ("plan_numbers", "artifact", "canon", "prose")


def _plan_values() -> set:
    sys.path.insert(0, str(ROOT / "training"))
    try:
        r = __import__("plan_numbers").report()
    except Exception:                                 # noqa: BLE001
        return set()
    out = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)):
            out.add(f"{o:g}")
    walk(r)
    return out


def _num_in_plan(n: str) -> bool:
    """末尾ゼロの表記ゆれを吸収して照合する（`0.50` と `0.5`、`0.10` と `0.1`）。"""
    try:
        x = float(n)
    except ValueError:
        return False
    return any(abs(x - v) < 1e-9 for v in _plan_floats())


def _plan_floats() -> set:
    out = set()
    for v in _plan_values():
        try:
            out.add(float(v))
        except ValueError:
            pass
    return out


def check_number_provenance(doc: Path, text: str) -> None:
    """太字の数値すべてに出所があるか。無い数値は「未分類」として WARN。

    出所の種別:
      plan_numbers … `plan_numbers.py` が正本（PLAN_PATTERNS で照合済み）
      artifact     … `results/z0/*.json` が正本（本文はリテラルを持たない）
      canon        … 正典（CLAUDE.md / interpretable_vc.md）の固定値
      prose        … 判定に効かない説明（明示的に許可）

    未分類が減ることが、この文書群が「腐らない」ことの唯一の指標。
    件数だけを出し、個別の FAIL にはしない（141 個を一度に赤くしても直せない）。
    """
    import json as _j
    f = ROOT / "results/z0/number_provenance.json"
    known = _j.loads(f.read_text()) if f.exists() else {}
    # ⚠ 節番号（**5.4** / **2.4b**）を数値として拾わない。
    #    `check_number_provenance` は「判定に効く量」を数えるための検査で、
    #    節番号を混ぜると誤検出が出て台帳が信用されなくなる。
    body = re.sub(r"(?:手順|§|節|`PROCEDURE(?:_APP)?\.md`)\s*\*\*[0-9.]+[a-z]*\*\*",
                  "", text)
    nums = re.findall(r"\*\*((?<![0-9])[0-9]+(?:\.[0-9]+)?)\s*"
                      r"(?:ms|dB|%|サンプル|件|本|名|回|フレーム|時間)?\*\*", body)
    per = known.get(doc.name, {})
    # ⚠ `prose`（判定に効かない）は逃げ道になりうる。**判定語の近くにある prose** は
    #    誤分類の疑いとして必ず報告する（「1 行で全部 prose にする」を封じる）。
    JUDGE = ("合格", "失格", "PASS", "FAIL", "閾値", "上限", "下限", "以内", "未満",
             "以上", "許容", "予算", "判定")
    susp = []
    for m in re.finditer(r"\*\*((?<![0-9])[0-9]+(?:\.[0-9]+)?)\s*"
                         r"(?:ms|dB|%|サンプル|件|本|名|回|フレーム|時間)?\*\*", body):
        n = m.group(1)
        if per.get(n) != "prose":
            continue
        near = text[max(0, m.start() - 60): m.end() + 60]
        # ⚠ 「撤回済み」「参考」と明示されている数値は判定に使われないので許す。
        #    ただし**その明示が近くにある場合だけ**（逃げ道にしない）。
        if any(k in near for k in ("撤回", "参考", "旧", "前版", "節番号")):
            continue
        if any(k in near for k in JUDGE):
            susp.append((n, near.strip()[:40]))
    if susp:
        fails.append(f"{doc.name}: `prose` と宣言した数値が判定語の近くにある {len(susp)} 件"
                     f"（誤分類の疑い。例 {susp[0][0]}: …{susp[0][1]}…）")
    # ⚠ `artifact` は「正本が results/z0/*.json にあり、本文はそれを指すだけ」の意味。
    #    リテラルが本文にある以上、**正本と一致するかを誰かが照合しなければ嘘になる**。
    #    PLAN_PATTERNS が拾っていない artifact を「照合されていない実測値」として報告する。
    unchecked = sorted(n for n, k in per.items()
                       if k == "artifact" and n in set(nums)
                       and not _num_in_plan(n))
    if unchecked:
        fails.append(f"{doc.name}: `artifact` なのに正本と照合されていない実測値 "
                     f"{len(unchecked)} 種（例 {unchecked[:5]}）。"
                     f"PLAN_PATTERNS に足すか、本文からリテラルを消して参照にする")
    un = [n for n in set(nums) if n not in per]
    if un:
        fails.append(f"{doc.name}: 出所が宣言されていない太字数値 {len(un)} 種"
                     f"（例 {sorted(un)[:6]}）。"
                     f"results/z0/number_provenance.json に "
                     f"{{\"{doc.name}\": {{\"<値>\": \"plan_numbers|artifact|canon|prose\"}}}} を足す")



def check_gate_fixtures(doc: Path, text: str) -> None:
    """関門コードが fixture で実行され、変異が捕まることを確かめる。

    32 巡目: テキスト検査は「needle を残したまま意味だけ反転」を見られない。
    ∴ 関門は `training/latency_gate.py` に出し、`gate_mutate.py` が
    10 変異すべてを捕まえることを条件にする。**手順書はそれを呼ぶだけ**。
    """
    if doc.name == "PROCEDURE.md":
        # ⚠ 「文字列が本文にあるか」では、コメントに残っていれば呼び出し行を
        #    消しても通る（34 巡目の実測）。**呼び出し行の形**まで見る。
        if not re.search(r"^bash pre_launch_gate\.sh \|\| \{ return 2>/dev/null \|\| exit 1; \}$",
                         text, re.M):
            fails.append(f"{doc.name}: 4.1 の関門呼び出し行が無い／`|| {{ exit 1; }}` が外れている"
                         f"（27 時間を焼く直前の唯一の関門）")
        return
    if doc.name != "PROCEDURE_APP.md":
        return
    for f_, why in (("latency_gate.py", "−1.1c"),):
        if f_ not in text:
            fails.append(f"{doc.name}: {why} が {f_} を呼んでいない"
                         f"（コードを本文に貼ると意味の反転が検出できない）")
    import subprocess
    r = subprocess.run(["uv", "run", "python", "gate_mutate.py"],
                       capture_output=True, text=True, cwd=ROOT / "training",
                       timeout=1800)
    if r.returncode:
        tail = (r.stdout or r.stderr).strip().splitlines()[-3:]
        fails.append(f"{doc.name}: gate_mutate が捕まえられない変異がある -> "
                     + " / ".join(t.strip() for t in tail))



# CLAUDE.md が禁じている語。**本文に出たら FAIL**（needle 方式は「追記」に無力）。
BANNED_PHRASES = [
    ("42 話者", "少数話者・部分集合での本番学習は誤り（CLAUDE.md Data）"),
    ("少数話者で予備", "少数話者・部分集合での本番学習は誤り"),
    ("関門で落ちても", "関門の迂回を認める追記"),
    ("関門が FAIL でも", "関門の迂回を認める追記"),
    ("FAIL でも、GPU に空きがある", "関門の迂回を認める追記"),
    ("conda ", "conda 禁止（CLAUDE.md Environment）"),
    ("conda activate", "conda 禁止"),
    ("synthetic parallel", "VC teacher 蒸留禁止"),
    ("RVC の変換音声", "VC teacher 蒸留禁止"),
    ("rcav_feat の 42 話者だけ", "少数話者・部分集合での本番学習は誤り"),
]
# 禁止条項の直後に例外を足す抜け道を塞ぐ
PROHIBITIONS = [
    "推論経路に発話全体の統計を置かない",
    "昇格は耳のみ",
    "白色雑音で測らない",
    "同じ TAG で再走行しない",
    "自由位相にしない",
    "V に時間軸をまたぐ正規化層を置かない",
]
EXCEPTION_WORDS = ("ただし", "例外", "してよい", "使ってよい", "代替してよい",
                   "省略してよい", "上書き可", "でよい")


def check_prohibitions(doc: Path, text: str) -> None:
    """禁止条項が「例外の追記」で骨抜きにされていないか＋禁止語が本文に無いか。

    33 巡目: `GATE_INVARIANTS` は「消えたか」しか見ないので、
    needle を残したまま直後に「ただし〜してよい」を足すと**全部素通り**した。
    さらに CLAUDE.md 違反の指示を**新規に足す**方向にも無力だった
    （conda / VC teacher 蒸留 / 少数話者）。
    """
    for ph, why in BANNED_PHRASES:
        if ph in text:
            i = text.find(ph)
            near = text[max(0, i - 60): i + 40].replace("\n", " ")
            if "禁止" in near or "違反" in near or "使わない" in near:
                continue                              # 禁止を述べている文脈
            fails.append(f"{doc.name}: 禁止語「{ph}」が本文にある（{why}）: …{near[:50]}…")
    for pr in PROHIBITIONS:
        for m in re.finditer(re.escape(pr), text):
            tail = text[m.end(): m.end() + 90]
            hit = [w for w in EXCEPTION_WORDS if w in tail]
            if hit:
                fails.append(f"{doc.name}: 禁止条項「{pr}」の直後に例外語 {hit} がある"
                             f"（needle を残したまま骨抜きにする型）: …{tail[:44]}…")



def check_pins(doc: Path, text: str) -> None:
    """検査資産（ハーネス・fixture・禁止段落・検査の本数）の同一性。

    34 巡目: ハーネスの 1 行削除が 20/22 素通りした。検査を足して守ってきた
    33 巡ぶんの資産が、検査ファイル側を消すだけで**無音のまま失われる**。
    しかも壊れた状態ほど健全に見える。∴ 資産そのものを pin する。
    """
    if doc.name != DOCS[0].name:
        return                                        # 1 回だけ
    sys.path.insert(0, str(ROOT / "training"))
    try:
        bad = __import__("pins").verify()
    except Exception as e:                            # noqa: BLE001
        fails.append(f"pins を読めない ({e})")
        return
    for b in bad:
        fails.append(b)



def check_threshold_provenance(doc: Path, text: str) -> None:
    """判定語の近くの数値すべてに出所があるか（閾値版の burden 逆転）。

    34 巡目: `THRESHOLDS` の 7 個以外は 33/33 素通りした。個別列挙は
    4,500 行に対して一般化しない（29 巡目の needle と同じ結論）。
    ∴ **「合格線・失格・以内・未満・以上」の近くにある数値で、
    正本と照合されていないものを数え上げる**。件数が台帳より増えたら FAIL。
    """
    import json as _j
    f = ROOT / "results/z0/threshold_provenance.json"
    known = _j.loads(f.read_text()) if f.exists() else {}
    per = known.get(doc.name, {})
    JUDGE = ("合格線", "失格", "以内", "未満", "以上", "を超え", "上限", "下限",
             "許容", "閾値", "合格 |", "PASS", "FAIL")
    found = {}
    for m in re.finditer(r"((?<![0-9])[0-9]+(?:\.[0-9]+)?)\s*(ms|dB|%|回|本|名|サンプル|フレーム|時間)?",
                         text):
        near = text[max(0, m.start() - 45): m.end() + 45]
        if not any(k in near for k in JUDGE):
            continue
        key = f"{m.group(1)}{m.group(2) or ''}"
        found.setdefault(key, 0)
        found[key] += 1
    un = sorted(k for k in found if k not in per)
    if un:
        fails.append(f"{doc.name}: 判定語の近くにあるのに出所が宣言されていない数値 "
                     f"{len(un)} 種（例 {un[:6]}）。"
                     f"results/z0/threshold_provenance.json に "
                     f"{{\"<値>\": \"plan_numbers|canon|artifact|prose\"}} を足す")
    # 件数が減る方向（＝閾値を消す）も検出する
    gone = sorted(k for k in per if k not in found)
    if gone:
        fails.append(f"{doc.name}: 台帳にあるのに本文から消えた閾値 {len(gone)} 種"
                     f"（例 {gone[:6]}）")



def check_net_stale(doc: Path, text: str) -> None:
    """net RTF が未再現の間、その数字を使う節すべてに開示が付いているか。

    0.20 / 0.26 / 0.53 は 2.1 のノブ確定と 2.4b-1 の到達論の両方を支えるのに、
    「rtf_bench.py が無いので再現できない」開示は 2.4b の 1 箇所にしか無かった。
    2.1 から入った読者は「実測」としか読めず、未再現の数字で 27 時間を焼く。
    """
    if doc.name != "PROCEDURE.md":
        return
    sys.path.insert(0, str(ROOT / "training"))
    try:
        stale = __import__("plan_numbers").report()["rtf"]["net_stale"]
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: net_stale を読めない ({e})")
        return
    if not stale:
        return
    n = text.count("net RTF は全て未再現")
    if n < 3:
        fails.append(f"{doc.name}: net RTF が未再現なのに開示が {n} 箇所。"
                     f"2.1 のノブ表・2.4b-1 の段別表・2.4b の 3 箇所に要る")


def check_thresholds(doc: Path, text: str) -> None:
    """判断の閾値が動いていないか。

    実装定数（NBIN/cin/リング）は検査済みだったが、**昇格を決める閾値**
    （N_TRIAL / Nv≥4 / δ=0.15 / SNR≥80 / 0.35 / 30ms / 50ms）は
    改竄しても 1 つも検出されなかった（変異テストで実証）。
    """
    # ⚠ 近傍語ヒューリスティクスは 1 語で黙る（32 巡目の実測: 閾値の 60 字以内に
    #    「参考」を足すだけで検査が消えた。`旧` は 1 文字なので日本語文中に極めて出やすい）。
    #    ∴ **除外は語ではなくマーカー**にする: 直前行に `<!-- th:skip 理由 -->`。
    lines = text.split("\n")
    _off, _line_of = 0, {}
    for _i, _l in enumerate(lines):
        _line_of[_off] = _i
        _off += len(_l) + 1

    def _skipped(pos: int) -> bool:
        k = max((o for o in _line_of if o <= pos), default=0)
        i = _line_of[k]
        return any("<!-- th:skip" in lines[j] for j in range(max(0, i - 1), i + 1))

    for name, (pat, want) in THRESHOLDS.items():
        for m in re.finditer(pat, text):
            if _skipped(m.start()):
                continue                              # 明示マーカーでのみ除外
            got = m.group(1)
            if abs(float(got) - float(want)) > 1e-9:
                fails.append(f"{doc.name}: 閾値 {name} が {got}（正 {want}）。"
                             f"変えるなら doc_check.THRESHOLDS も同時に直す")


def check_ci_exists(doc: Path, text: str) -> None:
    """「CI が検査する」を担保にしている箇所があるのに CI が無い状態を検出する。

    rev37 まで 12 箇所が CI を検収の担保にしていたが `.github/workflows` が無かった。
    """
    if doc.name != "PROCEDURE_APP.md":
        return
    n = len(re.findall(r"CI が[^。\n]{0,24}検査", text))
    has_ci = any((ROOT / ".github/workflows").glob("*.y*ml")) if (
        ROOT / ".github/workflows").exists() else False
    if n and not has_ci and "CI は存在しない" not in text:
        fails.append(f"{doc.name}: 「CI が検査する」が {n} 箇所あるが "
                     f".github/workflows が無い。CI の新設を作業項目にするか、"
                     f"手動代替を明記する")


def check_bash_blocks(doc: Path, text: str) -> None:
    """```bash ブロックが bash -n を通るか＋孤立した継続行が無いか。

    ⚠ PROCEDURE.md の実行コードは全部 ```bash で、```python は 0 個。
    「3 種類の欠陥」の 3 番目（動かないコード）はこの文書に対して空回りしていた。
    実際 rev53 で `git add ... \` の残骸行が独立コマンドとして残り、
    `git commit` が走らない状態が混入した（bash -n は通るので行頭検査も併用する）。
    """
    import subprocess
    for i, blk in enumerate(re.findall(r"```bash\n(.*?)```", text, re.S)):
        body = "\n".join(l[2:] if l.startswith("> ") else l
                         for l in blk.split("\n"))
        # heredoc（<<'EOF' … EOF）と python -c "…" の中身は別言語なので落とす
        body = re.sub(r"<<\s*'?(\w+)'?\n.*?^\1\s*$", "true", body, flags=re.S | re.M)
        body = re.sub(r'python3?\s+-c\s+"(?:[^"\\]|\\.)*"', "true", body, flags=re.S)
        # <プレースホルダ> はその行だけ無害化する（ブロック全体を落とすと
        # 4.1 / 5.2 / 1.1 の構文検査が丸ごと消える＝rev54 で 32 中 5 ブロックが素通りした）
        body = "\n".join(
            "true" if re.search(r"<[^<>\s][^<>]{0,40}>", l) else l
            for l in body.split("\n"))
        r = subprocess.run(["bash", "-n"], input=body, capture_output=True, text=True)
        if r.returncode:
            fails.append(f"{doc.name}: bash ブロック #{i} が構文エラー "
                         f"({r.stderr.strip().splitlines()[-1][:70]})")
        # 継続行の残骸: 直前行が `\` で終わっていないのに深いインデントで始まる
        # ⚠ if/for/while の中身は正当なインデント。深さを追って除外する
        #    （追わないと `if ... then` を書いた瞬間に誤検出する）
        # ⚠ `-c "` を含むブロックは Python 本体を抱えており、インデントは正当。
        #    構文は bash -n が既に見ているので、孤児継続行の推定はシェル専用ブロックに限る。
        # ⚠ `sha256sum -c "..."` のような別コマンドの -c を拾わない
        if re.search(r'(?:uv run )?python3?\s+-c\s+"', body):
            # ⚠ `-c "…"` が閉じずに残る ＝ 中の未エスケープ `"` で
            #    シェル文字列が途中終了している（`bash -n` は通るが python が走らない。
            #    実際に「無出力 rc=0」で 1 度作り込んだ）。
            fails.append(f"{doc.name}: bash ブロック #{i} の `-c \"…\"` が閉じていない"
                         f"（中の `\"` を `\\\"` にエスケープする。"
                         f"閉じないと python が走らず、無出力のまま成功扱いになる）")
            continue
        prev_cont = False
        depth = 0
        for ln in body.split("\n"):
            t = ln.strip()
            if (not prev_cont and depth == 0 and re.match(r"^\s{4,}\S", ln)
                    and not t.startswith(("#", "&&", "||", "|", ")", "}"))):
                fails.append(f"{doc.name}: bash ブロック #{i} に孤立した継続行 "
                             f"-> {ln.strip()[:50]}")
                break
            if re.match(r"^(if|for|while|case|until)\b", t) or t.endswith(("then", "do", "{")):
                depth += 1
            if re.match(r"^(fi|done|esac|\})\b", t):
                depth = max(0, depth - 1)
            prev_cont = ln.rstrip().endswith("\\")


def check_code_claims(doc: Path, text: str) -> None:
    """文書中の「コードに対する断定」を実際に検証する。

    `<!-- claim: unique_grep <needle> <path> -->` … その文字列を含む .py が
    指定 1 本だけか。`<!-- claim: absent <needle> <path> -->` … 含まないか。
    「唯一のコード」「argparse に無い」型の主張が巡ごとに古くなった。
    """
    for m in re.finditer(r"<!--\s*claim:\s*(unique_grep|absent)\s+(\S+)\s+(\S+)\s*-->", text):
        kind, needle, path = m.groups()
        hits = sorted(f.name for f in (ROOT / "training").glob("*.py")
                      if needle in f.read_text(errors="ignore"))
        if kind == "unique_grep" and hits != [Path(path).name]:
            fails.append(f"{doc.name}: claim unique_grep 不成立 -> 「{needle}」は {hits}")
        if kind == "absent" and Path(path).name in hits:
            fails.append(f"{doc.name}: claim absent 不成立 -> {path} に「{needle}」がある")


def check_ladder(doc: Path, text: str) -> None:
    """逓減ラダーの段が全部既定に織り込まれていないか。

    「未達なら降りる」と書いた段が既に既定になっていると、
    2.4b が未達のときに行き先が消える（rev51 で実際に起きた）。
    """
    if doc.name != "PROCEDURE.md":
        return
    if "逓減ラダー（未達なら上から順に）" in text:
        fails.append(f"{doc.name}: 逓減ラダーが「未達なら上から順に」の形。"
                     f"段 1/2/4/5 は 2.1 の既定に織り込み済みで降りる先が無い"
                     f"（rev51 で確認）。行き先（候補 0）を明記する")


def check_k_constraint(doc: Path, text: str) -> None:
    """`K` が K 制約（K × HOP_S が HOP_A の倍数）を満たすか。

    K は 2 文書に散らばり、破ると 5.1 の parity と −1.1c の assert が同時に落ちる。
    """
    sys.path.insert(0, str(ROOT / "training"))
    try:
        sf = __import__("ship_front")
    except Exception:                                 # noqa: BLE001
        return
    for m in re.finditer(r"ブロック長 `K`[^|\n]*\|\s*\*\*([0-9]+)", text):
        k = int(m.group(1))
        if (k * sf.HOP_S) % sf.HOP_A:
            fails.append(f"{doc.name}: K = {k} は K 制約違反"
                         f"（K × HOP_S = {k*sf.HOP_S} が HOP_A = {sf.HOP_A} の倍数でない）")


def check_derived(doc: Path, text: str) -> None:
    """実装から導出される値（NBIN / cin）が本文とずれていないか。"""
    if doc.name != "PROCEDURE.md":
        return
    sys.path.insert(0, str(ROOT / "training"))
    try:
        sf = __import__("ship_front")
    except Exception as e:                            # noqa: BLE001
        warns.append(f"{doc.name}: ship_front を読めない ({e})")
        return
    for name, fn in DERIVED.items():
        real = fn(sf)
        # 候補 0（prior 無し）は cin=80 と別構成なので除外する
        body = re.sub(r"[^\n]*候補 0[^\n]*", "", text)
        for h in set(_const_hits(name, body)):
            if abs(float(h) - float(real)) > 1e-9:
                fails.append(f"{doc.name}: {name} が実装から導出した値と不一致 "
                             f"(手順書 {h} / 実装 {real})")


def check_cd(doc: Path, text: str) -> None:
    """複数の異なる作業ディレクトリを指していないか（worktree 事故の再発防止）。"""
    tgts = {m.group(1).rstrip("/") for m in re.finditer(r"^cd\s+(/\S+)", text, re.M)}
    if len(tgts) > 1:
        fails.append(f"{doc.name}: cd 先が複数ある -> {sorted(tgts)}")


def check_python_blocks(doc: Path, text: str) -> None:
    """```python ブロックが構文として通り、未定義の名前を含まないか。"""
    for i, blk in enumerate(re.findall(r"```python\n(.*?)```", text, re.S)):
        try:
            tree = ast.parse(blk)
        except SyntaxError as e:
            fails.append(f"{doc.name}: python ブロック #{i} が構文エラー ({e.msg})")
            continue
        assigned, used = set(), set()
        for nd in ast.walk(tree):
            if isinstance(nd, ast.Name):
                (assigned if isinstance(nd.ctx, ast.Store) else used).add(nd.id)
            elif isinstance(nd, (ast.FunctionDef, ast.ClassDef)):
                assigned.add(nd.name)
            elif isinstance(nd, ast.arg):
                assigned.add(nd.arg)
            elif isinstance(nd, ast.alias):
                assigned.add((nd.asname or nd.name).split(".")[0])
            elif isinstance(nd, ast.ImportFrom) and nd.names:
                assigned.update((a.asname or a.name) for a in nd.names)
        builtins = set(dir(__builtins__)) | {
            "__name__", "self", "print", "range", "len", "float", "int", "str",
            "open", "sorted", "list", "dict", "set", "min", "max", "abs", "sum"}
        miss = sorted(used - assigned - builtins)
        if miss:
            fails.append(f"{doc.name}: python ブロック #{i} に未定義の名前 -> {miss}")


# 各文書が持っていなければならない節。編集ミスで丸ごと消えても
# 従来の検査は全部 PASS していた（2026-08-10 に約 1290 行を失った）
REQUIRED = {
    "PROCEDURE.md": ["## 0.", "## 手順 1", "## 手順 2", "## 手順 3", "## 手順 4",
                     "## 手順 5", "## 手順 6", "## 全手順に共通する規則", "## 記録の置き場",
                     "### 2.1", "### 2.2", "### 2.3", "### 2.4",
                     "### 3.1", "### 3.2", "### 3.5",
                     "### 4.1", "### 4.2", "### 4.3", "### 4.6",
                     "### 5.1", "### 5.2", "### 5.3", "### 5.4", "### 5.5", "### 5.6",
                     "#### 5.5-A", "#### 5.5-B", "#### 5.5-C",
                     "## 手順 0.5", "### 2.0", "#### 1.6-W", "#### 2.4a", "#### 2.4a-2",
                     "#### 2.4b", "#### 2.4b-1", "### 5.3.1", "### 6.1"],
    "PROCEDURE_APP.md": ["### −1.1", "### −1.1c", "### −1.7", "## 手順 A", "## 手順 B",
                         "## 手順 C", "## 手順 D", "## 手順 E", "## 手順 F", "## 手順 G",
                         "### C.5", "### C.6", "### F.1", "### F.2", "### F.5",
                         "### G.0", "### G.4", "### −1.5d", "### C.1", "### C.4",
                         "### D.検収", "### F.3", "### F.4", "### G.3"],
}


def check_required(doc: Path, text: str) -> None:
    """必須の節が丸ごと消えていないか。切り貼り事故はここでしか捕まらない。"""
    # ⚠ 素の startswith にしない。33 巡目: `### 5.6` → `### 5.6z` や、
    #    `#### 2.4b` を `#### 2.4b-1` の存在だけで満たす抜けがあった
    #    （節名に 1 文字足すだけで必須節検査を空にできた）。
    #    **接頭辞の直後が区切り（空白・終端・全角空白）であること**まで見る。
    heads = [l.rstrip() for l in text.split("\n") if l.startswith("#")]

    def _match(h: str, w: str) -> bool:
        if not h.startswith(w):
            return False
        rest = h[len(w):]
        return rest == "" or rest[0] in " \u3000（(【[:—-" or w.endswith((".", " "))

    for want in REQUIRED.get(doc.name, []):
        if not any(_match(h, want.rstrip()) for h in heads):
            fails.append(f"{doc.name}: 必須の節が無い -> {want}")


def check_dup_heads(doc: Path, text: str) -> None:
    """同じ節が 2 回現れていないか（差し替えで旧節が残る事故）。"""
    from collections import Counter
    heads = [h for h in (re.sub(r"[【（(].*$", "", l).rstrip()
                         for l in text.split("\n") if l.startswith("##"))
             if h.lstrip("#").strip()]
    for h, c in Counter(heads).items():
        if c > 1:
            fails.append(f"{doc.name}: 節が {c} 回ある -> {h[:50]}")


SIZE_FILE = ROOT / "results/z0/doc_sizes.json"


def check_size(doc: Path, text: str) -> None:
    """前回から行数が 10% 以上減っていないか。切り貼り事故は行数に出る。

    正当な削減なら results/z0/doc_sizes.json を消してベースラインを取り直す。
    黙って通さないことが目的。
    """
    import json as _j
    n = text.count("\n") + 1
    try:
        prev = _j.loads(SIZE_FILE.read_text())
    except Exception:                                # noqa: BLE001
        # ⚠ 不在で黙って無効化しない（inv_counts.json は FAIL にしたのに
        #    doc_sizes.json だけ fail-open だった。33 巡目）。
        if "--bless" not in sys.argv:
            fails.append("results/z0/doc_sizes.json が無い。行数ベースラインを"
                         "黙って無効化しない（意図して作り直すなら --bless）")
            return
        prev = {}
    was = prev.get(doc.name)
    if was and n < was * 0.9:
        fails.append(f"{doc.name}: 行数が {was} -> {n}"
                     f"（{100*(1-n/was):.0f}% 減）。意図した削減なら doc_sizes.json を消す")
    if was and n < was * 0.9:
        return                                       # FAIL 時はベースラインを下げない
    prev[doc.name] = max(n, was) if was else n
    SIZE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SIZE_FILE.write_text(_j.dumps(prev, ensure_ascii=False, indent=1))


def check_fences(doc: Path, text: str) -> None:
    """```フェンスが正しく開閉するか。閉じフェンスに info string が付くと閉じない。"""
    open_at = None
    for i, ln in enumerate(text.split("\n"), 1):
        st = ln.strip()
        if not st.startswith("```"):
            continue
        info = st[3:].strip()
        if open_at is None:
            open_at = (i, info)
        elif info:
            fails.append(f"{doc.name}: L{i} 閉じフェンスに info string 「{info}」"
                         f"（L{open_at[0]} で開いたブロックが閉じない）")
            open_at = (i, info)
        else:
            open_at = None
    if open_at is not None:
        fails.append(f"{doc.name}: L{open_at[0]} のフェンスが閉じていない")


def check_tables(doc: Path, text: str) -> None:
    """表の行がセル内改行で壊れていないか（`|` で始まり `|` で終わらない行）。"""
    in_fence = False
    for i, ln in enumerate(text.split("\n"), 1):
        if ln.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        st = ln.rstrip()
        if st.startswith("|") and not st.endswith("|"):
            fails.append(f"{doc.name}: L{i} 表の行が `|` で終わっていない"
                         f"（セル内改行。<br> にする） -> {st[:50]}")


ALL_HEADS: set[str] = set()


def collect_heads() -> None:
    """2 文書の見出しを合わせて集める（相互参照があるため）。"""
    for d in DOCS:
        if not d.exists():
            continue
        t = d.read_text(errors="ignore")
        for m in re.finditer(r"^#{2,4}\s*(?:手順\s*)?([-−\d.A-G]+)", t, re.M):
            ALL_HEADS.add(m.group(1).split(".")[0])
        for m in re.finditer(r"^###\s+([0-9]+\.[0-9]+|[A-G]\.[0-9]+)", t, re.M):
            ALL_HEADS.add(m.group(1).split(".")[0])


def check_sections(doc: Path, text: str) -> None:
    """本文が参照する手順番号が、2 文書のどちらかに実在するか。"""
    for m in re.finditer(r"手順\s*([0-9]+(?:\.[0-9]+)?|[A-G]|−1)", text):
        ref = m.group(1).split(".")[0]
        if ref not in ALL_HEADS:
            warns.append(f"{doc.name}: 参照された手順 {m.group(1)} の見出しが無い")


def check_stale(doc: Path, text: str) -> None:
    """削除した run / 撤回した方針への参照が残っていないか。"""
    # 「使わない」と書いてある否定文は対象外。指示として残っているものだけを見る
    for bad, why in (("git worktree add", "rev4 で撤回した隔離方式"),
                     ("source ../results/*/TAG.env", "rev4 で撤回した TAG 受け渡し"),
                     ('decided_by": "R2', "削除した run への参照")):
        if bad in text:
            fails.append(f"{doc.name}: 撤回済みの記述が残っている -> 「{bad}」（{why}）")


TO_BE_ADDED_FLAGS = {
    ("train_gvoc.py", "--snap"), ("train_gvoc.py", "--resume"),
    ("train_gvoc.py", "--hfw"), ("train_gvoc.py", "--gainaug"),
    ("hf_gate.py", "--ckpt"), ("parity_check.py", "--ckpt"),
    ("ceiling_check.py", "--ckpt"), ("ceiling_check.py", "--ref"),
}
# これから足す予定の choices 値。(script, flag, value)
TO_BE_ADDED_CHOICES = {
    ("train_gvoc.py", "--melframe", "centered"),
}

# 学習を起動するコマンドが必ず渡さなければならないフラグ。
# 「渡し忘れて既定値で 27 時間走る」事故が巡ごとに出た（--hfw / --gainaug / --melframe）
# ⚠ フラグの有無だけでなく**値**を見る。28 巡目: --steps 200000 → 20000 も
#    --snap 8000 → 20000（5.5-C の 176k snapshot が採れず 24 GPU 時間が耳に届かない）も素通りした。
# ⚠ 値が文脈で変わるフラグ（--snap は本走行 20000 / 5.5-C 8000）はここに入れない。
#    個別の不変条件（GATE_INVARIANTS）で持つ。
REQUIRED_FLAG_VALUES = {"--dstart": "20000"}
REQUIRED_FLAGS = {
    "train_gvoc.py": ["--front", "--prior", "--ch", "--layers", "--tag",
                      "--hfw", "--gainaug"],
}

# 撤回した主張。比較表・否定文の行だけ残してよい（下の否定マーカー参照）
RETRACTED = [
    (r"N\s*−\s*gcd\(io44,\s*N\)", "旧 accum 式"),
    (r"N\s*-\s*gcd\(io44i,\s*N\)", "旧 accum 式（検算スクリプト）"),
    (r"gcd\(io_host44,\s*N\)", "旧 accum 式（G.3.1）"),
    (r"`?gen`?\s*を同一\s*seed\s*で固定", "撤回済みの parity 仕様（実測 −3.53 dB）"),
    (r"履歴リング\s*768\s*sample", "リングは 2 本・入力 1792（2.4a-2）"),
    (r"往復\s*11\.61", "I/O の二重計上（accum は 1 回だけ）"),
    (r"32\.3〜35\.3", "撤回した E2E 算術"),
    (r"AD/DA\s*2〜5", "§7.2 の『計算 wall-clock 余裕 2–5ms』の読み違い"),
    (r"a\(diag\)\s*−\s*a\(pathA\)", "1.6-W の判定式は符号が逆だった"),
    (r"K\s*は.{0,8}暫定", "K = 2 は確定"),
    (r"唯一の道", "手順 6 候補 0 の方が確実（2.4b-1）"),
    (r"27\.41|27\.42", "queue 制約で不成立の構成"),
    (r"第\s*2\s*ヘッド", "HN3 は README の『新仮説』で採用ではない。R3 に F0 ヘッドの記述は無い"),
    (r"R3\s*(?:（HN3）)?\s*を?(?:遅延予算の)?(?:必須)?前提", "同上。遅延予算は causal_f0 の gather 融合を前提にする"),
    (r"front-end\s*(?:だけで|計)?\s*(?:RTF\s*)?0\.649", "測定対象を誤った値（リング全体の再計算）"),
    (r"増分\s*f0\s*にすれば直る」は成立しない(?!（実測）。ただし)", "融合すれば直る（ラダー 5・ビット一致 85% 削減）"),
]


def _argparse_flags(script: Path):
    """script の argparse から {flag: choices|None} を取る。"""
    try:
        tree = ast.parse(script.read_text(errors="ignore"))
    except SyntaxError:
        return None
    out: dict[str, set[str] | None] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        ch = None
        for kw in node.keywords:
            if kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                ch = {e.value for e in kw.value.elts
                      if isinstance(e, ast.Constant) and isinstance(e.value, str)}
        for a in node.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                out[a.value] = ch
    return out or None                                # argparse 不使用なら検査しない


def check_flags(doc: Path, text: str) -> None:
    """手順書のコマンドが使う --flag が、その script の argparse に実在するか。

    「存在しないフラグを手順書が指示する」事故が 2 巡で 5 件出た。
    これから足す予定のものは TO_BE_ADDED_FLAGS に明示させ、暗黙に通さない。
    """
    # 行継続（\\ 改行）を先に畳む。畳まないと複数行の学習起動が 1 つも検査されない
    joined = re.sub(r"\\\n\s*", " ", text)
    for m in re.finditer(r"(?:uv run )?python\s+([\w./-]+\.py)((?:\s+[^\n`|]*)?)", joined):
        name = Path(m.group(1)).name
        script = ROOT / "training" / name
        if not script.exists():
            continue
        have = _argparse_flags(script)
        if have is None:
            continue
        toks = m.group(2).split()
        # 学習を起動する行（--steps を持つ）は必須フラグの欠落も見る
        if "--steps" in toks:
            miss = [f for f in REQUIRED_FLAGS.get(name, []) if f not in toks]
            if miss:
                fails.append(f"{doc.name}: {name} の起動に必須フラグが無い -> {miss}"
                             f"（既定値で走ると別条件の学習になる）")
        for i, tok in enumerate(toks):
            if not re.fullmatch(r"--[a-z][\w-]*", tok):
                continue
            if tok in ("--help", "--version"):
                continue
            if tok not in have and (name, tok) not in TO_BE_ADDED_FLAGS:
                fails.append(f"{doc.name}: {name} に存在しない引数 -> {tok}"
                             f"（足す予定なら doc_check.TO_BE_ADDED_FLAGS に書く）")
                continue
            ch = have.get(tok)
            if not ch or i + 1 >= len(toks):
                continue
            val = toks[i + 1]
            if val.startswith("-") or "$" in val or "<" in val:
                continue
            if val not in ch and (name, tok, val) not in TO_BE_ADDED_CHOICES:
                fails.append(f"{doc.name}: {name} {tok} の choices に無い値 -> {val}"
                             f"（実体は {sorted(ch)}。足す予定なら TO_BE_ADDED_CHOICES に書く）")


def check_retracted_formulas(doc: Path, text: str) -> None:
    """撤回した式が、比較表以外の場所に残っていないか。

    「1 箇所だけ直して伝播しない」事故が accum / io / gen 固定で 3 回出た。
    比較表の行（`旧式` を含む行）だけは残してよい。
    """
    for pat, why in RETRACTED:
        for m in re.finditer(pat, text):
            line = text[text.rfind("\n", 0, m.start()) + 1: text.find("\n", m.end())]
            if any(k in line for k in ("旧式", "前版", "誤り", "書かない", "保証されない",
                                       "不足", "採らない", "使わない", "撤回", "не",
                                       "ではない", "しても", "不成立", "参考", "禁止", "しない",
                                       "rev", "新設したとき")):
                continue
            fails.append(f"{doc.name}: 撤回した式が残っている -> {why}")


def check_shell_vars(doc: Path, text: str) -> None:
    """bash ブロックで使う $VAR が、文書内のどこかで export されているか。

    `--hfw $HFW` を渡す起動行があるのに `export HFW=` が文書のどこにも無く、
    argparse が exit 2 で即死する事故が出た（nohup なのでログを開くまで気づかない）。
    ブロックを跨いだ export は正当なので、位置ではなく文書全体で見る。
    """
    known = {"RANDOM", "PATH", "HOME", "PWD", "PIPESTATUS", "CLOG", "IFS", "SHELL"}
    declared = set()
    for m in re.finditer(r"^\s*(?:export\s+)?((?:[A-Z_][A-Z0-9_]*=\S*\s*)+)", text, re.M):
        declared |= set(re.findall(r"([A-Z_][A-Z0-9_]*)=", m.group(1)))
    declared |= set(re.findall(r"\bfor\s+([A-Z_][A-Z0-9_]*)\s+in\b", text))
    used = set()
    for blk in re.findall(r"```bash\n(.*?)```", text, re.S):
        used |= set(re.findall(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", blk))
    miss = sorted(used - declared - known)
    if miss:
        fails.append(f"{doc.name}: 文書内に export が無い変数 -> {miss}"
                     f"（起動行が既定値で走るか argparse が exit 2 になる）")


# 2 文書が共有する量。片方だけ直して片方が腐る事故が巡ごとに出た
# ⚠ ここには**両文書に出る量だけ**を置く。片方にしか無い量は check_plan_numbers 側で
#    正本（plan_numbers）と突き合わせる。28 巡目まで 3 量が「2 文書比較」の顔で
#    単一文書しか見ておらず、しかも空回りの警告も出なかった。
SHARED = {
    "候補0 net 予算":   r"候補 0[^\n]{0,140}?net 予算 \*?\*?([0-9.]+)",
    "本命 E2E":        r"`?io ?= ?64`?[^\n]{0,24}`?q ?= ?2`?[^\n]{0,24}?([0-9]+\.[0-9]+)\s*ms",
    # ⚠ 状態数と参照 parity dB は 2 文書間の一致検査の対象外だった。
    #    APP が「6 状態 / +100.07 dB」（撤回済み）のまま取り残された（23 巡目）。
    "front-end 状態数":  r"([0-9]+) 状態を全部持",
    "参照 parity dB":   r"最悪 \+?([0-9]+\.[0-9]+) dB",
}
_shared_seen: dict[str, dict[str, set]] = {}


_shared_hit: set = set()


def check_shared(doc: Path, text: str) -> None:
    """2 文書が共有する量が食い違っていないか。

    ⚠ 28 巡目: 7 量のうち 3 量は PROCEDURE.md にしか一致せず、2 文書比較に
    なっていなかった（しかも空回りの WARN も出なかった）。
    """
    for name, pat in SHARED.items():
        vals = {next(g for g in m.groups() if g) for m in re.finditer(pat, text)
                if "候補 0" not in text[max(0, m.start() - 90):m.start()]
                or name.startswith("候補0")}
        if vals:
            _shared_seen.setdefault(name, {})[doc.name] = vals
    if doc.name != DOCS[-1].name:
        return
    for name in SHARED:
        per = _shared_seen.get(name, {})
        if not per:
            warns.append(f"共有量「{name}」がどちらの文書にも一致しない（検査が空回り）")
            continue
        if len(per) < 2:
            warns.append(f"共有量「{name}」が {list(per)[0]} にしか無い（2 文書比較になっていない）")
        allv = set().union(*per.values())
        if len(allv) > 1:
            fails.append(f"共有量「{name}」が {len(allv)} 通り -> "
                         + " / ".join(f"{d}:{sorted(v)}" for d, v in per.items()))


def check_counts(doc: Path, text: str) -> None:
    """宣言した個数が、直後の表・列挙の実数と合っているか。

    「見出し 17 修正 / 表 18 行」「6 箇所 / 列挙 7 ファイル」の類が巡ごとに出た。
    数え上げのずれは読み返しでは絶対に見つからない。
    """
    for m in re.finditer(r"^#+ .*?の\s*\**(\d+)\s*修正", text, re.M):
        rest = text[m.end():]
        end = rest.find("\n#")
        rows = len([l for l in rest[: end if end > 0 else len(rest)].split("\n")
                    if re.match(r"^\|\s*\**[a-z0-9-]+\**\s*\|", l)
                    and not re.match(r"^\|[\s:-]+\|", l)])
        if rows and rows != int(m.group(1)):
            fails.append(f"{doc.name}: 見出しの「{m.group(1)} 修正」と表の {rows} 行が不一致")

    for m in re.finditer(r"\*\*(\d+)\s*ファイル(?:・\d+\s*箇所)?\*\*", text):
        seg = text[m.end(): m.end() + 500].split("\n")[0]
        names = re.findall(r"`([\w./-]+\.(?:py|rs))`", seg)
        uniq = {Path(x).name for x in names}
        if uniq and len(uniq) != int(m.group(1)):
            fails.append(f"{doc.name}: 「{m.group(1)} ファイル」と列挙 {len(uniq)} 件が不一致"
                         f" -> {sorted(uniq)}")


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


def main() -> None:
    acquire_lock()
    collect_heads()
    for d in DOCS:
        if not d.exists():
            fails.append(f"{d} が存在しない")
            continue
        t = d.read_text(errors="ignore")
        for fn in (check_paths, check_commands, check_consts, check_cd,
                   check_python_blocks, check_sections, check_stale,
                   check_fences, check_tables, check_required, check_dup_heads, check_size,
                   check_flags, check_retracted_formulas, check_counts,
                   check_shell_vars, check_shared, check_derived, check_k_constraint, check_plan_numbers,
                   check_code_claims, check_ladder, check_bash_blocks, check_ci_exists, check_orphan_rows, check_thresholds, check_net_stale, check_solution_counts, check_rtf_verdict, check_cliff, check_json_blocks, check_table_cols, check_tee_pipefail, check_formula_sync, check_key_set, check_retracted_framing, check_conventions, check_stage_evidence, check_gate_invariants, check_flag_values, check_export_consts, check_number_provenance,
                   check_gate_fixtures, check_prohibitions, check_pins,
                   check_threshold_provenance):
            # ⚠ `--content-only` は pin 照合だけ外す（`pins.py --bless` の前提検査）。
            #    内容が壊れたまま bless すると変異が正本に焼き込まれる（36 巡目の事故）。
            if "--content-only" in sys.argv and fn is check_pins:
                continue
            fn(d, t)

    print(f"\n  検査対象: {', '.join(d.name for d in DOCS)}")
    print(f"  FAIL {len(fails)} 件 / WARN {len(warns)} 件\n")
    for f in fails:
        print(f"  [FAIL] {f}")
    for w in warns:
        print(f"  [WARN] {w}")
    print()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
