"""検査資産そのものを sha256 で凍結する。変更は `--bless` を明示させる。

34 巡目の測定: 176 変異中 142 素通り。素通りは「守りが 1 つも無い領域」に集中し、
**ハーネス自身の 1 行削除が 20/22 素通り**した（検査を足して守ってきた 33 巡ぶんの資産が、
検査ファイル側を 1 行消すだけで無音のまま失われる）。しかも壊れた状態ほど健全に見える。

∴ 「守る対象を列挙する」のではなく、**資産の同一性を凍結する**。
新しい検査を足したときだけ `--bless` が要る。それ以外の変更はすべて FAIL。

    uv run python pins.py            # 照合（doc_check が先頭で呼ぶ）
    uv run python pins.py --bless    # 現状を新しい正本にする
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIN = ROOT / "results/z0/harness_pins.json"

HARNESS = ["doc_check.py", "gate_run.py", "gate_mutate.py", "mutate_check.py",
           "latency_gate.py", "pre_launch_check.py", "plan_numbers.py", "pins.py"]

# CLAUDE.md 由来の禁止・過去の事故の記憶。**段落ごと**固定する
# （34 巡目: needle は「ただし〜してよい」の追記や、90 字の外へ逃がすので抜けた）。
# 判定に効く「節」を段落ごと固定する。34/35 巡目の実測: 素通り 73 件のうち
# 72 件が手順書テキストで、内訳は判定条件（doc-cond）・閾値（doc-threshold）・
# 関門（doc-gate）・禁止（doc-prohibition）。**needle でも語ホワイトリストでも
# 一般化しない**ので、段落の同一性そのものを凍結する。
DECISION_ANCHORS = [
    ("PROCEDURE.md", "**Nv ≥ 4 を必須**"),
    ("PROCEDURE.md", "**a ≥ Nv/2**"),
    ("PROCEDURE.md", "再レンダは合計 2 回まで"),
    ("PROCEDURE.md", "1 候補あたりの上限は GPU 時間で数える: 120 h"),
    ("PROCEDURE.md", "**実施できた腕が 2 本未満なら族の否定を出さない**"),
    ("PROCEDURE.md", "ceiling_decision.json` から `ref` / `speakers` を引く"),
    ("PROCEDURE.md", "決定はファイルから引く。決め打ちしない"),
    ("PROCEDURE.md", "検収 3 は独立標本 2 つで見る"),
    ("PROCEDURE.md", "bash pre_launch_gate.sh"),
    ("PROCEDURE.md", "0.265"),
    ("PROCEDURE.md", "**⚠ 正規近似（1.96·sd/√n）を当てない**"),
    ("PROCEDURE.md", "MIN_PROBES"),
    ("PROCEDURE.md", "1 時間を超える学習"),
    ("PROCEDURE_APP.md", "**≥ 50 ms**"),
    ("PROCEDURE_APP.md", "最大絶対誤差 ≤ 1e-4"),
    ("PROCEDURE_APP.md", "±2 サンプル以内で一致する"),
    ("PROCEDURE_APP.md", "provisional のまま手順 F / G に入らない"),
]

# 35 巡目: 回帰スイート 279 件の素通り 48 件はすべて「散文で書かれた判定」だった。
# 過去のレビューが実際に狙った箇所を**一意アンカー**として機械的に抽出し、段落ごと固定する。
# 35 巡目: 回帰スイート 279 件の素通り 48 件はすべて「散文で書かれた判定」だった。
# 過去のレビューが狙った箇所を**一意アンカー**として抽出し、段落ごと固定する。
# ⚠ アンカーはコードに埋めず外部 JSON に持つ。改行やクォートを含むので、
#    埋め込むと構文を壊す（35 巡目に実際に壊した）。
_AA = ROOT / "results/z0/decision_anchors.json"
AUTO_ANCHORS = [tuple(x) for x in json.loads(_AA.read_text())] if _AA.exists() else []

PROHIBITION_ANCHORS = [
    ("PROCEDURE.md", "推論経路に発話全体の統計を置かない"),
    ("PROCEDURE.md", "昇格は耳のみ"),
    ("PROCEDURE.md", "白色雑音で測らない"),
    ("PROCEDURE.md", "同じ TAG で再走行しない"),
    ("PROCEDURE.md", "自由位相にしない"),
    ("PROCEDURE.md", "V に時間軸をまたぐ正規化層を置かない"),
    ("PROCEDURE.md", "PESQ 単独で昇格させない"),
    ("PROCEDURE_APP.md", "rtf_*.json` でグロブしない"),
]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _paragraph(text: str, anchor: str) -> str | None:
    """anchor を含む段落（空行で区切られた塊）を返す。

    ⚠ anchor 自体が消えたら `None` を返し、呼び出し側が "MISSING" として
    pin と食い違わせる。35 巡目: 段落**全体**が削除される変異（T104:
    「PESQ 単独で昇格させない」の 2 行削除）で、段落 sha が別の段落に
    すり替わって素通りしていた。
    """
    i = text.find(anchor)
    if i < 0:
        return None
    s = text.rfind("\n\n", 0, i)
    e = text.find("\n\n", i)
    return text[(s + 2 if s >= 0 else 0): (e if e >= 0 else len(text))]


def _counts() -> dict:
    """検査の本数。1 行消しても気づけるようにする。"""
    sys.path.insert(0, str(ROOT / "training"))
    import doc_check as DC
    import gate_mutate as GM
    out = {"GATE_INVARIANTS": len(DC.GATE_INVARIANTS),
           "THRESHOLDS": len(DC.THRESHOLDS),
           "PLAN_PATTERNS": len(DC.PLAN_PATTERNS),
           "BANNED_PHRASES": len(DC.BANNED_PHRASES),
           "PROHIBITIONS": len(DC.PROHIBITIONS),
           "MUTATIONS": len(GM.MUTATIONS),
           "PL_MUTATIONS": len(GM.PL_MUTATIONS),
           "FIXTURE_MUTATIONS": len(GM.FIXTURE_MUTATIONS),
           "DOCS": len(DC.DOCS)}
    for d, r in DC.REQUIRED.items():
        out[f"REQUIRED[{d}]"] = len(r)
    return out


# 関門が読む正本 artifact。**手順書の雛形だけ守っても、実物を書き換えれば通る**
# （35 巡目: H10「target_platform に未実測 RTF を書き込む」が素通りした）。
ARTIFACTS = ["target_platform.json", "abi.json", "ceiling_decision.json"]


def snapshot() -> dict:
    cur: dict = {"harness": {}, "fixtures": {}, "prohibitions": {},
                 "artifacts": {}, "counts": _counts()}
    for a in ARTIFACTS:
        f = ROOT / "results/z0" / a
        cur["artifacts"][a] = _sha(f.read_bytes()) if f.exists() else "MISSING"
    for f in HARNESS:
        p = ROOT / "training" / f
        cur["harness"][f] = _sha(p.read_bytes()) if p.exists() else "MISSING"
    # ⚠ 再帰。34 巡目: 直下の *.json しか見ておらず、pl_fixtures/z_*/abi.json を
    #    書き換え放題だった。
    for d in ("fixtures", "pl_fixtures"):
        for p in sorted((ROOT / "results/z0" / d).rglob("*")):
            if p.is_file():
                cur["fixtures"][f"{d}/{p.relative_to(ROOT / 'results/z0' / d)}"] = \
                    _sha(p.read_bytes())
    for doc, anchor in PROHIBITION_ANCHORS + DECISION_ANCHORS + AUTO_ANCHORS:
        t = (ROOT / "current" / doc).read_text()
        n = t.count(anchor)
        para = _paragraph(t, anchor)
        # ⚠ 出現数も一緒に固定する。35 巡目: アンカーが複数箇所にあると、
        #    片方（判定を書いた段落）を丸ごと削除しても別の出現が拾われて
        #    素通りした（T104: PESQ 単独昇格禁止の 2 行削除）。
        cur["prohibitions"][f"{doc}::{anchor}"] = (
            f"{n}:{_sha(para.encode())}" if para else "MISSING")
    return cur


def verify() -> list[str]:
    if not PIN.exists():
        return ["results/z0/harness_pins.json が無い。検査資産の同一性が担保されない"
                "（意図して作るなら pins.py --bless）"]
    old = json.loads(PIN.read_text())
    cur = snapshot()
    out = []
    for sec in ("harness", "fixtures", "prohibitions", "artifacts", "counts"):
        o, c = old.get(sec, {}), cur.get(sec, {})
        for k in sorted(set(o) | set(c)):
            if o.get(k) != c.get(k):
                kind = "削除" if k not in c else "追加" if k not in o else "変更"
                out.append(f"{sec}: {k} が{kind}された"
                           f"（{o.get(k)} -> {c.get(k)}）——意図した変更なら pins.py --bless")
    return out


def _content_ok() -> list[str]:
    """内容検査（pin 以外）が通っているか。**bless の前提**。

    ⚠ 36 巡目の事故: SIGKILL された `mutate_check` が変異を書き戻せず
    ディスクに残し、その状態で `--bless` したので**変異が正本に焼き込まれた**
    （5.1 の「未来不変性 ネット込み」行が「ネットは見ない」に反転したまま固定）。
    pin は「変わったこと」しか見ないので、**壊れた状態を新しい正しさにできてしまう**。
    ∴ bless は内容検査に従属させる。
    """
    import subprocess
    r = subprocess.run(["uv", "run", "python", "doc_check.py", "--content-only"],
                       capture_output=True, text=True,
                       cwd=ROOT / "training", timeout=1800,
                       env=dict(__import__("os").environ,
                                DOC_HARNESS_LOCK_HELD="1"))
    return [] if r.returncode == 0 else [l for l in r.stdout.splitlines()
                                         if "[FAIL]" in l]


def main() -> int:
    if "--bless" in sys.argv:
        if "--force" not in sys.argv:
            bad = _content_ok()
            if bad:
                print("  ⚠ 内容検査が FAIL している状態で bless しない"
                      "（変異を正本に焼き込む）。先に直す:")
                for b in bad[:6]:
                    print(f"   {b.strip()[:140]}")
                print("  どうしても焼くなら --force（理由を残すこと）")
                return 1
        PIN.parent.mkdir(parents=True, exist_ok=True)
        PIN.write_text(json.dumps(snapshot(), ensure_ascii=False, indent=1))
        print(f"  pin を更新した -> {PIN}")
        return 0
    bad = verify()
    for b in bad:
        print(f"  ⚠ {b}")
    print(f"  pin 不一致 {len(bad)} 件")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
