"""腕 64 の合成の男声のコーパスの文: 生成器の学習の話者(train_ppgvoc.load_rows の実女声)の female-dataset の書き起こし(.lab)から、普通の台詞を選ぶ。
評価・保留・開発の話者の書き起こしは使わない(学習の話者だけ)。出力 = data/male_tts_v4l_train/texts.json。
    uv run python prep_male_train_texts.py --n 4320
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import train_ppgvoc as TP

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/male_tts_v4l_train"
NG = re.compile(r"ちゅ|ぢゅ|じゅる|じゅぷ|じゅぽ|れろ|ぴちゃ|くちゅ|はぁ|はあっ|んっ|あっ|あぁ|ん[ー〜～]|[♡♥]|[a-zA-Zａ-ｚＡ-Ｚ]")
KANJI = re.compile(r"[一-鿿]")
STRIP = str.maketrans("", "", "（）()「」『』【】〈〉《》\"' 　\n\t")


def clean(s: str) -> str | None:
    s = s.translate(STRIP)
    if not 22 <= len(s) <= 70 or NG.search(s) or len(KANJI.findall(s)) < 3:
        return None
    if sum(c in "…ー〜～っッ・" for c in s) / len(s) >= 0.12:
        return None
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4320)
    a = ap.parse_args()
    rows, spks, _ = TP.load_rows({h[0] for h in TP.e2e_held()}, True)
    files = sorted({r[0] for r in rows})
    seen, pool = set(), []
    for w in files:
        lab = Path(w).with_suffix(".lab")
        if not lab.exists():
            continue
        s = clean(lab.read_text(encoding="utf-8"))
        if s and s not in seen:
            seen.add(s)
            pool.append({"text": s, "from": Path(w).stem})
    random.Random(691).shuffle(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"train_spk": len(spks), "train_files": len(files), "eligible": len(pool), "picked": min(a.n, len(pool))}
    (OUT / "texts.json").write_text(json.dumps({"rule": "prep_male_train_texts.py", **rep, "texts": pool[: a.n]}, ensure_ascii=False, indent=1))
    print(rep, flush=True)
    for t in pool[:12]:
        print(t["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
