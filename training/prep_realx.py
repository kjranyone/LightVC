"""D177: 実女声を交差ペアの目標側に置くための擬似パラレル(diag_rr8c 用)。data/rr_feat/real_fem の話者(held・評価 / 開発のプールは prep_rr で除外済み)ごとに
普通の台詞(d165_real_parallel と同じ字の条件)を 1 本選び、その文を男声 TTS(学習用 m030〜m045・検証用の実女声には m046〜m047)に読ませる項目を作る。
目標 = 実女声のそのクリップ(伸縮しない実音声)・入力 = 男声 TTS の読み(C1 で目標の時間へ整列 = 入力側だけ)・記憶 = 同じ話者の別のクリップ。
    uv run python prep_realx.py --out data/realx/items.json
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_male_for_real.py --items ../LightVC/data/realx/items.json --out ../LightVC/data/realx/wav
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TRAIN_M = [f"m{i:03d}" for i in range(30, 46)]
VAL_M = ["m046", "m047"]


def clean(lab: Path) -> str | None:
    text = re.sub(r"[「」『』（）()\s]", "", unicodedata.normalize("NFKC", lab.read_text(errors="ignore")).strip())
    core = re.sub(r"[…ー~!?・、。,.―-]", "", text)
    if not (6 <= len(core) <= 60) or re.search(r"(.)\1\1", core) or sum(ch in "ぁぃぅぇぉっッ" for ch in core) > 0.2 * len(core):
        return None
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=1)
    a = ap.parse_args()
    man = json.loads((ROOT / "data/rr_feat/real_fem/manifest.json").read_text())["items"]
    by: dict = {}
    for it in man:
        by.setdefault(it["spk"], []).append(it)
    reals = sorted(s for s, v in by.items() if len(v) >= 8)
    val = set(reals[::20])
    rng = random.Random(177)
    items = []
    for si, s in enumerate(reals):
        got = 0
        for it in by[s]:
            if got >= a.per:
                break
            w = ROOT / it["wav"]
            lab = w.with_suffix(".lab")
            if not lab.exists() or not (1.2 <= sf.info(str(w)).duration <= 12.0):
                continue
            text = clean(lab)
            if text is None:
                continue
            items.append({"item": len(items), "ti": si, "target": s, "stem": it["stem"], "spk": rng.choice(VAL_M if s in val else TRAIN_M), "clip": str(w), "text": text, "val": s in val})
            got += 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print("items", len(items), "speakers", len({i["target"] for i in items}), "val", sum(i["val"] for i in items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
