"""最終モデルの選択(規則は results/ddsp_vc2r/prereg.yaml の selection_rule_fixed_before_eval で評価前に固定)。

候補の eval.json(eval_ddsp_vc.py の出力)を読み、英 CER 増分 ≤ 0.15 かつ 目標 ECAPA > 元話者 ECAPA を満たすものの中で目標 ECAPA 最大を選ぶ。
同じ変換評価セットで選ぶので楽観的な偏りがある(耳の A/B で最終判定)。

    uv run python select_ddsp_final.py name=path/to/eval.json ...
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    rows = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        c = json.load(open(path))["conv"]
        ok = c["cer_increase_median_en"] <= 0.15 and c["secs_tgt_mean"] > c["secs_output_vs_source_speaker_mean"]
        rows.append({"name": name, "eval": path, "secs_tgt": c["secs_tgt_mean"], "secs_srcspk": c["secs_output_vs_source_speaker_mean"],
                     "cer_inc_en": c["cer_increase_median_en"], "eligible": ok})
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    el = [r for r in rows if r["eligible"]]
    best = max(el, key=lambda r: r["secs_tgt"]) if el else None
    print("SELECTED", json.dumps(best, ensure_ascii=False))
    return 0 if best else 1


if __name__ == "__main__":
    raise SystemExit(main())
