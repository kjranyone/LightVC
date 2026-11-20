"""paper_snv §7 rev44 の成績表を作る: training/snv_dev_eval.py の出力(results/snv_d5/dev_rows_devA.json・dev_rows_devB.json と、段ごとの dev_rows)の試行から、
有効な源(元の音の書き起こしが読み上げの文と合う・text_vs_orig ≤ 0.1)の列と全部の源の列を計算して results/LEADERBOARD.md に書く。
    cd training && uv run --no-sync python snv_leaderboard.py [rows=results/<段>/dev_rows_devA.json,...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VALID = 0.1
FILES = ["results/snv_d5/dev_rows_devA.json", "results/snv_i25/dev_rows_devA.json", "results/snv_p16/dev_rows_devA.json", "results/snv_p19/dev_rows_devA.json", "results/snv_p20/dev_rows_devA.json", "results/snv_p21/dev_rows_devA.json", "results/snv_p24c/dev_rows_devA.json", "results/snv_p26c/dev_rows_devA.json", "results/snv_p28c/dev_rows_devA.json", "results/snv_p30c/dev_rows_devA.json", "results/snv_p31c/dev_rows_devA.json", "results/snv_p34c/dev_rows_devA.json", "results/snv_d5/dev_rows_devB.json", "results/snv_p30c/dev_rows_devB.json", "results/snv_p31c/dev_rows_devB.json", "results/snv_p16/dev_rows_devB.json",
         "results/snv_p37/dev_rows_devA_rewav.json", "results/snv_p39/dev_rows_devA_s1_rewav.json", "results/snv_p38/dev_rows_devA_gate.json", "results/snv_p39/dev_rows_devA_s1_gate.json",
         "results/snv_p39/dev_rows_devB_s0_rewav.json", "results/snv_p39/dev_rows_devB_s1_rewav.json", "results/snv_p39/dev_rows_devB_s0_gate.json", "results/snv_p39/dev_rows_devB_s1_gate.json",
         "results/snv_p40/dev_rows_devA_t80.json", "results/snv_p42/dev_rows_devA_s1_t80.json",
         "results/snv_p45/dev_rows_devA_t80.json"]
NOTE = {"orig": "元の音そのもの(物差しの上限)", "copy": "写しの描画(前段 v3 + rvoc3hi・変換なし)", "anchor": "信号処理の一律のずらしの錨(P0c・声は変わらない)", "snv_p3_x": "P3(CPC の演奏)",
        "snv_p8_wst": "P8(操作者の統計で白くした演奏)", "snv_i10_c_pr": "c_pr(段 I8 の E + M15 の C)", "snv_i10_c_pr+mask:snv_i24_gate_hyst": "c_pr + 持続の門(段 I24・c_pr の行は保存の特徴で、解析し直した対照ではない)",
        "snv_i12_combo": "診断: 声と録音の族の E + M15 の C(段 I12)", "snv_i10_c_pr+vtl": "診断: c_pr + 登録での伸縮の当てはめ(M17・製品の候補でない)", "snv_i25_combo": "段 I25: 白くした演奏 → 軌跡の E(M28)+ M15 の C",
        "snv_p16_x": "段 P16: P8 の型 + 源の側を擬似の操作者に替えた自己の reamp(M29)",
        "snv_p19_x": "段 P19: P16 + 演奏の式を PCA 32・遅れなし・縮み 0.6 に", "snv_p20_q": "段 P20: P19 の式 + 番地だけの回路",
        "snv_p21_x": "段 P21: P19 + 内容ごとの平均の正規化(級 128)",
        "snv_p24_c": "段 P24: P21 の符号を、擬似の操作者の視点を足して学び直したものに(M30)",
        "snv_p26_c1": "段 P26: snv_p24_c の再現(学習の種 1)", "snv_p28_c2": "段 P28: snv_p24_c + 構造の読みの広がりの帯(参照の 1〜4 %)と問いで平均した広がりの下限(M32・修正 1)",
        "snv_p30_c": "段 P30: snv_p24_c の符号を、同じ文を読む別の声の揃えた位置を正例に足して学んだもの(M33・snv_p30_dcpc)に", "snv_p31_c1": "段 P31: snv_p30_c の再現(学習の種 1)", "snv_p34_c": "段 P34: snv_p30_c の演奏の式に、登録の統計で標準化した強さを明示の量として足したもの(M34)",
        "snv_p30_c+rewav": "対照: snv_p30_c を、同じ wav から解析し直した特徴で(前段つきの系はこの行と比べる・paper §7 rev98)", "snv_p31_c1+rewav": "対照: snv_p31_c1 を、同じ wav から解析し直した特徴で",
        "snv_p30_c+mask:snv_p38_gate": "段 P39: snv_p30_c + 学んだ前段(門 + M36 の重み + 持続)・devB の確認で足さない入力の代価が示され、候補にしていない", "snv_p31_c1+mask:snv_p38_gate": "段 P39: snv_p31_c1 + 学んだ前段(種 1)",
        "snv_p30_c+mask:snv_p40_gate_t80": "段 P42: snv_p30_c + 学んだ前段(門の閾 0.8)・devB の未使用の話者で伴奏 10 dB の利得が再現せず、候補にしていない", "snv_p31_c1+mask:snv_p40_gate_t80": "段 P42: snv_p31_c1 + 学んだ前段(門の閾 0.8・種 1)",
        "snv_p30_c+mask:snv_p45_gate_t80": "段 P45: snv_p30_c + 学んだ前段(重みの予算 60k・門の閾 0.8)= 研究の形(段 P46 で数値の関門へ)", "snv_p31_c1+mask:snv_p45_gate_t80": "段 P46: snv_p31_c1 + 研究の形の前段(種 1)"}
HEAD = """# 成績表(current/paper_snv.md §7 rev44・current/ml_method.md §4)

最適化指標 = **開発の組の有効な源の、本物の男声 → A0F の女声の CER の中央値**(large-v3 の仮名・参照 = 源の元の音の書き起こし・k 0 と k 7・低いほど良い)。
有効な源 = 元の音の書き起こしが Common Voice の読み上げの文と合う源(仮名の CER ≤ 0.1・系に依らない・devA 105 / 128・devB 56 / 64)。合わない源は元の音で既に認識が崩れている(物差しの外)。全部の源の値は並べて残す。
開発の組 = Common Voice 22.0 ja の男声(CC0・devA の先頭 64 人 × 2 発話 × 目標 2 声 × k 2 / devB の先頭 32 人・`training/snv_dev_set.py`・`training/snv_dev_eval.py`)。JVNV の男声 2 人(48 試行)はきれいなスタジオの録音の参考。
目標の割り当ては話者の番号で決まるので、目標ごとの値は話者の組と交絡する(目標ごとに読むときは `results/snv_p15/rows.json`)。
試験の 12 区間(外郎売・namikawa)は最終判定だけに使う(製品の線 = 試験の平均 CER ≤ 0.275)。devB は段の候補の確認だけに使う。
満足指標 = 女声 → 女声の写し ≤ 0.15・M の g ≥ 0.25・同じ声の自己の reamp ≤ 0.05・Irodori どうしの男 → 女 ≤ 0.115・S ≥ 0.70・出荷の関門。進捗 = 区間(話者で束ねた 95 %)を超える改善。
リセット前の成績表は `results/LEADERBOARD_pre_reset.md`(撤回された記録)。

本人らしさ M = 出力の ECAPA + LDA が A0F の 24 声の中で目標を 1 位に当てる率(g = 試験の床 0.061 と天井 0.635 で測った割合)。CER だけが低い系(写し・錨・P3)は声を変えていない。

| 組 | 構成 | 有効な源の CER の中央値 [95 %] | k 0 / k 7 | CER ≤ 0.2 の率 | M(g) | 全部の源の中央値 | UTMOS | JVNV の中央値 | 証拠 |
|---|---|---|---|---|---|---|---|---|---|
"""


def ci(rows: list[dict]) -> list[float]:
    spk = sorted({r["spk"] for r in rows})
    per = {s: [r["cer"] for r in rows if r["spk"] == s] for s in spk}
    rng = np.random.default_rng(0)
    bs = [float(np.median(np.concatenate([per[s] for s in rng.choice(spk, len(spk))]))) for _ in range(1000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    lines = []
    for f in FILES + [x for x in opt.get("rows", "").split(",") if x]:
        p = ROOT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ref = d["ref"] or json.loads((ROOT / f"results/snv_d5/dev_rows_{d['set']}.json").read_text())["ref"]
        for tag, rows in d["rows"].items():
            R = [r for r in rows if r["group"] == d["set"]]
            V = [r for r in R if ref[r["src"]]["text_vs_orig"] <= VALID]
            if not V:
                continue
            J = [r["cer"] for r in rows if r["group"] == "jvnv"]
            lo, hi = ci(V)
            k = " / ".join(f"{np.median([r['cer'] for r in V if r['k'] == x]):.3f}" for x in (0, 7) if any(r["k"] == x for r in V))
            M = float(np.mean([r["m"] for r in V])) if "m" in V[0] else None
            m = f"{M:.2f}({(M - 0.061) / (0.635 - 0.061):.2f})" if M is not None else "—"
            lines.append(f"| {d['set']} | {NOTE.get(tag, tag)} `{tag}` | {np.median([r['cer'] for r in V]):.3f} [{lo:.3f}, {hi:.3f}] | {k} | {np.mean([r['cer'] <= 0.2 for r in V]):.2f} | {m} | {np.median([r['cer'] for r in R]):.3f} | {np.mean([r['utmos'] for r in V]):.2f} | "
                         f"{(f'{np.median(J):.3f}' if J else '—')} | {f} |")
    (ROOT / "results/LEADERBOARD.md").write_text(HEAD + "\n".join(lines) + "\n")
    print(HEAD + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
