"""盲検A/B回答の開鍵集計(回答保存後に実行)。

入力: results/earbattery/d1_ab/answers_ear.json(listen.htmlの保存)・_key_聴取後に開く.json・
      _proxy_prediction_聴取後に開く.json。出力: 同dirの ear_result.json と要約表示。
各試行: 系ごとの順位(1=コーラス最少)・錨(gt_decode)の順位(聴取の信頼性チェック)。
対ごとの勝敗: ARの d1_g1full / d1_g0 と各比較相手(コーラス順位・総合順位)。
代理予測との対応: 系の対ごとの順序一致率(comb降順・錨からの|Δcomb|昇順)。

    uv run python score_d1_ab.py
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "results/earbattery/d1_ab"
AR = ("d1_g1full", "d1_g0")


def sysname(s: str) -> str:
    return s.split("[")[0]


def main() -> int:
    ans = json.loads((AB / "answers_ear.json").read_text())["trials"]
    key = json.loads((AB / "_key_聴取後に開く.json").read_text())
    pred = json.loads((AB / "_proxy_prediction_聴取後に開く.json").read_text())["trials"]
    out: dict = {"trials": {}, "pairs": {}, "anchor": {}, "proxy_agreement": {}}
    for t in sorted(key):
        a = ans.get(t) or {}
        m = key[t]["map"]
        row = {}
        for kind in ("chorus", "overall"):
            r = a.get(kind) or {}
            if sorted(v for v in r.values() if v) != list(range(1, len(m) + 1)):
                continue
            row[kind] = {sysname(m[L]): int(r[L]) for L in m}
        if not row:
            continue
        row["note"] = a.get("note", "")
        out["trials"][t] = row
        s = t.split("_", 1)[0]
        if "chorus" in row:
            out["anchor"].setdefault(s, []).append(row["chorus"]["gt_decode"])
        for kind in ("chorus", "overall"):
            if kind not in row:
                continue
            rk = row[kind]
            for x in AR:
                if x not in rk:
                    continue
                for y in rk:
                    if y in (x, "gt_decode"):
                        continue
                    k = f"{s}|{kind}|{x} vs {y}"
                    w = out["pairs"].setdefault(k, {"AR_better": 0, "other_better": 0})
                    w["AR_better" if rk[x] < rk[y] else "other_better"] += 1
        if "chorus" in row and t in pred:
            ms = pred[t]["measures"]
            comb = {sysname(v["sys"]): v["comb_db"] for v in ms.values()}
            g = comb["gt_decode"]
            ear = row["chorus"]
            for nm, score in (("comb_desc", lambda z: -comb[z]), ("abs_dcomb_to_anchor", lambda z: abs(comb[z] - g))):
                agree = tot = 0
                for x, y in combinations([z for z in ear if z != "gt_decode"], 2):
                    tot += 1
                    agree += (ear[x] < ear[y]) == (score(x) < score(y))
                d = out["proxy_agreement"].setdefault(nm, {"agree": 0, "total": 0})
                d["agree"] += agree
                d["total"] += tot
    (AB / "ear_result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("試行別(コーラス順位 1=最少):")
    for t, r in out["trials"].items():
        print(f"  {t}: {r.get('chorus')}  総合 {r.get('overall')}  {r.get('note', '')}")
    print("錨の順位(1なら聴取は錨を最もクリーンと判定):", out["anchor"])
    print("AR対比較相手:")
    for k, v in sorted(out["pairs"].items()):
        print(f"  {k}: AR勝ち {v['AR_better']} / 相手勝ち {v['other_better']}")
    print("代理予測との対順序一致:", out["proxy_agreement"])
    print("->", AB / "ear_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
