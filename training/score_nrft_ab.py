"""ys1_nrft 盲検A/Bの開鍵集計(results/earbattery/nrft_ab/answers_ear.json 保存後に実行)。

各試行の系 {source, gt_orig, gt_ft, s11_orig, s11_ft} の順位(1=コーラス最少/最も好ましい)。
主判定(prereg): s11_ft が s11_orig よりコーラスが少ない試行数(3中2以上)。
非劣化: gt_ft が gt_orig より総合で劣る試行数。錨: source の順位。

    uv run python score_nrft_ab.py
"""
from __future__ import annotations

import json
from pathlib import Path

AB = Path(__file__).resolve().parent.parent / "results/earbattery/nrft_ab"


def main() -> int:
    ans = json.loads((AB / "answers_ear.json").read_text())["trials"]
    key = json.loads((AB / "_key_聴取後に開く.json").read_text())
    out: dict = {"trials": {}, "counts": {}}
    c = {"s11_ft_less_chorus": 0, "s11_ft_better_overall": 0, "gt_ft_worse_overall": 0,
         "gt_ft_more_chorus": 0, "n": 0}
    for t, info in sorted(key.items()):
        a = ans.get(t) or {}
        rk = {}
        for kind in ("chorus", "overall"):
            r = a.get(kind) or {}
            if sorted(v for v in r.values() if v) == list(range(1, len(info["map"]) + 1)):
                rk[kind] = {info["map"][L]: int(r[L]) for L in info["map"]}
        if len(rk) < 2:
            continue
        c["n"] += 1
        c["s11_ft_less_chorus"] += rk["chorus"]["s11_ft"] < rk["chorus"]["s11_orig"]
        c["s11_ft_better_overall"] += rk["overall"]["s11_ft"] < rk["overall"]["s11_orig"]
        c["gt_ft_worse_overall"] += rk["overall"]["gt_ft"] > rk["overall"]["gt_orig"]
        c["gt_ft_more_chorus"] += rk["chorus"]["gt_ft"] > rk["chorus"]["gt_orig"]
        out["trials"][t] = {**rk, "note": a.get("note", "")}
        print(f"  {t}: コーラス {rk['chorus']}  総合 {rk['overall']}  {a.get('note', '')}")
    out["counts"] = c
    out["primary_PASS"] = c["n"] >= 3 and c["s11_ft_less_chorus"] >= 2
    (AB / "ear_result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("集計:", json.dumps(c, ensure_ascii=False), "主判定PASS" if out["primary_PASS"] else "主判定FAIL/未達")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
