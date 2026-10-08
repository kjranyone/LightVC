"""盲検A/B試行への代理指標予測の事前登録(耳回答の前に実行・未登録試行のみ追記)。

comb_db(調波鋭さ)は大きいほどコーラス少の予測だが過周期(ロボット)でも上がるため、
錨(gt_decode)からの|Δcomb|が小さい順=錨に近い順の予測も併記する。period_nccは参考。

    uv run python ab_proxy_predict.py   # results/earbattery/d1_ab/_proxy_prediction_聴取後に開く.json
"""
from __future__ import annotations

import json
from pathlib import Path

import chorus_proxy as cp

ROOT = Path(__file__).resolve().parent.parent
AB = ROOT / "results/earbattery/d1_ab"


def main() -> int:
    key = json.loads((AB / "_key_聴取後に開く.json").read_text())
    pp = AB / "_proxy_prediction_聴取後に開く.json"
    pred = json.loads(pp.read_text()) if pp.exists() else {"trials": {}}
    for trial, info in sorted(key.items()):
        if trial in pred["trials"]:
            continue
        ms = {L: {"sys": nm, **cp.measure(cp.load48(AB / trial / f"{L}.wav"))}
              for L, nm in info["map"].items()}
        gt = next(L for L, m in ms.items() if m["sys"] == "gt_decode")
        others = [L for L in ms if L != gt]
        pred["trials"][trial] = {
            "measures": ms,
            "pred_least_chorus_first_by_comb": ">".join(sorted(ms, key=lambda L: -ms[L]["comb_db"])),
            "pred_closest_to_anchor_by_abs_dcomb": ">".join(
                sorted(others, key=lambda L: abs(ms[L]["comb_db"] - ms[gt]["comb_db"]))),
            "dcomb_vs_anchor": {L: (ms[L]["sys"].split("[")[0],
                                    round(ms[L]["comb_db"] - ms[gt]["comb_db"], 2)) for L in others}}
        print(trial, pred["trials"][trial]["dcomb_vs_anchor"], flush=True)
    pp.write_text(json.dumps(pred, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
