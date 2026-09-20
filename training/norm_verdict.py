"""正規化モード対照の判定。**曲線の 1 点で決めない。**

36 巡目の事故: `wavehax_ctrl` の step 2000（−0.069）だけを見て
「建築の差ではない」と結論し、20000 step（+0.0912）で反転して撤回した。
∴ 判定は **(a) 終端値 (b) 後半の傾き (c) 反転の有無** の 3 つで読む。

    uv run python norm_verdict.py
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARMS = {
    "wavehax(製品不可・基準)": "diag_wavehax_ctrl",
    "v2f norm=freq": "diag_v2f16_cplx1",
    "v2f norm=none": "diag_v2f16_none",
    "v2f norm=ema": "diag_v2f16_ema",
    "v2f norm=fixed": "diag_v2f16_fixed",
}
PAT = re.compile(r"^  step\s+(\d+)\s+loss\s+(\S+)\s+TEST\s+([0-9.]+)\s+\(([+-][0-9.]+)")


def curve(tag: str) -> list[tuple[int, float]]:
    f = ROOT / "results" / tag / "train.log"
    if not f.exists():
        return []
    out = []
    for ln in f.read_text(errors="ignore").splitlines():
        m = PAT.match(ln)
        if m:
            out.append((int(m.group(1)), float(m.group(4))))
    return out


def main() -> int:
    rows = []
    for name, tag in ARMS.items():
        c = curve(tag)
        if len(c) < 5:
            rows.append((name, None, None, None, len(c)))
            continue
        end = c[-1][1]
        half = len(c) // 2
        # 後半の傾き（1000 step あたり）
        (x0, y0), (x1, y1) = c[half], c[-1]
        slope = (y1 - y0) / max(x1 - x0, 1) * 1000
        worst = min(v for _, v in c)
        rows.append((name, end, slope, end - worst, c[-1][0]))
    print(f"{'arm':26s} {'終端':>9s} {'後半傾き':>10s} {'底からの戻り':>12s} {'step':>7s}")
    for n, e, s, r, st in rows:
        if e is None:
            print(f"{n:26s} {'(未完)':>9s} {'-':>10s} {'-':>12s} {st:>7d}")
            continue
        print(f"{n:26s} {e:>+9.4f} {s:>+10.5f} {r:>+12.4f} {st:>7d}")
    done = [r for r in rows if r[1] is not None and r[4] >= 20000]
    if len(done) < 4:
        print("\n  ⚠ まだ全腕が 20000 step に達していない。**途中で判定しない**"
              "（36 巡目に早期の 1 点で誤った結論を書いて撤回した）。")
        return 0
    best = max(done, key=lambda r: r[1])
    print(f"\n  終端最良: {best[0]}（{best[1]:+.4f}）")
    ship = [r for r in done if not r[0].startswith("wavehax")]
    bs = max(ship, key=lambda r: r[1])
    print(f"  出荷可能な範囲での最良: {bs[0]}（{bs[1]:+.4f}）")
    if bs[1] > 0:
        print("  => **出荷可能な建築で prior を越えた。** 本走行へ。")
    else:
        print("  => 出荷可能な範囲では prior を越えない。"
              "正規化以外に律速がある（GAN / step 数 / 別の設計）。")
    (ROOT / "results/z0/norm_verdict.json").write_text(json.dumps(
        {n: {"end": e, "slope_per_1k": s, "recovery": r, "step": st}
         for n, e, s, r, st in rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
