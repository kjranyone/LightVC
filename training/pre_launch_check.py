"""4.1 の起動前ゲート。**手順書に貼るのではなく、ここが正本**。

27 時間を焼く直前の唯一の関門。32 巡目の測定では、この関門への 14 変異
（`assert v == 'PASS'` を `v in ('PASS','INCONCLUSIVE')` に、ABI 凍結検査の恒真化、
未来不変性 `== 0.0` → `<= 1.0`、証跡と env の突合せの恒真化、`|| { exit 1; }` → `|| true`）
が**全件素通り**した。テキスト検査は意味の反転を見られない。

    uv run python pre_launch_check.py            # 実環境で判定
    uv run python pre_launch_check.py --fixtures # fixture 全件を検証
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def check(z0: pathlib.Path, env: dict, ear: pathlib.Path | None = None,
          via_step6: bool = False, verdict: tuple | None = None) -> list[str]:
    """入場条件を全部見て、落ちた理由を返す（空なら合格）。

    ⚠ 1 本目で例外を投げない。**全部の理由を集めて返す**——
    「1 つ直すたびに次が出る」を避け、作業者が一度に把握できるようにする。
    """
    import plan_numbers as PN
    bad: list[str] = []

    # ⚠ **オーナーによる明示免除**。捏造の代わりに置く逃げ道であって、無音の穴ではない。
    #    `waives` に**免除する検査名を列挙**したものだけが外れる（包括免除は無い）。
    #    外した検査は `waived` として返り、呼び出し側が**大書きして残す**。
    #    証跡（VERDICT.md / abi.json）を書き換えて通すことは絶対にしない——
    #    それは 36 巡目に事故った「変異を正本に焼き込む」と同じ形になる。
    ov = z0 / "owner_override.json"
    waives: set[str] = set()
    if ov.exists():
        o = json.loads(ov.read_text())
        if o.get("owner_approved") and len(str(o.get("reason", ""))) >= 40:
            waives = set(o.get("waives", []))

    # ⚠ fixture は判定を注入できる（実環境の RTF 判定に依存させない）。
    #    実行時（--fixtures 以外）は必ず**実測**から取る。36 巡目まで
    #    `rtf_verdict(RTF_NET_DEFAULT)`（計画値の literal）を呼んでいたので、
    #    2.4b で net を測っても関門の答えが変わらなかった＝空の関門だった。
    # ⚠ **実測の在否は注入と独立に見る**。fixture が差せるのは PASS/FAIL の
    #    判定だけで、「測ったかどうか」まで差せると空の関門に戻る。
    net, src = PN.measured_net(z0)
    if net is None:
        bad.append(f"{src}＝net RTF が未実測（計画値では通さない）")
    if verdict:
        v, m = verdict
    elif net is None:
        v, m = None, None
    else:
        # fullgraph は front 込みの実測なので予算 0.35 をそのまま使う
        v, m = PN.rtf_verdict(net, budget=0.35 if src == "fullgraph" else None)
    if v is not None and v != "PASS":
        bad.append(f"2.4b の RTF 判定が {v}"
                   f"（余裕 {m} / 凍結許容差 {PN.RTF_TOL} / 起動 {PN.RTF_RUNS} 回）")

    abi = z0 / "abi.json"
    if not abi.exists():
        bad.append("abi.json が無い（手順 3 未実施）")
    else:
        a = json.loads(abi.read_text())
        left = [k for k in ("_undecided", "decided_by")
                if k in a.get("synthesis", {})]
        if left:
            bad.append(f"手順 3（ABI 凍結）が未完了: synthesis に {left} が残っている")
        # ⚠ `TRAINING_PLAN.md` §4 条件 b は「`NFFT_A=1024` の根拠を要記載」。
        #    rev61 の関門は `_undecided`/`decided_by` しか見ておらず、
        #    3.5 の (i)/(ii)/(iii) をどれも書かずに凍結しても通った。
        #    §11 は「§4 の a–e 未達での R1 起動」をスコープ外と宣言している。
        why = str(a.get("mel_analysis", {}).get("n_fft_rationale", "")).strip()
        if len(why) < 20 and "nfft_a_rationale" not in waives:
            bad.append("abi.json の mel_analysis.n_fft_rationale が空/短すぎる"
                       "（TRAINING_PLAN §4 条件 b: NFFT_A の凍結根拠を 3.5 の "
                       "(i)/(ii)/(iii) のどれかの文面で書く）")

    if not via_step6 and "step1_ear" not in waives:
        vd = ear / "VERDICT.md" if ear else None
        if not (vd and vd.exists() and "手順 2 へ" in vd.read_text()):
            bad.append("手順 1 の判定が「手順 2 へ」でない（VERDICT.md）")
    elif "step1_ear" in waives:
        # ⚠ 耳ゲートを外したなら**代わりの判定が要る**。免除は「検査が消える」
        #    ことではない。閾値は走行前に凍結し、走行後に緩めない（規則 5）。
        pg = z0 / "proxy_gate.json"
        if not pg.exists():
            bad.append("耳ゲートを免除したのに proxy_gate.json が無い"
                       "（代替判定の閾値を走行前に凍結する）")
        else:
            g = json.loads(pg.read_text())
            if not g.get("owner_approved"):
                bad.append("proxy_gate.json の owner_approved が false")
            th = g.get("thresholds", {})
            miss = [k for k in ("step1_substitute", "promote_5_4", "ceiling_5_1")
                    if k not in th]
            if miss:
                bad.append(f"proxy_gate.json に閾値 {miss} が無い")

    cd = z0 / "ceiling_decision.json"
    if not cd.exists():
        bad.append("5.1 の 4 決定（比較相手 3 択 ＋ (n, δ)）が未確定（規則 5）")
    else:
        miss = [k for k in ("ref", "speakers", "n", "delta")
                if k not in json.loads(cd.read_text())]
        if miss:
            bad.append(f"ceiling_decision.json に {miss} が無い")

    cfg = (f"d{PN.DIM}_L{PN.L_BLK}_ki{PN.K_IN}_k{PN.K_BLK}"
           f"_nb{PN.NBIN}_cin{PN.CIN}_p1")
    f = z0 / f"future_inv_net_{cfg}.json"
    if not f.exists():
        bad.append(f"未来不変性 ネット込みの証跡が無い: {f.name}")
    else:
        d = json.loads(f.read_text())
        if d.get("lookahead_ms") != 0.0:
            bad.append(f"未来不変性 ネット込みが {d.get('lookahead_ms')} ms（0.00 が必要）")
        if d.get("unbounded"):
            bad.append("未来不変性 ネット込みが UNBOUNDED（発話全体統計）")
        if d.get("inconclusive"):
            bad.append("未来不変性 ネット込みが INCONCLUSIVE（編集が出力を動かさない）")
        prior = d.get("prior", True)
        for k, want in (("dim", PN.DIM), ("L", PN.L_BLK), ("k_in", PN.K_IN),
                        ("k", PN.K_BLK), ("nbin", PN.NBIN),
                        ("cin", PN.CIN)):  # 設計点の正本（v2f は 4 面）
            if d.get(k) != want:
                bad.append(f"証跡の {k}={d.get(k)}、今の構成は {want}")
        for k, e in (("dim", "DIM"), ("L", "NLAYER")):
            val = env.get(e)
            if not val:
                bad.append(f"{e} が未設定（4.1 の先頭で export する）")
            elif d.get(k) != int(val):
                bad.append(f"証跡の {k}={d.get(k)} と env {e}={val} が違う")
    return bad


def _fixtures() -> int:
    fx = ROOT / "results/z0/pl_fixtures"
    if not fx.exists():
        print(f"  ⚠ {fx} が無い")
        return 1
    ng = 0
    for f in sorted(fx.glob("*.json")):
        spec = json.loads(f.read_text())
        z0 = fx / spec["z0"]
        ear = fx / spec["ear"] if spec.get("ear") else None
        bad = check(z0, spec.get("env", {}), ear, spec.get("via_step6", False),
                    tuple(spec["verdict"]) if spec.get("verdict") else None)
        want = spec["expect_ok"]
        if (not bad) != want:
            ng += 1
            print(f"  NG  {f.name}: 合格={not bad}（期待 {want}） -> {bad[:2]}")
            continue
        need = spec.get("expect_reason_contains")
        if need and not any(need in b for b in bad):
            ng += 1
            print(f"  NG  {f.name}: 理由に '{need}' が無い -> {bad}")
            continue
        print(f"  ok  {f.name}")
    print(f"\n  fixture 不一致 {ng} 件")
    return 1 if ng else 0


def main() -> int:
    sys.path.insert(0, str(ROOT / "training"))
    if "--fixtures" in sys.argv:
        return _fixtures()
    z0 = ROOT / "results/z0"
    bad = check(z0, dict(os.environ), ROOT / "results/z0c_ear")
    ov = z0 / "owner_override.json"
    if ov.exists():
        o = json.loads(ov.read_text())
        print("  " + "=" * 66)
        print("  ⚠⚠ オーナー免除が有効。**この走行は無条件の合格ではない**")
        for w in o.get("waives", []):
            print(f"     免除: {w}")
        print(f"     理由: {o.get('reason', '')[:180]}")
        print(f"     риск: {o.get('risk', '')[:180]}".replace("риск", "残る危険"))
        print("  " + "=" * 66)
    for b in bad:
        print(f"  ⚠ {b}")
    print("  4.1 の入場条件 OK" if not bad else f"  入場不可（{len(bad)} 件）")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
