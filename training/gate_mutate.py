"""関門コード（`latency_gate.py`）を変異させて fixture が捕まえるかを見る。

32 巡目の測定: 122 変異のうち **53 件（43%）が「実行できるはずのコード」への変異**で、
そのどれも捕まらなかった（テキスト検査は意味の反転を見られない）。
∴ **コードを fixture で実行し、期待値との一致で捕まえる。**

⚠ 原本を書き換えない。コピーに変異を当て、`--target` で差し替えて実行する。
   32 巡目に原本へ直接 sed して 2 箇所を壊した（復元スクリプトが別の行に当たった）。

    uv run python gate_mutate.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "training/latency_gate.py"
SRC2 = ROOT / "training/pre_launch_check.py"
SRC3 = ROOT / "training/plan_numbers.py"

# (置換前, 置換後, 何の再発を防ぐか)
MUTATIONS = [
    ("and meas and stage", "and True and stage", "io 未実測 / provisional で pass:true"),
    ("tot < lim", "tot < 999", "30/50 ms 線の無効化"),
    ("q != qd", "False", "q 導出検査の恒真化"),
    ("if (K * hop) % d[\"hop_a\"]:", "if False:", "K 制約の無効化"),
    ("io < round(SR / 100)", "io < 0", "WASAPI assert の無効化"),
    ("(d[\"nfft_s\"] - hop) / INT", "0.0", "項 9 recon を消す"),
    ("\"3 queue\": q * io / SR", "\"3 queue\": 0.0", "項 3 queue を消す"),
    ("and reserve_ok", "and True", "予備 10 ms の承認証跡を恒真化"),
    ("and abi_ok", "and True", "3 者一致の無効化"),
    ("math.ceil(rtf * N / INT * SR / io)", "int(rtf * N / INT * SR / io)",
     "導出 q の切り上げ→切り捨て"),
    # ---- 33 巡目: 値空間の 1 点しか張られておらず 9 文が未実行だった分
    ("(d[\"sinc_len\"] / 256)", "1.0", "sinc_len を無視（48kHz で 200 サンプルずれる）"),
    ("if int(SR) == 44100 else round(", "if True else round(", "48 kHz のリサンプル項を消す"),
    ("50.0 if oav else 30.0", "50.0", "オーナー承認なしで 50 ms 線を使う"),
    ("all(k in ev for k in (\"obtained_at\", \"path\", \"by\"))", "True",
     "ASIO 証跡の形式検査を恒真化"),
    ("math.ceil(N - math.gcd(io * 44100, N * int(SR)) / SR)",
     "int(N - math.gcd(io * 44100, N * int(SR)) / SR)", "accum の切り上げ→切り捨て"),
    # ---- 34 巡目: 行は実行されていたが分岐が張られていなかった 12 件
    ('"5 lookahead": D * hop / INT', '"5 lookahead": 0.0', "先読み項を台帳から落とす"),
    ('("hop", hop, A["synthesis"]["hop"])', '("hop", hop, hop)', "3 者一致の hop 照合を恒真化"),
    ('("hop_a", d["hop_a"], A["mel_analysis"]["hop"])',
     '("hop_a", d["hop_a"], d["hop_a"])', "3 者一致の hop_a 照合を恒真化"),
    ('all(k in ra for k in ("by", "at", "evidence"))', 'ra is not None',
     "予備承認の必須鍵検査を緩める"),
    ('"scope": "fullgraph"', '"scope": "vonly"', "V 単体を full graph と名乗る"),
    ('"content_encoder_reserve_ms": 10.0', '"content_encoder_reserve_ms": 0.0',
     "予備 10 ms を 0 にする"),
    ('"budget_base_ms": 30.0', '"budget_base_ms": 50.0', "設計目標を 50 ms にする"),
]


def _load(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("lg_mut", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _check(mod) -> int:
    # ⚠ 理由照合キーの本数も返り値に含める。キーを消すだけで通る穴を塞ぐ
    #    （33 巡目: 「abort 理由の照合を消す」変異が素通りした）。
    fx = ROOT / "results/z0/fixtures"
    ng = 0
    for f in sorted(fx.glob("*.json")):
        spec = json.loads(f.read_text())
        ct = ROOT / "crates/lightvc-core/contract.toml"
        if "contract" in spec:
            ct = fx / f".{f.stem}.contract.toml"
            ct.write_text(spec["contract"])
        led, bad = mod.build_ledger(spec["input"], ct,
                                    ROOT / "results/z0/abi.json")
        if "contract" in spec and ct.exists():
            ct.unlink()
        for k, want in spec["expect"].items():
            got = led.get(k)
            good = (abs(got - want) < 0.01
                    if isinstance(want, float) and isinstance(got, (int, float))
                    else got == want)
            ng += 0 if good else 1
        if "expect_abort_contains" in spec and not any(
                spec["expect_abort_contains"] in b for b in bad):
            ng += 1
    n_reason = sum(1 for f in sorted(fx.glob("*.json"))
                   if "expect_abort_contains" in json.loads(f.read_text()))
    return ng + (1 if n_reason < 11 else 0)


PL_MUTATIONS = [
    # 36 巡目: 実測経路が生きたのでここを変異で試す。rev61 までは
    # `rtf_verdict(RTF_NET_DEFAULT)` の literal で、下 4 件は**存在すらしなかった**。
    ('net, src = PN.measured_net(z0)', 'net, src = (PN.RTF_NET_PLAN, "plan")',
     "4.1: 実測が無いとき計画値へフォールバック（＝測らずに通る空の関門）"),
    ('if net is None:\n        bad.append', 'if False:\n        bad.append',
     "4.1: net 未実測の検査を恒真化"),
    ('    if verdict:\n        v, m = verdict\n    elif net is None:',
     '    if True:\n        v, m = ("PASS", 1.0)\n    elif net is None:',
     "4.1: 判定を常に PASS に差し替え"),
    ('if v is not None and v != "PASS":', 'if v is not None and v == "NEVER":',
     "4.1: RTF 判定の検査を恒真化"),
    ('if v is not None and v != "PASS":',
     'if v is not None and v not in ("PASS", "INCONCLUSIVE", "FAIL"):',
     "4.1: RTF 判定が INCONCLUSIVE/FAIL でも起動できる"),
    ('if left:', 'if False:', "4.1: ABI 凍結検査の恒真化"),
    ('        if not pg.exists():', '        if False:',
     "4.1: 耳ゲート免除時に代替判定(proxy_gate)を要求しない"),
    ('            if not g.get("owner_approved"):', '            if False:',
     "4.1: proxy_gate の owner 承認検査を恒真化"),
    ('            if miss:', '            if False:',
     "4.1: proxy_gate の閾値欠落検査を恒真化"),
    ('if len(why) < 20 and "nfft_a_rationale" not in waives:',
     'if False:',
     "4.1: NFFT_A の凍結根拠（§4 条件 b）の検査を恒真化"),
    ('if d.get("lookahead_ms") != 0.0:', 'if d.get("lookahead_ms") > 1.0:',
     "4.1: 未来不変性の許容が緩んだ"),
    ('if d.get("unbounded"):', 'if False:', "4.1: UNBOUNDED を通す"),
    ('if d.get("inconclusive"):', 'if False:', "4.1: INCONCLUSIVE を通す"),
    ('if not val:', 'if False:', "4.1: DIM/NLAYER 未設定を fail-open に"),
    ('elif d.get(k) != int(val):', 'elif False:', "4.1: 証跡と env の突合せを恒真化"),
    ('if not (vd and vd.exists() and "手順 2 へ" in vd.read_text()):', 'if False:',
     "4.1: 手順 1 の判定検査を恒真化"),
    ('if not cd.exists():', 'if False:', "4.1: ceiling_decision の検査を恒真化"),
]


def _check_pl(mod) -> int:
    # ⚠ 理由照合キーの本数も返り値に含める。キーを消すだけで通る穴を塞ぐ
    #    （33 巡目: 「abort 理由の照合を消す」変異が素通りした）。
    fx = ROOT / "results/z0/pl_fixtures"
    ng = 0
    for f in sorted(fx.glob("*.json")):
        spec = json.loads(f.read_text())
        bad = mod.check(fx / spec["z0"], spec.get("env", {}),
                        fx / spec["ear"] if spec.get("ear") else None,
                        spec.get("via_step6", False),
                        tuple(spec["verdict"]) if spec.get("verdict") else None)
        if (not bad) != spec["expect_ok"]:
            ng += 1
        if "expect_reason_contains" in spec and not any(
                spec["expect_reason_contains"] in b for b in bad):
            ng += 1
    n_reason = sum(1 for f in sorted(fx.glob("*.json"))
                   if "expect_reason_contains" in json.loads(f.read_text()))
    return ng + (1 if n_reason < 15 else 0)


FIXTURE_MUTATIONS = [
    ("fixtures", "m_measured_pass.json", "expect", {"pass": False},
     "−1.1c: 合格側 fixture の期待値を False にして甘くする"),
    ("fixtures", "d_q_not_derived.json", "expect_abort_contains", "__DELETE__",
     "−1.1c: abort 理由の照合を消す"),
    ("pl_fixtures", "a_all_green.json", "expect_ok", False,
     "4.1: 合格側 fixture の期待値を False にして甘くする"),
    ("pl_fixtures", "c_unbounded.json", "expect_reason_contains", "__DELETE__",
     "4.1: UNBOUNDED の理由照合を消す"),
]


# ⚠ fixture の pin は `pins.py` に一本化した（34 巡目: 二重管理で
#    「pins は bless したのに gate_mutate が古い台帳で鳴る」状態になった）。


def _fixture_guard() -> list[str]:
    """fixture 自体が甘くされていないか。

    ⚠ fixture 方式の弱点は「fixture を甘くすれば通る」こと。
    **合格側と不合格側が両方あること**と**期待値が実測と一致すること**を数える。
    32 巡目に合格側 fixture が 1 本も無く、5 変異が素通りした。
    """
    out = []
    for d, need_pass in (("fixtures", True), ("pl_fixtures", True)):
        fx = ROOT / "results/z0" / d
        if not fx.exists():
            out.append(f"{d}/ が無い（fixture を消すだけで検査が空になる）")
            continue
        specs = [json.loads(f.read_text()) for f in fx.glob("*.json")]
        if len(specs) < 8:
            out.append(f"{d}/ の fixture が {len(specs)} 本（8 本以上を要求）")
        key = "expect" if d == "fixtures" else "expect_ok"
        ok = [s for s in specs
              if (s[key].get("pass") is True if d == "fixtures" else s[key] is True)]
        ng = [s for s in specs
              if (s[key].get("pass") is False if d == "fixtures" else s[key] is False)]
        if not ok:
            out.append(f"{d}/ に合格側の fixture が無い（合格経路の検査が存在しない）")
        if not ng:
            out.append(f"{d}/ に不合格側の fixture が無い")
        rk = "expect_abort_contains" if d == "fixtures" else "expect_reason_contains"
        # ⚠ 不合格 fixture は「なぜ落ちたか」まで見る。理由照合を消せば
        #    「何かの理由で落ちた」だけになり、別の原因で落ちても通ってしまう。
        # ⚠ **本数の下限で守らない**（36 巡目）。下限 15 本に対して
        #    fixture を 1 本足すと 1 本から理由照合を消せてしまい、
        #    「UNBOUNDED の理由照合を消す」変異が実際に素通りした。
        #    ∴ 不合格側は**全件必須**にする（足しても抜け穴が増えない）。
        # 不合格 fixture は「なぜ落ちたか」を必ず 1 つ以上主張する。
        # latency_gate は ledger を返すので、abort 理由の代わりに
        # **pass 以外の判別フィールド**を固定していれば可とする
        # （abort が出ても ledger は組み上がるため、そちらが本体の主張）。
        miss = [s_ for s_ in ng if rk not in s_ and not (
            d == "fixtures" and set(s_.get("expect", {})) - {"pass"})]
        if miss:
            out.append(f"{d}/ の不合格 fixture {len(miss)} 本が落ちた理由を主張していない"
                       f"（{rk} も pass 以外の期待値も無い）——「何かの理由で落ちた」"
                       "だけになり、別の原因で落ちても通る")
    return out


# 36 巡目: 4.1 の判定は plan_numbers の意味論に乗っているのに、そこは
# 一度も変異させていなかった（許容差の凍結・未実測時の挙動・FAIL/INCONCLUSIVE の別）。
PN_MUTATIONS = [
    ('        return None, "2.4b の実測（rtf_bench.py）が無い"',
     '        return RTF_NET_PLAN, "2.4b の実測（rtf_bench.py）が無い"',
     "plan_numbers: 未実測に計画値を返す（測らずに通る）"),
    ('    if tol is None:\n        return "INCONCLUSIVE", margin',
     '    if tol is None:\n        return "PASS", margin',
     "plan_numbers: 許容差が未凍結でも PASS（規則 5 の骨抜き）"),
    ('    if margin < tol:', '    if margin < 0:',
     "plan_numbers: 凍結許容差を判定から外す"),
    ('    if not b.get("owner_approved"):', '    if False:',
     "plan_numbers: RTF 基準の owner 承認検査を恒真化"),
    ('        if d.get("basis") != b["basis"]:', '        if False:',
     "plan_numbers: 測定の基準と宣言の突合せを恒真化（測ってから基準を選べる）"),
    ('    if not bf.exists():', '    if False:',
     "plan_numbers: 基準未宣言でも通す（規則 5 の骨抜き）"),
    ('    if runs < MIN_RUNS:', '    if False:',
     "plan_numbers: 起動回数の下限を無視する"),
]


def _check_verdict() -> int:
    """判定表 fixture。`rtf_verdict` の境界を直接踏む。"""
    import importlib
    PN = importlib.import_module("plan_numbers")
    rows = json.loads((ROOT / "results/z0/verdict_fixtures.json").read_text())
    ng = 0
    for r in rows:
        v, _ = PN.rtf_verdict(r["net"], r["tol"], r["runs"])
        if v != r["want"]:
            ng += 1
    return ng


def main() -> int:
    src = SRC.read_text()
    fg = _fixture_guard()
    for x in fg:
        print(f"  ⚠ fixture 側の欠陥: {x}")
    if _check(_load(SRC)):
        print("  ⚠ latency_gate: 変異前に既に不一致がある。先にそちらを直す")
        return 1
    # ⚠ 33 巡目: これが無く、pre_launch_check を no-op にしても 0/24 rc=0 だった。
    #    しかも壊れたベースラインの上では全変異が「guarded」に見える（壊れているほど健全）。
    if _check_pl(_load(SRC2)):
        print("  ⚠ pre_launch_check: 変異前に既に不一致がある。先にそちらを直す")
        return 1
    unguarded = []
    with tempfile.TemporaryDirectory() as td:
        for old, new, what in MUTATIONS:
            if src.count(old) != 1:
                print(f"  SKIP  {what}（対象が {src.count(old)} 箇所）")
                unguarded.append(what + "（変異の定義が古い）")
                continue
            p = pathlib.Path(td) / "lg.py"
            p.write_text(src.replace(old, new, 1))
            try:
                ng = _check(_load(p))
            except Exception:                          # noqa: BLE001
                ng = 1                                 # 落ちるのも「捕まえた」
            if ng:
                print(f"  guarded    {what}")
            else:
                print(f"  UNGUARDED  {what}")
                unguarded.append(what)
    src2 = SRC2.read_text()
    with tempfile.TemporaryDirectory() as td:
        for old, new, what in PL_MUTATIONS:
            if src2.count(old) != 1:
                print(f"  SKIP  {what}（対象が {src2.count(old)} 箇所）")
                unguarded.append(what + "（変異の定義が古い）")
                continue
            p = pathlib.Path(td) / "pl.py"
            p.write_text(src2.replace(old, new, 1))
            try:
                ng = _check_pl(_load(p))
            except Exception:                          # noqa: BLE001
                ng = 1
            print(f"  {'guarded   ' if ng else 'UNGUARDED '} {what}")
            if not ng:
                unguarded.append(what)
    src3 = SRC3.read_text()
    with tempfile.TemporaryDirectory() as td:
        for old, new, what in PN_MUTATIONS:
            if src3.count(old) != 1:
                print(f"  SKIP  {what}（対象が {src3.count(old)} 箇所）")
                unguarded.append(what + "（変異の定義が古い）")
                continue
            bak = src3
            try:
                SRC3.write_text(src3.replace(old, new, 1))
                # ⚠ **再読込を強制する**。pre_launch_check は関数内で
                #    `import plan_numbers` するので、sys.modules に載ったままだと
                #    ディスクの変異が一切効かず、全変異が「guarded」に見える。
                sys.modules.pop("plan_numbers", None)
                ng = _check_pl(_load(SRC2)) or _check_verdict()
            except Exception:                          # noqa: BLE001
                ng = 1
            finally:
                SRC3.write_text(bak)
                sys.modules.pop("plan_numbers", None)
            print(f"  {'guarded   ' if ng else 'UNGUARDED '} {what}")
            if not ng:
                unguarded.append(what)

    # fixture の期待値を甘くする変異
    with tempfile.TemporaryDirectory() as td:
        for d, name, key, val, what in FIXTURE_MUTATIONS:
            f = ROOT / "results/z0" / d / name
            if not f.exists():
                unguarded.append(what + "（fixture が無い）")
                print(f"  SKIP  {what}")
                continue
            orig = f.read_text()
            spec = json.loads(orig)
            if val == "__DELETE__":
                spec.pop(key, None)          # 理由照合の「キーごと削除」
            elif key == "expect" and isinstance(val, dict):
                spec["expect"].update(val)
            else:
                spec[key] = val
            f.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
            try:
                # ⚠ **fixture ガードも判定に入れる**（36 巡目）。rev61 までは
                #    _check だけを見ており、「理由照合のキーごと削除」は
                #    合否が変わらないので素通りしていた（fixture を甘くする
                #    変異は、実行結果ではなく fixture の形で捕まえるしかない）。
                ng = (_check(_load(SRC)) if d == "fixtures"
                      else _check_pl(_load(SRC2))) or len(_fixture_guard())
            finally:
                f.write_text(orig)
            print(f"  {'guarded   ' if ng else 'UNGUARDED '} {what}")
            if not ng:
                unguarded.append(what)
    total = (len(MUTATIONS) + len(PL_MUTATIONS) + len(PN_MUTATIONS)
             + len(FIXTURE_MUTATIONS))
    unguarded += fg
    print(f"\n  fixture が捕まえられない変異: {len(unguarded)} / {total}")
    for u in unguarded:
        print(f"    - {u}")
    return 1 if unguarded else 0


if __name__ == "__main__":
    raise SystemExit(main())
