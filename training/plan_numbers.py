"""計画書が引く数値の正本。文書にリテラルを散在させない。

同じ量が節をまたいで腐る事故が 16 巡で毎回出た（front-end RTF / net 予算 /
q / fixed_path_latency_samples / 解の件数 / 48kHz の可否）。検査を足しても
「腐ったことが分かる」だけなので、**値をここでしか定義しない**。

    uv run python plan_numbers.py            # 表を印字
    uv run python plan_numbers.py --json     # results/z0/plan_numbers.json

doc_check.py の check_plan_numbers がこれと文書中の数値を照合する。
"""
from __future__ import annotations

import json
import sys
from math import ceil, gcd
from pathlib import Path

import ship_front as SF

OUT = Path(__file__).resolve().parent.parent / "results/z0/plan_numbers.json"

# ---- 実装から来る量（ship_front.py が正本）
HOP_S = SF.HOP_S
HOP_A = SF.HOP_A
NFFT_S = SF.NFFT_S
NFFT_A = SF.NFFT_A
K = 2
N = K * HOP_S
NBIN = NFFT_S // 2 + 1
def cin_of(prior: bool = True) -> int:
    """入力チャネル数。**prior の有無で変わる**。

    ⚠ 30 巡目まで `CIN` は式で固定されており、4.1 のゲートが
    `assert d['cin'] == PN.CIN` を課していたので、**腕 B1（prior を外して cin=80）は
    証跡に 80 と書けば落ち、851 と書けば嘘になる**＝構造的に通せなかった。
    """
    return 80 + 3 * NBIN if prior else 80


ARCH = "v2f"                          # 設計点（2026-08-19 確定。RESEARCH.md 切り分け 8-9）
CIN = 4 if ARCH == "v2f" else cin_of(True)
BLOCK_MS = N / 44100 * 1e3
RECON = NFFT_S - HOP_S
IN_RING = NFFT_A + 3 * HOP_A          # (NFFT_A - HOP_A) + 4*HOP_A
EXC_RING = NFFT_S - HOP_S

# ---- 手順 0.5 の実測。**正本は results/z0/rtf_front.json**（rtf_front.py が書く）
# 走行ごとに ±5% ぶれるので **文書に書くのは 2 桁**（3 桁は偽の精度）。
_ART = Path(__file__).resolve().parent.parent / "results/z0/rtf_front.json"


def _measured() -> dict:
    if _ART.exists():
        return json.loads(_ART.read_text())
    return {"front_end_current": 0.509, "front_end_fused": 0.138,
            "candidate0": 0.023, "_stale": True}


_HIST = Path(__file__).resolve().parent.parent / "results/z0/rtf_front_history.jsonl"


def _history() -> list[float]:
    """front-end RTF の**起動をまたいだ**観測列。判定はこれで引く。

    プロセス内の反復は熱・キャッシュ・アロケータ状態を共有するので
    ぶれ幅を 8 倍過小評価する（実測: 内 0.0019〜0.0073 / 間 0.0081）。
    **負荷下の観測を「外れ値」として捨てない**——製品は負荷下でも回る。
    """
    if not _HIST.exists():
        return []
    return [json.loads(x)["front_end_fused"]
            for x in _HIST.read_text().splitlines() if x.strip()]


_M = _measured()
_H = _history()
# ⚠ front_end_* は **最悪値**（rtf_front.py が n 回走らせた max）。中央値で読むと
# 合否がマシン負荷で反転する（実測: 同一機で 0.127〜0.143、ぶれ幅 0.0073）。
MIN_RUNS = 3
# ⚠ **ぶれ幅を max-min で持たない**（36 巡目）。判定は
#   `0.35 - max(H) - net >= max(H) - min(H)` になっていて、走行を足すと
#   左辺は減り右辺は増える——**両辺が単調に悪化する**。それでいて 0.5 と
#   2.4b-1a は「INCONCLUSIVE なら走行を足せ」と指示していた。
#   ＝ 座標上昇で自力で FAIL 域へ歩く構造（`memory/benchmark-degeneracy-gate`）。
# ∴ 許容差は**走行前に凍結した定数**にする（規則 5）。走行を足して動くのは
#   max(H) だけで、それが予算を割ったなら metric の綾ではなく本当に FAIL。
_TOL = Path(__file__).resolve().parent.parent / "results/z0/rtf_tolerance.json"
_T = json.loads(_TOL.read_text()) if _TOL.exists() else None
RTF_TOL = _T["tol"] if _T else None
RTF_TOL_N = _T.get("n_at_freeze") if _T else None
RTF_SPREAD = (round(max(_H) - min(_H), 4) if len(_H) >= MIN_RUNS else None)
RTF_RUNS = len(_H)
RTF_FRONT_CURRENT = round(_M["front_end_current"], 2)
_FUSED_WORST = max(_H) if _H else _M["front_end_fused"]
RTF_FRONT_FUSED = round(_FUSED_WORST, 2)
CAND0_FRONT = round(_M["candidate0"], 2)
# ⚠ 2.1 の既定は dim256 / L6 / k_in=7 / k=3（rev52）＝旧ラダー 4 込み。
# **`v_only` はこの既定で計算する**（旧ラダー 2 の 0.26 は参考値）。
# ⚠ **これは計画値であって実測ではない**（36 巡目）。rev61 まで関門が
#   `rtf_verdict(RTF_NET_DEFAULT)` を呼んでおり、**2.4b で net を測っても
#   関門の答えが変わらなかった**（測定に意味が無い＝空の関門）。
#   実測は `measured_net()` が `rtf_K*_T*_d*_L*_n*.json` から読む。
RTF_NET_PLAN = 0.20                   # dim256/L6/k3 の計画値（引用専用）
RTF_NET_DEFAULT = RTF_NET_PLAN
RTF_NET_LADDER2 = 0.26                # dim256/L6/k7（参考）
RTF_NET_LADDER4 = RTF_NET_DEFAULT
NET_BUDGET_RAW = 0.35 - _FUSED_WORST            # 判定はこちら（丸めない・最悪値）
NET_BUDGET = round(0.35 - RTF_FRONT_FUSED, 2)   # 文書引用用の 2 桁
E_BUDGET = 0.10
G_BUDGET = 0.05

# ---- 幹のアーキ（2.1 の既定。rev53）
# ⚠ 設計点は **v2f ch24/L8**（2026-08-19 確定）。v1d dim128 は品質不足で退役
#   （60k step で prior 比 −0.06、v2f ch24 は +0.0632。RESEARCH.md 切り分け 8-9）。
#   K_IN/K_BLK は v2f の (kf, kt) を指す。CIN は入力面数 4（mel_lin/Re/Im/log|P|）。
K_IN = 7                              # v2f の kf（周波数方向 kernel）
K_BLK = 3                             # v2f の kt（時間方向 kernel。左パディングのみ）
L_BLK = 8
# ⚠ **dim 256 は C.4 の Rust 実測で否決された**（candle-cpu / K=2 / p95 0.4302 vs 予算 0.207）。
#    PyTorch 代理指標（blas_floor 0.0978）は 4 倍楽な値を出しており、実測が覆した。
#    2.4b-1a の逓減ラダーを 1 段降りて 128（p95 0.0878 ＝ 2.4 倍の余裕）。
#    ⚠ ここを動かしたら parity_check の証跡と rtf の実測を**両方**取り直す。
DIM = 24
# v2f: 入口 1x1、周波数方向 kf は時間文脈に効かない。時間は各層 kt-1。
CTX = ((K_BLK - 1) * L_BLK if ARCH == "v2f"
       else (K_IN - 1) + (K_BLK - 1) * L_BLK)   # 左文脈フレーム = 18

IOS = [8, 16, 32, 48, 64, 96, 128, 192, 256]
QMAX = 16
SINCS = [256, 64]
AD_DA_MS = 10.0
BUDGET_MS = 30.0
DISQUALIFY_MS = 50.0


def accum(io: int, sr: int) -> int:
    return ceil(N - gcd(io * 44100, N * sr) / sr)


def resamp(sr: int, sinc: int = 256) -> int:
    if sr == 44100:
        return 0
    return round(((117 * sr / 44100 + 139) / 2) * (sinc / 256))


def q_of(rtf: float, io: int, sr: int) -> int:
    return max(1, ceil(rtf * N / 44100 * sr / io))


def latency(sr: int, io: int, q: int, sinc: int = 256, d_ahead: int = 0,
            ad: float = AD_DA_MS) -> tuple[float, float, float]:
    conv = ((accum(io, sr) + d_ahead * HOP_S + RECON) / 44100 * 1e3
            + q * io / sr * 1e3 + 2 * resamp(sr, sinc) / sr * 1e3)
    hw = 2 * io / sr * 1e3 + ad
    return conv, hw, q * io / sr * 1e3


def fixed_host(io: int, sr: int, q: int, d_ahead: int = 0, sinc: int = 256) -> int:
    """PDC 申告値。⚠ `sinc` を取る（rev60 まで 256 決め打ちで、48 kHz・sinc=64 では
    200 サンプル ＝ 4.17 ms ずれた。48 kHz で 30 ms を満たす解は全部 sinc=64 なので、
    **DAW 経路で採る予定の設定でだけ正本が壊れていた**）。"""
    return (round((accum(io, sr) + d_ahead * HOP_S + RECON) * sr / 44100)
            + q * io + 2 * resamp(sr, sinc))


def solutions(rtf: float, free_q: bool = False) -> list[tuple[float, int, int, int, int]]:
    """30 ms 未満の構成。

    ⚠ 既定は **`q` を導出値に固定**する（`q = ceil(rtf × N/44100 × SR/io)`）。
    −1.1c の検算スクリプトが `assert q == 導出値` を課しているので、
    `q` を自由変数として数えると**関門が弾く構成を「解」に数えてしまう**。
    `free_q=True` は「q を自由にできたら何件あったか」の参考値。
    """
    need = rtf * N / 44100 * 1e3
    out = []
    for sr in (44100, 48000):
        for io in IOS:
            qs = range(1, QMAX + 1) if free_q else [q_of(rtf, io, sr)]
            for q in qs:
                if q > QMAX:
                    continue
                for sinc in (SINCS if sr == 48000 else [256]):
                    conv, hw, qms = latency(sr, io, q, sinc)
                    if qms >= need and conv + hw < BUDGET_MS:
                        out.append((round(conv + hw, 3), sr, io, q, sinc))
    return sorted(out)


def pdc_table(rtf: float, sr: int = 44100,
              sinc: int = 256) -> dict[int, tuple[int, int, float]]:
    t = {}
    for io in (64, 128, 256, 512, 1024):
        q = q_of(rtf, io, sr)
        conv, hw, _ = latency(sr, io, q, sinc)
        t[io] = (q, fixed_host(io, sr, q, sinc=sinc), round(conv + hw, 2))
    return t


def measured_net(z0: "Path | None" = None) -> tuple[float | None, str]:
    """2.4b の実測 net RTF を読む。無ければ `(None, 理由)`。

    ⚠ **計画値へフォールバックしない。** 落とすのが正しい——
    「測っていないのに通る」が 36 巡目に見つかった空の関門そのもの。
    """
    d0 = z0 or (Path(__file__).resolve().parent.parent / "results/z0")
    # ⚠ full graph の実測があればそれが最優先（rtf_basis.json の凍結どおり
    #   「最終判定は C.4 の full graph」）。予算は 0.35 をそのまま使う。
    fg = d0 / "rtf_fullgraph_K2.json"
    if fg.exists():
        d = json.loads(fg.read_text())
        if "p95_worst_of_runs" in d:
            return float(d["p95_worst_of_runs"]), "fullgraph"
    fs = sorted(d0.glob("rtf_K*_T*_d*_L*_n*.json"))
    if not fs:
        return None, "2.4b の実測（rtf_bench.py）が無い"
    # ⚠ **測る基準を走行前に宣言させる**（規則 5）。宣言が無ければ通さない——
    #    測ってから都合のよい基準を選べる形にしない。
    bf = d0 / "rtf_basis.json"
    if not bf.exists():
        return None, "results/z0/rtf_basis.json が無い（RTF の採用基準が未宣言）"
    b = json.loads(bf.read_text())
    if not b.get("owner_approved"):
        return None, "rtf_basis.json の owner_approved が false"
    if len(str(b.get("rationale", ""))) < 50:
        return None, "rtf_basis.json の rationale が短すぎる（採用理由を書く）"
    vals = []
    for f in fs:
        d = json.loads(f.read_text())
        if "net_rtf" not in d:
            return None, f"{f.name} に net_rtf が無い"
        if d.get("basis") != b["basis"]:
            return None, (f"{f.name} の基準 {d.get('basis')} が宣言 "
                          f"{b['basis']} と違う（rtf_bench.py を測り直す）")
        vals.append(float(d["net_rtf"]))
    return max(vals), f"{len(fs)} 本の最悪値（基準 {b['basis']}）"


def rtf_verdict(net: float, tol: "float | None" = -1.0,
                runs: int = -1, budget: "float | None" = None) -> tuple[str, float]:
    """net が予算に入るかを**ぶれ幅つき**で判定する。

    余裕がぶれ幅未満なら PASS にしない（`CLAUDE.md` 出荷ゲートの
    「編集が出力を動かさない場合は INCONCLUSIVE ＝ PASS にしない」と同じ規律）。
    """
    # ⚠ tol/runs は**注入できる**（既定 -1 は「実環境から取る」の意）。
    #    36 巡目まで注入口が無く、判定表の境界を踏む fixture が書けないので
    #    「許容差を判定から外す」「起動回数の下限を無視する」変異が素通りした。
    tol = RTF_TOL if tol == -1.0 else tol
    runs = RTF_RUNS if runs == -1 else runs
    # budget=None は「net 予算（0.35 − front 履歴の最悪値）」。full graph 実測を
    #   渡すときは budget=0.35（front 込みなので二重に引かない）。
    margin = round((NET_BUDGET_RAW if budget is None else budget) - net, 4)
    if margin < 0:
        return "FAIL", margin
    if tol is None:
        return "INCONCLUSIVE", margin   # 許容差が未凍結（規則 5）。走行では解けない
    if runs < MIN_RUNS:
        # ⚠ **走行を足して解けるのはここだけ。** 0.5 / 2.4b-1a の「足せ」は
        #    この枝に限る。下の枝で足しても max(H) が増えるだけで悪化する。
        return "INCONCLUSIVE", margin
    if margin < tol:
        # ⚠ INCONCLUSIVE にしない。凍結した許容差に対しては**答えが出ている**。
        #    ラダーを降りる（2.4b-1a）のが正しい行き先。
        return "FAIL", margin
    return "PASS", margin


def report() -> dict:
    v_only = round(RTF_FRONT_FUSED + RTF_NET_DEFAULT, 2)      # 2.1 の既定
    full = round(v_only + E_BUDGET + G_BUDGET, 2)
    v_l4 = round(RTF_FRONT_FUSED + RTF_NET_LADDER2, 2)        # 参考（k=7）
    full_l4 = round(v_l4 + E_BUDGET + G_BUDGET, 2)
    r = {
        "impl": {"HOP_S": HOP_S, "HOP_A": HOP_A, "NFFT_S": NFFT_S,
                 "NFFT_A": NFFT_A, "K": K, "N": N, "NBIN": NBIN, "cin": CIN,
                 "recon_samples": RECON, "block_ms": round(BLOCK_MS, 3),
                 "input_ring_samples": IN_RING,
                 "excitation_ring_samples": EXC_RING,
                 "k_in": K_IN, "k": K_BLK, "L": L_BLK, "dim": DIM, "CTX": CTX},
        "rtf": {"front_current": RTF_FRONT_CURRENT, "front_fused": RTF_FRONT_FUSED,
                "net_budget": NET_BUDGET, "net_default": RTF_NET_DEFAULT, "net_ladder2": RTF_NET_LADDER2, "cand0_front": CAND0_FRONT,
                "v_only": v_only, "full_graph": full,
                "v_only_k7": v_l4, "full_graph_k7": full_l4,
                # net_stale: 実測ファイルが揃っているかを**実物で判定**する
                #   （リテラル True は v1d 時代の名残。rtf_bench.py・実測 JSON・
                #    full graph 実測が全部あれば再現可能＝stale ではない）。
                "net_stale": not (
                    (Path(__file__).resolve().parent / "rtf_bench.py").exists()
                    and measured_net()[0] is not None)},
        "solutions": {str(x): len(solutions(x))
                      for x in (0.35, v_only, 0.5, full, 0.821, 1.101)},
        "solutions_free_q": {str(x): len(solutions(x, free_q=True))
                             for x in (0.35, v_only, 0.5, full, 0.821, 1.101)},
        "pdc_44k": {str(rt): {str(io): list(v) for io, v in pdc_table(rt).items()}
                    for rt in (0.5, full)},
    }
    # ⚠ 文書が太字で書く数値は**すべてここから導出できる**ようにする。
    #    30 巡目: 「artifact なのに正本と照合されていない」18 種を潰すため。
    r["derived"] = {
        "candidate0_net_budget": round(0.35 - CAND0_FRONT, 2),
        "cliff_margin": 0.01,
        "e_budget": E_BUDGET, "g_budget": G_BUDGET,
        "resamp_48k_sinc256": resamp(48000, 256),
        "resamp_48k_sinc64": resamp(48000, 64),
        "wasapi_441": round(sum(latency(44100, 441, 1)[:2]), 2),
        "wasapi_480": round(sum(latency(48000, 480, 1)[:2]), 2),
        "e2e_48k_io256_q1": round(sum(latency(48000, 256, 1)[:2]), 2),
        "e2e_48k_io512_q1": round(sum(latency(48000, 512, 1)[:2]), 2),
        "e2e_44k_io256_q1_nfft256": 24.51, "e2e_48k_io256_q1_nfft256": 32.25,
        "front_current_measured": RTF_FRONT_CURRENT,
        "osc_share": 0.075, "chunk_clap": 2048, "chunk_app": 4096,
    }
    r["rtf"]["spread"] = RTF_SPREAD
    r["rtf"]["tol_frozen"] = RTF_TOL
    # ⚠ 目標は「予算」ではなく「予算 − 凍結許容差」。丸めた予算 0.21 を目標に
    #    書くと、余裕 0.0070 で PASS に見えるが実効ラインは 0.1983（36 巡目）。
    r["derived"]["net_target"] = (round(NET_BUDGET_RAW - RTF_TOL, 4)
                                  if RTF_TOL is not None else None)
    r["rtf"]["spread_runs"] = RTF_RUNS
    _mn, _src = measured_net()
    vd, mg = rtf_verdict(_mn if _mn is not None else RTF_NET_DEFAULT,
                         budget=0.35 if _src == "fullgraph" else None)
    r["rtf"]["verdict"] = vd
    r["rtf"]["margin"] = mg
    # ⚠ 48 kHz の解は sinc=64 でしか成立しない。その解の PDC を **その sinc で** 出す
    #   （既定 256 で出すと 200 サンプル ＝ 4.17 ms 嘘になる）。
    r["solutions_48k"] = [
        {"rtf": rt, "io": io, "q": q, "sinc": sc, "e2e_ms": tot,
         "fixed_host": fixed_host(io, 48000, q, sinc=sc)}
        for rt in (0.39, v_only, full, 0.5)
        for tot, sr, io, q, sc in solutions(rt) if sr == 48000]
    r["k_constraint_ok"] = (K * HOP_S) % HOP_A == 0
    r["measured_stale"] = _M.get("_stale", False)
    return r


def main() -> int:
    r = report()
    print(f"  幹  : dim {DIM} / L {L_BLK} / k_in {K_IN} / k {K_BLK} / CTX {CTX}")
    print(f"  実装: NBIN {NBIN} / cin {CIN} / K {K} / recon {RECON} sa "
          f"({RECON/44100*1e3:.2f} ms) / 入力リング {IN_RING} / 励起リング {EXC_RING}")
    print(f"  RTF : front 現行 {RTF_FRONT_CURRENT} → 融合 {RTF_FRONT_FUSED} "
          f"/ net 予算 {NET_BUDGET} / 候補0 {CAND0_FRONT}")
    print(f"        V 単体 {r['rtf']['v_only']} / full graph {r['rtf']['full_graph']}"
          f"（参考 k=7 なら {r['rtf']['v_only_k7']} / {r['rtf']['full_graph_k7']}）")
    _m, _s = measured_net()
    _bud = 0.35 if _s == "fullgraph" else NET_BUDGET_RAW
    print(f"  判定: {'full graph' if _s == 'fullgraph' else 'net'} "
          f"{_m if _m is not None else RTF_NET_DEFAULT} "
          f"vs 予算 {_bud:.4f} → "
          f"{r['rtf']['verdict']}（余裕 {r['rtf']['margin']} / ぶれ幅 {RTF_SPREAD}）")
    for k, n in r["solutions"].items():
        print(f"  解の件数 RTF {k:>6s}: {n:3d}（q 自由なら {r['solutions_free_q'][k]:3d}）")
    for rt, tab in r["pdc_44k"].items():
        print(f"  PDC 44.1k rtf={rt}: " + " / ".join(
            f"io{io}(q{v[0]},{v[1]},{v[2]}ms)" for io, v in tab.items()))
    if "--json" in sys.argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
