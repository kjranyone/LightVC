"""−1.1c の遅延台帳。**手順書に貼るのではなく、ここが正本**。

32 巡目の測定: 122 変異のうち **53 件（43%）が「実行できるはずのコード」への変異**で、
そのどれも 1 件も実行されていなかった（`gate_run` の実行対象は 47 ブロック中 6、
うち 3 本は自分の assert に到達する前に abort、−1.1c 本体は除外されていた）。
∴ **テキストを検査するのをやめる。** コードを 1 本のファイルに出し、
fixture で実行して期待値を照合する。

    uv run python latency_gate.py <target_platform.json>   # 台帳を出す
    uv run python latency_gate.py --fixtures               # fixture 全件を検証

手順書は本ファイルを呼ぶだけにし、`doc_check` は「呼んでいるか」だけを見る。
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INT = 44100.0


def build_ledger(d: dict, contract: pathlib.Path | None = None,
                 abi: pathlib.Path | None = None) -> tuple[dict, list[str]]:
    """台帳を作る。**abort 条件は例外ではなく理由のリストで返す**（部分結果も出す）。

    ⚠ 32 巡目まで assert を並べていたので、1 本目で落ちると 9 項の内訳が出ず、
    「provisional 段で内訳が出ていること」という手順 C の入場条件を満たせなかった。
    """
    bad: list[str] = []
    SR = float(d["sample_rate"])
    hop = d["hop"]
    K = d["net_block_frames"]
    N = K * hop
    q = d["queue_depth_io_blocks"]

    _m = d.get("io_block_samples_measured")
    io = _m if _m is not None else d["io_block_samples"]
    src = "measured" if _m is not None else "declared"
    meas = _m is not None

    if d.get("primary_os") == "windows" and not d.get("asio_sdk_available") \
            and io < round(SR / 100):
        bad.append("WASAPI 共有モードでは io を選べない（cpal は BufferSize::Fixed を無視）")
    ev = d.get("asio_sdk_evidence")
    if d.get("asio_sdk_available") and not (
            isinstance(ev, dict) and all(k in ev for k in ("obtained_at", "path", "by"))):
        bad.append("asio_sdk_available: true は現物証跡（asio_sdk_evidence）が要る")
    if (K * hop) % d["hop_a"]:
        bad.append(f"K 制約違反: K×hop={K*hop} は hop_a={d['hop_a']} の倍数であること")

    D, dsrc = 0, "absent(C 未着手)"
    # ⚠ ファイル不在を黙って D_ahead=0 にしない（33 巡目: これが fail-open だった）。
    #    「鍵が無ければ abort」と書いたのに、ファイルごと無いときだけ効かなかった。
    if contract is not None and not contract.exists():
        bad.append("crates/lightvc-core/contract.toml が無い＝D_ahead が未確定"
                   "（C.5 で作る。不在も『未確定』として止める）")
    if contract and contract.exists():
        m = re.search(r"lookahead_frames_D_ahead\s*=\s*(\d+)", contract.read_text())
        if m is None:
            bad.append("contract.toml に lookahead_frames_D_ahead が無い（D_ahead=0 で素通りさせない）")
        else:
            D, dsrc = int(m.group(1)), "contract.toml"

    abi_ok = False
    if abi and abi.exists():
        A = json.loads(abi.read_text())
        abi_ok = True
        for k, got, want in (("nfft_s", d["nfft_s"], A["synthesis"]["n_fft"]),
                             ("hop", hop, A["synthesis"]["hop"]),
                             ("hop_a", d["hop_a"], A["mel_analysis"]["hop"])):
            if got != want:
                bad.append(f"{k} 不一致: target_platform {got} vs abi {want}")
    else:
        bad.append("results/z0/abi.json が無い＝3 者一致が空検査になる")

    rs = 0 if int(SR) == 44100 else round(
        ((117 * SR / 44100 + 139) / 2) * (d["sinc_len"] / 256))
    accum = math.ceil(N - math.gcd(io * 44100, N * int(SR)) / SR)
    cv = {"2 accum": accum / INT, "3 queue": q * io / SR,
          "4 resamp_in": rs / SR, "5 lookahead": D * hop / INT,
          "6 resamp_out": rs / SR, "9 recon": (d["nfft_s"] - hop) / INT}
    hw = {"1 in_buf": io / SR, "7 out_buf": io / SR}
    ad = d["ad_da_estimate_ms"]
    Lc = sum(cv.values()) * 1e3
    Lh = sum(hw.values()) * 1e3 + ad
    tot = Lc + Lh

    rtf = d.get("rtf_target")
    qd = None if rtf is None else max(1, math.ceil(rtf * N / INT * SR / io))
    if rtf is None:
        bad.append("rtf_target が null（未測定でゲートを開けない）")
    elif rtf >= 1.0:
        bad.append("RTF >= 1 は遅延枠を広げても救えない")
    elif q != qd:
        bad.append(f"q は導出値: ceil(rtf×N/44100×SR/io) = {qd}（凍結値を書かない）")
    elif q * io / SR < rtf * N / INT:
        bad.append("queue 制約違反: 出力先詰めが演算時間を覆えていない")

    rf = sorted((ROOT / "results/z0").glob("rtf_fullgraph_*.json"))
    rsrc = "none"
    if rtf is not None and rf:
        t = json.loads(rf[-1].read_text()).get("total_rtf")
        rsrc = str(rf[-1]) if t is not None and abs(t - rtf) < 1e-9 else "_rtf_target(provisional)"
    elif rtf is not None:
        rsrc = "_rtf_target(provisional)"
    stage = "measured" if rsrc.startswith(str(ROOT)) or rsrc.startswith("results/") else "provisional"

    oav = d.get("owner_accepted_over_30ms")
    if oav is not None and not (isinstance(oav, dict)
                                and all(k in oav for k in ("by", "at", "evidence"))):
        bad.append("owner_accepted_over_30ms は null か {by, at, evidence} の dict")
    lim = 50.0 if oav else 30.0
    ra = d.get("reserve_owner_approved")
    reserve_ok = isinstance(ra, dict) and all(k in ra for k in ("by", "at", "evidence"))

    ok = (qd is not None and rtf is not None and rtf < 1.0 and q == qd
          and q * io / SR >= rtf * N / INT and tot < lim
          and meas and stage == "measured" and reserve_ok and abi_ok and not bad)

    led = {"items_ms": {k: v * 1e3 for k, v in {**cv, **hw}.items()}, "8 hw_ms": ad,
           "L_conv_ms": Lc, "L_hw_ms": Lh, "total_ms": tot,
           "budget_ms": lim, "budget_base_ms": 30.0, "slack_ms": lim - tot,
           "owner_accepted_over_30ms": bool(oav), "pass": ok,
           "io_measured": meas, "io": io, "io_source": src, "q": q, "q_derived": qd,
           "K": K, "d_ahead": D, "d_ahead_source": dsrc, "abi_checked": abi_ok,
           "rtf_target": rtf, "rtf_target_source": rsrc, "gate_stage": stage,
           "content_encoder_reserve_ms": 10.0, "reserve_assumption": "A",
           "reserve_owner_approved": ra,
           "nfft_s": d["nfft_s"], "sinc_len": d["sinc_len"], "resamp_samples": rs,
           "sample_rate": int(SR), "scope": "fullgraph", "abort_reasons": bad}
    return led, bad


def _run(tp: pathlib.Path) -> int:
    d = json.loads(tp.read_text())
    led, bad = build_ledger(d, ROOT / "crates/lightvc-core/contract.toml",
                            ROOT / "results/z0/abi.json")
    out = ROOT / "results/z0/latency_ledger.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(led, ensure_ascii=False, indent=1))
    for k, v in sorted(led["items_ms"].items()):
        print(f"  {k:14s} {v:6.2f} ms")
    print(f"  {'8 hw':14s} {led['8 hw_ms']:6.2f} ms")
    print(f"  合計 {led['total_ms']:.2f} ms / 残予算 {led['slack_ms']:+.2f} ms "
          f"/ pass={led['pass']} / stage={led['gate_stage']}")
    for b in bad:
        print(f"  ⚠ {b}")
    return 0 if led["pass"] else 1


def _fixtures() -> int:
    """fixture ごとに期待値を照合する。**これが唯一の実効カバレッジ**。"""
    fx = ROOT / "results/z0/fixtures"
    if not fx.exists():
        print(f"  ⚠ {fx} が無い。--make-fixtures で作る")
        return 1
    ng = 0
    for f in sorted(fx.glob("*.json")):
        spec = json.loads(f.read_text())
        # ⚠ fixture でも実物の abi.json / contract.toml を渡す。
        #    渡さないと 3 者一致の分岐が fixture から一度も踏まれない。
        # ⚠ fixture は contract.toml の有無を選べる（実環境の未着手状態に縛られない）。
        #    `"contract": "<内容>"` があれば一時ファイルに書いて渡す。
        # ⚠ 既定は「不在」を渡す（実環境と同じく C.5 未着手を既定にする）。
        ct = ROOT / "crates/lightvc-core/contract.toml"
        if "contract" in spec:
            ct = fx / f".{f.stem}.contract.toml"
            ct.write_text(spec["contract"])
        led, bad = build_ledger(spec["input"], ct,
                                ROOT / "results/z0/abi.json")
        if ct and ct.exists():
            ct.unlink()
        for k, want in spec["expect"].items():
            got = led.get(k)
            if isinstance(want, float) and isinstance(got, (int, float)):
                good = abs(got - want) < 0.01
            else:
                good = got == want
            if not good:
                ng += 1
                print(f"  NG  {f.name}: {k} = {got!r}（期待 {want!r}）")
        if "expect_abort_contains" in spec:
            if not any(spec["expect_abort_contains"] in b for b in bad):
                ng += 1
                print(f"  NG  {f.name}: abort 理由に "
                      f"'{spec['expect_abort_contains']}' が無い -> {bad}")
        print(f"  {'ok ' if ng == 0 else '   '} {f.name}")
    print(f"\n  fixture 不一致 {ng} 件")
    return 1 if ng else 0


def main() -> int:
    if "--fixtures" in sys.argv:
        return _fixtures()
    tp = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "results/z0/target_platform.json"
    return _run(tp)


if __name__ == "__main__":
    raise SystemExit(main())
