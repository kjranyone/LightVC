"""2.4b の net RTF 実測。**計画値を関門に渡さないための唯一の作り手**。

`plan_numbers.measured_net()` が読む `results/z0/rtf_K{K}_T{T}_d{dim}_L{L}_n{nbin}.json`
を書く。判定は `plan_numbers.rtf_verdict()`（凍結許容差つき）が下す。

    uv run python rtf_bench.py --dim 256 --layers 6 --k 3 --k-in 7 --repeat 30

⚠ 最悪値を採る。中央値で読むと合否がマシン負荷で反転する（実測 0.127〜0.143）。
⚠ 判定は本スクリプトでは出さない。閾値との突合せは凍結側（規則 5）に任せる。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch

import ship_front as SF
import v1d

ROOT = pathlib.Path(__file__).resolve().parent.parent


def bench(net, span: int, emit: int, repeat: int, warmup: int = 5) -> float:
    """1 ブロック（新規 `emit` フレーム）を出すコストを、その `emit` フレームぶんの
    音声時間で割る。

    ⚠ **`span` ぶんの音声で割らない**。それはスループットであってブロック実行の
    コストではない（初版はそれを書いて 0.0397 という 5 倍楽な値を出した。
    `memory/benchmark-degeneracy-gate` の形）。
    """
    # ⚠ **層ごとキャッシュ（2c）で測る**。キャッシュ無しの全再計算は
    #    未完成実装であり、その値で方式を否定しない。
    st = v1d.V1DStream(net)
    mel = torch.randn(SF.N_MEL, emit)
    P = torch.randn(net.nbin, emit) + 1j * torch.randn(net.nbin, emit)
    for _ in range(warmup):
        st.step(mel, P)
    worst = 0.0
    audio = emit * SF.HOP_S / 44100.0
    for _ in range(repeat):
        t0 = time.perf_counter()
        st.step(mel, P)
        worst = max(worst, (time.perf_counter() - t0) / audio)
    return worst


def gemm_floor(net, emit: int, threads: int, rep: int = 400) -> float:
    """K=emit の**実形状**で GEMM だけを積む。Rust/Candle の下限に相当する。

    ⚠ 3 つの基準はどれも単独では製品を代表しない。**選ぶのは走行前の宣言**
    （`results/z0/rtf_basis.json`）であって、測ってから都合のよい方を採らない（規則 5）。
      - `pytorch_e2e`  : Python の op 起動費込み。**Rust には存在しない費用**を含む
      - `blas_floor`   : 実形状 GEMM のみ。elementwise を含まない下限
      - `marginal`     : 大 emit の傾き。**K=2 の小行列効率を反映しないので楽すぎる**
    """
    n = net
    shapes = [(emit, n.cin * n.k_in, n.dim)]
    for _ in range(n.layers):
        shapes += [(emit, n.dim * n.k, n.dim), (emit, n.dim, n.dim * 3),
                   (emit, n.dim * 3, n.dim)]
    shapes.append((emit, n.dim, 2 * n.nbin))
    tot = 0.0
    for m, k, o in shapes:
        A, B = torch.randn(m, k), torch.randn(k, o)
        for _ in range(30):
            A @ B
        w = 0.0
        for _ in range(rep):
            t0 = time.perf_counter()
            A @ B
            w = max(w, time.perf_counter() - t0)
        tot += w
    return tot / (emit * SF.HOP_S / 44100.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--k-in", type=int, default=7)
    ap.add_argument("--block", type=int, default=2, help="K: 1 ブロックの合成フレーム数")
    ap.add_argument("--T", type=int, default=2, help="推論 1 回で出す合成フレーム数")
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4, help="target_cpu 4C8T 相当")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    net = v1d.V1D(dim=a.dim, L=a.layers, k_in=a.k_in, k=a.k).eval()
    # ⚠ ブロック実行はリングの左文脈込みで走る。CTX を無視して測ると過小評価になる。
    span = a.T + net.ctx
    rtf = bench(net, span, a.T, a.repeat)

    out = ROOT / "results/z0" / (f"rtf_K{a.block}_T{a.T}_d{a.dim}"
                                 f"_L{a.layers}_n{net.nbin}.json")
    floor = gemm_floor(net, a.T, a.threads)
    basis_f = ROOT / "results/z0/rtf_basis.json"
    basis = json.loads(basis_f.read_text())["basis"] if basis_f.exists() else None
    rec = {"net_rtf": round({"blas_floor": floor}.get(basis, rtf), 4),
           "net_rtf_pytorch_e2e": round(rtf, 4),
           "net_rtf_blas_floor": round(floor, 4),
           "basis": basis,
           "impl": "V1DStream(層ごとキャッシュ)", "K": a.block, "T": a.T, "dim": a.dim,
           "L": a.layers, "k": a.k, "k_in": a.k_in, "nbin": net.nbin,
           "ctx": net.ctx, "span_frames": span, "repeat": a.repeat,
           "threads": a.threads, "worst_of": a.repeat,
           "note": "最悪値。中央値で読まない（負荷で合否が反転する）"}
    # ⚠ 起動をまたいだ履歴を持ち、**最悪値**で判定する（front-end と同じ規律。
    #    同一プロセス内の反復はぶれ幅を過小評価する）。
    hist = ROOT / "results/z0/rtf_net_history.jsonl"
    with hist.open("a") as fh:
        fh.write(json.dumps({k: rec[k] for k in
                             ("net_rtf", "net_rtf_pytorch_e2e",
                              "net_rtf_blas_floor", "basis", "dim", "L", "k",
                              "k_in", "nbin")}, ensure_ascii=False) + "\n")
    same = [json.loads(x) for x in hist.read_text().splitlines() if x.strip()]
    same = [r for r in same if r.get("basis") == basis and r.get("dim") == a.dim
            and r.get("L") == a.layers and r.get("k") == a.k]
    rec["net_rtf"] = round(max(r["net_rtf"] for r in same), 4)
    rec["runs"] = len(same)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"  pytorch_e2e {rtf:.4f} ／ blas_floor {floor:.4f}"
          f"  採用基準 {basis}  -> {out.name}")
    print("  ⚠ 合否は `uv run python plan_numbers.py` の判定行が正本")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
