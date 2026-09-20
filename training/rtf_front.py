"""手順 0.5: front-end の段別 RTF と causal_f0 融合版の検収。

    uv run python rtf_front.py            # 段別 RTF -> results/z0/rtf_front.json
    uv run python rtf_front.py --equiv N  # 融合版の検収を N 発話で追加実行
    uv run python rtf_front.py --equiv 400 --shard-from 4   # 別の 400 発話で反転検証

2.1 の実装を待たずに走る（手順 0.5 が 2.1 の成果物に依存すると循環する）。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

import rddsp as R
import ship_front as SF
from rddsp_gpu import mel_to_linear

K = 2
BLK = K * SF.HOP_S
IN_HIST = 1792
EXC_HIST = SF.NFFT_S - SF.HOP_S
RT = BLK / 44100
THREADS = 2
OUT = Path(__file__).resolve().parent.parent / "results/z0/rtf_front.json"
HIST = OUT.parent / "rtf_front_history.jsonl"
MIN_RUNS = 3


def spread_from_history(cur: float) -> tuple[float | None, int]:
    """ぶれ幅は**プロセス間**で測る。プロセス内の反復は過小評価する。

    実測: 同一プロセス内 3 回の幅は 0.0019〜0.0073 だが、起動をまたぐと
    0.127〜0.143 ＝ 0.016。**8 倍違う**（熱・キャッシュ・アロケータの状態が
    プロセス内では共有されるため）。プロセス内の幅で合否を引くと、
    同じコードが起動ごとに INCONCLUSIVE と PASS を行き来する（実際に起きた）。
    """
    xs = []
    if HIST.exists():
        for ln in HIST.read_text().splitlines():
            if ln.strip():
                xs.append(json.loads(ln)["front_end_fused"])
    xs.append(cur)
    if len(xs) < MIN_RUNS:
        return None, len(xs)
    return round(max(xs) - min(xs), 4), len(xs)


def build_matrix():
    """TRAINING_PLAN §5.3 の [ncand x NB] 行列。起動時に 1 回だけ作る。"""
    nb = SF.NFFT_A // 2 + 1
    binhz = R.SR / SF.NFFT_A
    ncand = int(math.log2(SF.F0_MAX / SF.F0_MIN) * 1200 / 10) + 1
    cand = SF.F0_MIN * 2 ** (torch.arange(ncand, dtype=torch.float32) * 10 / 1200)
    ks = torch.arange(1, 21, dtype=torch.float32)
    allf = torch.cat([cand[None] * ks[:, None],
                      cand[None] * (ks[:, None] + 0.5)], 0).reshape(-1)
    w = torch.cat([(1 / ks.sqrt())[:, None].expand(20, ncand),
                   (-0.5 / ks.sqrt())[:, None].expand(20, ncand)], 0).reshape(-1)
    ok = ((allf < R.SR / 2 - binhz) & (allf > 0)).float()
    bb = (allf / binhz).clamp(0, nb - 2)
    lo = bb.long()
    fr = bb - lo
    ci = torch.arange(40 * ncand) % ncand
    m = torch.zeros(ncand, nb)
    m.index_put_((ci, lo), w * ok * (1 - fr), accumulate=True)
    m.index_put_((ci, lo + 1), w * ok * fr, accumulate=True)
    return m, cand, nb


MAT, CAND, NB = build_matrix()


def causal_f0_fused(x: torch.Tensor):
    mag = SF.cstft(x, SF.NFFT_A, SF.HOP_A).abs()
    sc = MAT @ mag
    idx = sc.argmax(0)
    f0 = CAND[idx]
    voi = sc.gather(0, idx[None])[0] / (mag.sum(0) / math.sqrt(NB) + R.EPS)
    return torch.where(voi > SF.VOI_ABS, SF._causal_median(f0, 5),
                       torch.zeros_like(f0)), voi


def _osc(fl, acc):
    p = torch.arange(BLK, dtype=torch.float64)
    t = (p + 1.0) / SF.HOP_A - 2.0
    i = t.floor().clamp(0, len(fl) - 2)
    fr = (t - i).clamp(0.0, 1.0)
    # ⚠ 補間の第 2 項を落とさない。製品の frame_upsample_causal / prior_stream_ref は
    #   fl[j]*(1-fr) + fl[j+1]*fr。計測値は変わらないが測定道具が製品と別物になる。
    _fl = torch.tensor(fl, dtype=torch.float64)
    j = i.long().clamp(max=len(fl) - 2)
    f0u = (_fl[j] * (1 - fr) + _fl[j + 1] * fr).clamp(min=0.0)
    a = torch.cumsum(f0u, 0) + acc
    phi = torch.remainder(2 * math.pi * a / R.SR, 2 * math.pi).float()
    fu = f0u.float()
    imp, cnt = torch.zeros(BLK), torch.zeros(BLK)
    for s0 in range(0, SF.KMAX, 32):
        kk = torch.arange(s0 + 1, min(s0 + 33, SF.KMAX + 1),
                          dtype=torch.float32)[:, None]
        m = (kk * fu[None] < R.SR / 2).float()
        imp = imp + (torch.cos(kk * phi[None]) * m).sum(0)
        cnt = cnt + m.sum(0)
    return imp / cnt.clamp(min=1.0).sqrt()


def bench(fn, n: int = 40) -> float:
    for _ in range(8):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[n // 2] / RT


def stages_repeated(n: int = 3) -> dict[str, list[float]]:
    """n 回走らせて段別 RTF の分布を返す。

    1 回の値を pin すると、**合否がマシン負荷で反転する**。実測: 同一機で
    front-end 融合後が 0.127〜0.140（9%）ぶれ、net 予算 0.21〜0.22。
    合格判定に使う余裕（0.01）より広い。∴ **合否は最悪値で読む**。
    """
    runs = [stages() for _ in range(n)]
    return {k: [r[k] for r in runs] for k in runs[0]}


def stages() -> dict[str, float]:
    w = mel_to_linear("cpu")
    buf = torch.randn(IN_HIST + BLK) * 0.05
    fl = [100.0] * 20
    acc = torch.zeros((), dtype=torch.float64)
    eb = torch.cat([torch.zeros(EXC_HIST), _osc(fl, acc)])
    e = SF.cstft(eb, SF.NFFT_S, SF.HOP_S)[:, EXC_HIST // SF.HOP_S:
                                          EXC_HIST // SF.HOP_S + K]
    ml = SF.to_frames(SF.mel(buf), SF.n_frames(buf.shape[-1]))[:, :K]
    return {
        "causal_f0_current": bench(lambda: SF.causal_f0(buf)),
        "causal_f0_fused": bench(lambda: causal_f0_fused(buf)),
        "osc": bench(lambda: _osc(fl, acc)),
        "mel": bench(lambda: SF.mel(buf)),
        "cistft": bench(lambda: SF.cistft(e, BLK)),
        "exc_cstft": bench(lambda: SF.cstft(eb, SF.NFFT_S, SF.HOP_S)),
        "envelope": bench(lambda: e * (w @ ml - SF.MEL_REF).exp()),
    }


def equiv(n_utt: int, shard_from: int = 0) -> dict:
    import train_gvoc as TG
    sh = sorted(TG.SHARDS.glob("sh_*.pt"))[shard_from:]
    frames = bad = 0
    worst = 0.0
    bad_utt = 0
    seen = 0
    oct_cur = oct_fus = pairs = pairs_f = 0
    for f in sh:
        for it in torch.load(f, map_location="cpu", weights_only=False):
            if seen >= n_utt:
                break
            seen += 1
            y = it["w"].float() / 32767.0
            a, _ = SF.causal_f0(y)
            b, _ = causal_f0_fused(y)
            d = (a - b).abs()
            frames += a.shape[-1]
            k = int(d.gt(0).sum())
            bad += k
            if k:
                bad_utt += 1
                worst = max(worst, float(d.max()))
            for src, box in ((a, "c"), (b, "f")):
                v = src[src > 0]
                if v.numel() < 2:
                    continue
                r = v[1:] / v[:-1]
                j = int((((r > 1.8) & (r < 2.2)) | ((r > 0.45) & (r < 0.55))).sum())
                if box == "c":
                    oct_cur += j
                    pairs += r.numel()
                else:
                    oct_fus += j
                    pairs_f += r.numel()
        if seen >= n_utt:
            break
    return {"utterances": seen, "frames": frames, "mismatch_frames": bad,
            "mismatch_rate": bad / max(frames, 1), "mismatch_utterances": bad_utt,
            "max_abs_diff_hz": worst, "voiced_pairs": pairs, "voiced_pairs_fused": pairs_f,
            "octave_jumps_current": oct_cur, "octave_jumps_fused": oct_fus,
            "shard_from": shard_from}


def main() -> int:
    torch.set_num_threads(THREADS)
    nrep = 3
    if "--repeat" in sys.argv:
        nrep = int(sys.argv[sys.argv.index("--repeat") + 1])
    d = stages_repeated(nrep)
    v = {k: sorted(x)[len(x) // 2] for k, x in d.items()}          # 中央値（表示用）
    w = {k: max(x) for k, x in d.items()}                          # 最悪値（判定用）
    cur = sum(x for k, x in w.items() if k != "causal_f0_fused")
    fus = sum(x for k, x in w.items() if k != "causal_f0_current")
    med = sum(x for k, x in v.items() if k != "causal_f0_current")
    within = round(fus - sum(min(x) for k, x in d.items()
                             if k != "causal_f0_current"), 4)
    spread, nruns = spread_from_history(fus)
    rep = {"threads": THREADS, "block_samples": BLK, "block_ms": RT * 1e3,
           "repeats": nrep,
           "stages": {k: round(x, 4) for k, x in v.items()},
           "stages_worst": {k: round(x, 4) for k, x in w.items()},
           "front_end_current": round(cur, 4), "front_end_fused": round(fus, 4),
           "front_end_fused_median": round(med, 4),
           "front_end_spread": spread,          # プロセス間（判定用。None=まだ n 不足）
           "front_end_spread_within": within,    # プロセス内（参考。過小評価する）
           "spread_runs": nruns, "spread_min_runs": MIN_RUNS,
           "net_budget": round(0.35 - fus, 4),
           "candidate0": round(w["mel"] + w["cistft"], 4)}
    if OUT.exists():
        try:
            _prev = json.loads(OUT.read_text()).get("equivalence")
        except Exception:                             # noqa: BLE001
            _prev = None
        if isinstance(_prev, dict):
            _prev = [_prev]
        rep["equivalence"] = _prev or []
    else:
        rep["equivalence"] = []
    if "--equiv" in sys.argv:
        i = sys.argv.index("--equiv")
        nxt = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        if not nxt.isdigit():
            raise SystemExit("--equiv には発話数を明示する（例: --equiv 400）。"
                             "既定 400 への黙った fallback は「何本で測ったか」を消す")
        n = int(nxt)
        # ⚠ 検収 3 の「別 400 発話での反転検証」に要る。CLI に無いと実行不能だった。
        sf_i = sys.argv.index("--shard-from") if "--shard-from" in sys.argv else -1
        sfrom = int(sys.argv[sf_i + 1]) if sf_i >= 0 else 0
        _e = equiv(n, sfrom)
        # ⚠ 上書きしない。shard_from 別に追記マージする（独立標本を 2 つ残すため）
        rep["equivalence"] = [x for x in rep["equivalence"]
                              if x.get("shard_from") != sfrom] + [_e]
    for k, x in v.items():
        print(f"  {k:20s} RTF {x:.4f}")
    print(f"  {'front-end 現行':20s} RTF {cur:.4f}")
    print(f"  {'front-end 融合後':20s} RTF {fus:.4f}（最悪値・{nrep} 回）"
          f"  中央値 {med:.4f}  プロセス内 {within:.4f}")
    sp = "未確定（起動 %d/%d 回）" % (nruns, MIN_RUNS) if spread is None else f"{spread:.4f}"
    print(f"  {'ぶれ幅（プロセス間）':20s} {sp}   ← 判定はこちら")
    print(f"  {'net 予算':20s} {0.35 - fus:.4f}"
          f"   ⚠ 余裕がぶれ幅未満なら INCONCLUSIVE（n<{MIN_RUNS} も INCONCLUSIVE）")
    for e in rep.get("equivalence", []):
        print(f"  等価[shard{e['shard_from']}]: {e['mismatch_frames']}/{e['frames']} フレーム "
              f"({e['mismatch_rate']*100:.4f}%) 最大 {e['max_abs_diff_hz']:.2f} Hz "
              f"／ オクターブ跳び 現行 {e['octave_jumps_current']} vs 融合 "
              f"{e['octave_jumps_fused']}（{e['voiced_pairs']} 対）")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with HIST.open("a") as fh:
        fh.write(json.dumps({"front_end_fused": fus, "repeats": nrep,
                             "threads": THREADS}) + "\n")
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
