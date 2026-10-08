"""出荷ゲート: 長時間学習を起動する前に、その推論グラフが本当に流せるか機械で確かめる。

これが無かったせいで 55 時間（gvoc f0 版 27.5h + NHV 版 27.5h）を、ストリーミング
できない front-end の上で使った。mel は centered（先読み 23.2ms）、f0 解析も centered
（同 23.2ms）、さらに発話全体の統計が 4 つ（有声判定の中央値・倍音数 K・包絡の最大値・
レベルの RMS）。どれも「重みを流用して front-end だけ差し替える」ができない種類の依存で、
直せば学習し直しになる。

中核は 1 つの検査だけ。

    未来不変性: 入力の t 以降を書き換えたとき、t より前を担当する出力フレームが
                1 bit も変わらないこと。

これは先読みと発話全体統計を同時に捕まえる。中央値や最大値を発話全体で取っていれば、
末尾を書き換えた瞬間に先頭のフレームまで動くので即座に落ちる。静的解析より強い。

ただし出力が離散なら弱くなる。argmax（f0 の候補格子）や閾値（有声判定）は、依存があっても
出力が動かないことがある。実際、白色雑音プローブでは harmonic_sum_f0 が 1.81 ms と出た
（中で centered 2048 STFT を読んでいるので構造上ありえない）。よって

  - プローブは実音声にする
  - 編集が後半フレームを動かさないなら INCONCLUSIVE として PASS を出さない

の 2 つを必須にする。感度が示せない検査結果は根拠にしない。

使い方:

    from ship_check import future_invariance, ledger
    la = future_invariance(lambda x: my_mel(x), hop=256)
    ledger([("mel", la.ms), ("f0", ...), ...], budget_ms=30.0, reserve_ms=10.0)

`python ship_check.py` で現行部品の実測台帳を出す。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R

ROOT = Path(__file__).resolve().parent.parent

BUDGET_MS = 30.0        # README 正本: 設計目標 p95 < 30 ms
RESERVE_MS = 10.0       # ROADMAP: content encoder の先読み予算
DISQUALIFY_MS = 50.0    # 到達しても不合格。合格ラインではない


@dataclass
class Lookahead:
    samples: int
    ms: float
    unbounded: bool
    inconclusive: bool = False
    lower_only: bool = False

    def __str__(self) -> str:
        if self.unbounded:
            return "UNBOUNDED (発話全体の統計あり = ストリーム不可)"
        if self.inconclusive:
            return "INCONCLUSIVE (編集が出力を動かさない = 感度なし)"
        tag = " (下界: 編集を含むフレームが動かない編集があった = 離散な依存)" if self.lower_only else ""
        return f"{self.samples} sample / {self.ms:.2f} ms{tag}"


def probes(n: int, k: int = 3, seed: int = 0, sr: int | None = None, male: int = 0):
    """実音声のプローブ(女声 k 本 + 男声 male 本・sr で読む)。無ければ雑音に落とすが、
    実音声が無い環境で出した PASS は根拠にしない、が運用。sr 既定は rddsp.SR(44.1k・旧版互換)。"""
    import random
    import librosa
    sr = sr or R.SR
    out = []

    def take(dirs, cnt):
        got = []
        rng = random.Random(seed)
        for d in rng.sample(dirs, min(len(dirs), 4 * cnt)):
            ws = sorted(d.glob("*.wav"))
            if not ws:
                continue
            x, _ = librosa.load(str(ws[0]), sr=sr, mono=True, duration=max(6.0, n / sr + 0.1))
            if len(x) >= n:
                step = max(1, sr // 10)
                e = [float((x[i:i + n] ** 2).mean()) for i in range(0, len(x) - n + 1, step)]
                i0 = step * int(np.argmax(e))
                got.append(torch.tensor(x[i0:i0 + n]))
            if len(got) >= cnt:
                break
        return got
    for r in (ROOT / "female-dataset", ROOT / "data/female_tts_corpus"):
        if r.exists() and len(out) < k:
            out += take(sorted(p for p in r.iterdir() if p.is_dir()), k - len(out))
    if male:
        mdirs = []
        for r in (ROOT / "data/male_tts_corpus", ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"):
            if r.exists():
                mdirs += sorted(p for p in r.iterdir() if p.is_dir())
        out += take(mdirs, male)
    g = torch.Generator().manual_seed(seed)
    return out or [torch.randn(n, generator=g) * 0.1]


def _residues(hop: int) -> list[int]:
    r = {0, hop - 1} | {min(hop - 1, 1 << k) for k in range(12) if (1 << k) < hop} | {hop * j // 8 for j in range(1, 8)}
    return sorted(r)


def future_invariance(fn, hop: int, n: int = None, sr: int = None,
                      n_edit: int = 4, seed: int = 0, quantity: bool = True, tol: float = 0.0,
                      probe_list: list | None = None, male: int = 0) -> Lookahead:
    """fn: [n] -> [..., T]。入力の末尾を書き換えて、どこまで過去のフレームが汚れるか測る。

    フレーム t が「担当する」最後の入力サンプルを t*hop + hop - 1 と定義する（その hop を
    出し終えた時点で t を出せるのが 0 先読み）。編集点 c 以降を潰したとき変化した最小の
    フレーム t について c - (t*hop + hop - 1) が、そのフレームが待った未来の量。

    書き換えは雑音と無音の 2 通り(閾値型の判定は雑音では反転しないことがある = 偽 PASS の型)。
    比較は既定で 1 bit(tol = 0)。旧版の 1e-5 は窓の裾(2.4e-6)を見逃した(2026-10-01 レビュー 2)。
    編集点の hop 内の位置(余り r)で見える量が変わる: 余り r の編集は先読み la > r を r+1+hop·⌊(la−r−1)/hop⌋ と
    報告する(下界)。余り 0 は任意の la ≥ 1 を検出する(0 判定は余り 0 だけで足りる: quantity=False)。
    quantity=True は複数の余りで測り、連続な依存なら [真値, 真値 + 余りの間隔) の上界を samples に返す。
    離散な依存(閾値・argmax)では編集を含むフレームすら動かない編集があり、その場合は観測の最大値を
    下界として返し lower_only を立てる(2026-10-01 レビュー 2: 閾値型 120/240/480 → 60/180/390)。"""
    sr = sr or R.SR
    n = n or 2 * sr
    g = torch.Generator().manual_seed(seed)
    res = _residues(hop) if quantity else [0]
    # プローブは sr で読む(旧版は 44.1k 固定)。male > 0 で男声を加える(2026-10-01 レビュー 3)。
    # 事象型(GCI・ピーク検出)の依存は編集点の数 n_edit が少ないと見逃す(n_edit=4 で真 360 → 211 の下界)。
    # その種の経路の合否は「出力を DELAY だけ遅らせた関数が quantity=False・n_edit ≥ 150 で 0」で判定する。
    worst, sensitive, blind = -10 ** 9, False, 0
    for x in (probe_list if probe_list is not None else probes(n, seed=seed, sr=sr, male=male)):
        x = x[:n]
        with torch.no_grad():
            y0 = fn(x)
        y0 = y0.reshape(-1, y0.shape[-1])
        T = y0.shape[-1]
        for i in range(n_edit):
            c0 = int(n * (0.35 + 0.6 * i / max(n_edit - 1, 1))) // hop * hop
            for r, kind in ((r, k) for r in res for k in ("noise", "zero")):
                c = c0 + r
                if c >= n:
                    continue
                xe = x.clone()
                xe[c:] = torch.randn(n - c, generator=g) * float(x.std()) if kind == "noise" else 0.0
                with torch.no_grad():
                    y1 = fn(xe)
                y1 = y1.reshape(-1, y1.shape[-1])
                d = (y0[:, :T] - y1[:, :T]).abs().amax(0)
                idx = (d > tol).nonzero()
                if idx.numel() == 0:
                    continue          # この編集では何も動かない = 感度の証拠にならない
                sensitive = True
                if c // hop < T and float(d[c // hop]) <= tol:
                    blind += 1
                t = int(idx[0])
                if t == 0:
                    return Lookahead(0, 0.0, True)
                worst = max(worst, c - (t * hop + hop - 1))
    if not sensitive:
        return Lookahead(0, 0.0, False, inconclusive=True)
    lo = max(worst, 0)
    if lo == 0:
        return Lookahead(0, 0.0, False)
    if blind or not quantity:
        return Lookahead(lo, 1000.0 * lo / sr, False, lower_only=True)
    rs = (lo - 1) % hop
    nxt = next((q for q in res if q > rs), hop + res[0])
    hi = lo + nxt - rs - 1
    return Lookahead(hi, 1000.0 * hi / sr, False)


def _selftest_quantization() -> None:
    """既知の先読みの fixture: 連続(abs の和)は [真値, 真値 + hop/8)・0 は 0。離散(閾値)は 0 と >0 を取り違えず、
    量は下界(lower_only)。微小な依存(重み 4e-6 の裾 = 旧版の 1e-5 比較では見えない大きさ)も 1 bit 比較で検出する。
    (float32 の丸めより小さい依存は原理的に検出できない。)"""
    hop = 240
    noise = [torch.randn(24000, generator=torch.Generator().manual_seed(s)) * 0.1 for s in range(2)]

    def cont(la):
        def fn(x):
            m = len(x) // hop * hop
            xx = torch.cat([x[:m], torch.zeros(la)])
            return torch.stack([xx[t * hop:(t + 1) * hop + la].abs().sum() for t in range(m // hop)])[None]
        return fn

    def disc(la):
        def fn(x):
            m = len(x) // hop * hop
            xx = torch.cat([x[:m], torch.zeros(la)])
            return torch.stack([(xx[t * hop + la:(t + 1) * hop + la].abs().max() > 0.25).float() + (xx[t * hop:(t + 1) * hop].abs().max() > 0.25).float()
                                for t in range(m // hop)])[None]
        return fn

    def tail(la):
        def fn(x):
            m = len(x) // hop * hop
            xx = torch.cat([x[:m], torch.zeros(la)])
            return torch.stack([xx[t * hop:(t + 1) * hop].abs().mean() + 4e-6 * xx[(t + 1) * hop:(t + 1) * hop + la].sum()
                                for t in range(m // hop)])[None]
        return fn
    for la in (0, 1, 7, 60, 119, 120, 239, 240, 241, 480, 700):
        r = future_invariance(cont(la), hop=hop, n=24000, sr=48000, probe_list=noise)
        assert (r.samples == 0) if la == 0 else (la <= r.samples < la + hop // 8), ("cont", la, r)
        z = future_invariance(cont(la), hop=hop, n=24000, sr=48000, probe_list=noise, quantity=False)
        assert (z.samples == 0) == (la == 0), ("zero-mode", la, z)
        print(f"  連続 真 {la:4d} → {r.samples:4d}  / 0 判定 {'0' if z.samples == 0 else '>0'}")
    for la in (0, 120, 240, 480):
        r = future_invariance(disc(la), hop=hop, n=24000, sr=48000, probe_list=noise)
        assert (r.samples == 0) == (la == 0) and r.samples <= la + hop // 8, ("disc", la, r)
        print(f"  離散 真 {la:4d} → {r}")
    for la in (0, 240):
        r = future_invariance(tail(la), hop=hop, n=24000, sr=48000, probe_list=noise, quantity=False)
        assert (r.samples == 0) == (la == 0), ("tail", la, r)
        old = future_invariance(tail(la), hop=hop, n=24000, sr=48000, probe_list=noise, quantity=False, tol=1e-5)
        print(f"  微小な裾 真 {la:4d} → {r}  /  旧 1e-5 比較なら {old}")
    m = len(noise[0]) // hop * hop
    r480 = future_invariance(disc(480), hop=hop, n=24000, sr=48000, probe_list=noise)
    assert r480.lower_only and not ledger([("閾値型 真 480", r480)]), "下界は台帳で FAIL"
    rz = future_invariance(cont(240), hop=hop, n=24000, sr=48000, probe_list=noise, quantity=False)
    assert rz.lower_only and not ledger([("0 判定モード 真 240", rz)]), "0 判定モードの非 0 は台帳で FAIL"
    assert ledger([("連続 真 0", future_invariance(cont(0), hop=hop, n=24000, sr=48000, probe_list=noise))])
    print("ship_check quantization selftest OK")


def ledger(entries, budget_ms: float = BUDGET_MS,
           reserve_ms: float = RESERVE_MS) -> bool:
    """entries = [(name, lookahead_ms or Lookahead), ...]。並列枝は名前を同じにする。"""
    print(f"\n  静的遅延台帳（設計枠 {budget_ms:.0f} ms / content encoder 予備 "
          f"{reserve_ms:.0f} ms / 失格 {DISQUALIFY_MS:.0f} ms）")
    print("  " + "-" * 62)
    total, bad = 0.0, False
    for name, la in entries:
        ms = la.ms if isinstance(la, Lookahead) else float(la)
        un = isinstance(la, Lookahead) and la.unbounded
        inc = isinstance(la, Lookahead) and la.inconclusive
        low = isinstance(la, Lookahead) and la.lower_only and la.samples > 0
        bad = bad or un or inc or low
        total += 0.0 if (un or inc) else ms
        tag = "UNBOUNDED" if un else ("INCONCLUSIVE" if inc else (f"≥{ms:8.2f} ms" if low else f"{ms:9.2f} ms"))
        why = ("   <-- ストリーム不可" if un else
               "   <-- 感度なし・根拠にしない" if inc else
               "   <-- 下界(真値はこれ以上)・台帳に足せない" if low else "")
        print(f"  {name:34s} {tag}{why}")
    print("  " + "-" * 62)
    print(f"  {'framing 小計':34s} {total:9.2f} ms")
    print(f"  {'content encoder 予備':34s} {reserve_ms:9.2f} ms")
    print(f"  {'合計':34s} {total + reserve_ms:9.2f} ms")
    ok = (not bad) and (total + reserve_ms) < budget_ms
    print(f"\n  判定: {'PASS' if ok else 'FAIL'}"
          f"  残予算 {budget_ms - total - reserve_ms:+.2f} ms")
    if bad:
        print("  FAIL 理由: 未来不変性が破れている、感度が示せていない、または先読みの量が下界しか分からない"
              "(離散な依存・quantity=False の非 0)。下界は真値として足さない(2026-10-01 レビュー 4)")
    return ok


def main() -> None:
    import ship_front as SF
    from rddsp_neural import mel_of

    print("\n=== 旧 front-end（gvoc_nhv が学習した構成）===")
    for name, fn in (("mel_of (librosa centered 2048)", lambda x: mel_of(x)),
                     ("rddsp.stft (centered 2048)", lambda x: R.stft(x).abs()),
                     ("rddsp.harmonic_sum_f0", lambda x: R.harmonic_sum_f0(x)[0][None])):
        print(f"  {name:34s} {future_invariance(fn, R.HOP)}")

    print("\n=== 出荷 front-end（ship_front）===")
    rows = []
    for name, fn, hop in (
            ("ship_front.mel", lambda x: SF.mel(x), SF.HOP_A),
            ("ship_front.causal_f0", lambda x: SF.causal_f0(x)[0][None], SF.HOP_A),
            ("ship_front.nhv_spec", _prior_probe, SF.HOP_S)):
        la = future_invariance(fn, hop)
        print(f"  {name:34s} {la}")
        rows.append((name, la))
    rows.append(("出力 iSTFT の重畳 (512-128)",
                 1000.0 * (SF.NFFT_S - SF.HOP_S) / R.SR))
    ok = ledger(rows)
    print()
    return ok


def main_ok() -> bool:
    """学習起動前のガード。CLAUDE.md「Shipping Gate」。"""
    return bool(main())


def _prior_probe(x):
    """事前分布は mel と f0 の関数。入力波形からの経路全体で測る。"""
    import ship_front as SF
    import train_gvoc as TG
    W = TG.mel_to_linear(x.device)
    m = SF.mel(x)
    T = x.shape[-1] // SF.HOP_S + 1
    g = torch.Generator(device=x.device).manual_seed(0)
    P = SF.nhv_spec(SF.to_frames(W @ m, T), SF.causal_f0(x)[0], x.shape[-1], g, T=T)
    return P.abs()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
