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

    def __str__(self) -> str:
        if self.unbounded:
            return "UNBOUNDED (発話全体の統計あり = ストリーム不可)"
        if self.inconclusive:
            return "INCONCLUSIVE (編集が出力を動かさない = 感度なし)"
        return f"{self.samples} sample / {self.ms:.2f} ms"


def probes(n: int, k: int = 3, seed: int = 0):
    """実音声のプローブ。無ければ雑音に落とすが、その旨を呼び出し側が知る必要はない
    ―― 実音声が無い環境で出した PASS は根拠にしない、が運用。"""
    import random
    out = []
    for r in (ROOT / "female-dataset", ROOT / "data/female_tts_corpus"):
        if not r.exists():
            continue
        spks = sorted(p for p in r.iterdir() if p.is_dir())
        rng = random.Random(seed)
        for d in rng.sample(spks, min(k, len(spks))):
            ws = sorted(d.glob("*.wav"))
            if not ws:
                continue
            import librosa
            x, _ = librosa.load(str(ws[0]), sr=R.SR, mono=True, duration=3.0)
            if len(x) >= n:
                out.append(torch.tensor(x[:n]))
            if len(out) >= k:
                return out
    g = torch.Generator().manual_seed(seed)
    return out or [torch.randn(n, generator=g) * 0.1]


def future_invariance(fn, hop: int, n: int = None, sr: int = None,
                      n_edit: int = 6, seed: int = 0) -> Lookahead:
    """fn: [n] -> [..., T]。入力の末尾を書き換えて、どこまで過去のフレームが汚れるか測る。

    フレーム t が「担当する」最後の入力サンプルを t*hop + hop - 1 と定義する（その hop を
    出し終えた時点で t を出せるのが 0 先読み）。編集点 c 以降を潰したとき変化した最小の
    フレーム t について c - (t*hop + hop - 1) が、そのフレームが待った未来の量。"""
    sr = sr or R.SR
    n = n or 2 * sr
    g = torch.Generator().manual_seed(seed)
    worst, sensitive = -10 ** 9, False
    for x in probes(n, seed=seed):
        with torch.no_grad():
            y0 = fn(x)
        y0 = y0.reshape(-1, y0.shape[-1])
        T = y0.shape[-1]
        for i in range(n_edit):
            c = int(n * (0.35 + 0.6 * i / max(n_edit - 1, 1)))
            xe = x.clone()
            xe[c:] = torch.randn(n - c, generator=g) * float(x.std())
            with torch.no_grad():
                y1 = fn(xe)
            y1 = y1.reshape(-1, y1.shape[-1])
            d = (y0[:, :T] - y1[:, :T]).abs().amax(0)
            idx = (d > 1e-5).nonzero()
            if idx.numel() == 0:
                continue          # この編集では何も動かない = 感度の証拠にならない
            sensitive = True
            t = int(idx[0])
            if t == 0:
                return Lookahead(0, 0.0, True)
            worst = max(worst, c - (t * hop + hop - 1))
    if not sensitive:
        return Lookahead(0, 0.0, False, inconclusive=True)
    return Lookahead(max(worst, 0), 1000.0 * max(worst, 0) / sr, False)


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
        bad = bad or un or inc
        total += 0.0 if (un or inc) else ms
        tag = "UNBOUNDED" if un else ("INCONCLUSIVE" if inc else f"{ms:9.2f} ms")
        why = ("   <-- ストリーム不可" if un else
               "   <-- 感度なし・根拠にしない" if inc else "")
        print(f"  {name:34s} {tag}{why}")
    print("  " + "-" * 62)
    print(f"  {'framing 小計':34s} {total:9.2f} ms")
    print(f"  {'content encoder 予備':34s} {reserve_ms:9.2f} ms")
    print(f"  {'合計':34s} {total + reserve_ms:9.2f} ms")
    ok = (not bad) and (total + reserve_ms) < budget_ms
    print(f"\n  判定: {'PASS' if ok else 'FAIL'}"
          f"  残予算 {budget_ms - total - reserve_ms:+.2f} ms")
    if bad:
        print("  FAIL 理由: 未来不変性が破れている、または感度が示せていない")
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
