"""手順 1 の耳ゲート: 先読み 0（path B）の音が商品として可か。

PROCEDURE.md 1.1-1.6。**Z0-C 条件 1 ではない**——正典 §7.2 条件 1 は path A 比の劣化なし
（比較判定）だが、本手順は絶対許容度で数える（path A は先読み UNBOUNDED で製品に出せない）。**新しい学習を 1 本も起動する前に、ディスクにある重みで判定できる**ので
最初に置く（`current/TRAINING_PLAN.md` §2）。

  gt        実音声
  pathA     gvoc_nhv  centered mel(2048) + centered f0 解析 + 発話全体統計 4 個
            → 先読み 29.0 ms。**製品には出せない**が、品質の上限側の係留として聴く
  pathB     gvoc_ship 左寄せ mel(1024) + 左寄せ f0 + 発話統計 0 個
            → 先読み 0 ms。台帳 18.71 ms・残予算 +11.29 ms（ship_check PASS）

**試行ごとに割り当てを再ランダム化する。** 2026-08-05 に「試行ごとに振ったら判定不能になった」
前科があるが、真因は割り当てではなく**回答形式**だった（文字単位の総評を求めたため
「Z が金属的」が 3 系にまたがった）。回答を**試行ごとの順位**にすれば両立する。

単一割り当てにすると内部対照が壊れる: t0 で gt の文字を同定した時点で、残りの試行は
その文字を 1 位に書くだけで自動的に「有効」になり、**gt 対照が判定者の記憶を測るだけ**になる。

gt を内部対照として混ぜ、gt を 1 位に置けない試行はその試行ごと無効。
RMS を gt に揃えるので音量では判別できない。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
import ship_front as SF
import train_gvoc as TG
from rddsp_gpu import CACHE_DIR, safe_score
from rddsp_hf import HOP, Wavehax2D

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/z0c_ear")
N_TRIAL = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 11
SYSTEMS = ["gt", "pathA", "pathB"]
LETTER = ["X", "Y", "Z"]


def load(tag: str, dev: str):
    ck = torch.load(CACHE_DIR / f"{tag}_ch96_s0.pt", map_location=dev,
                    weights_only=False)
    net = Wavehax2D(cin=4, ch=ck["args"]["ch"], layers=ck["args"]["layers"]).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck


@torch.no_grad()
def render_pathA(net, it, dev, gen):
    """gvoc_nhv が学習した通りの経路（centered mel・発話統計あり）。"""
    TG.PRIOR_FN = TG.nhv_prior
    n = it["w"].shape[-1]
    pw = TG.nhv_prior(it, gen, dev, float(it["w"].std()))
    f, _P = TG.feats(it, pw)
    o = net(f[None])[0]
    return TG.istft(torch.complex(o[0], o[1]), n).cpu().numpy()


@torch.no_grad()
def render_pathB(net, w, dev, gen, W):
    """ship front-end（先読み 0・発話統計 0）。"""
    n = w.shape[-1]
    T = SF.n_frames(n)
    f0, _ = SF.causal_f0(w)
    mlin = SF.to_frames(W @ SF.mel(w), T)
    P = SF.nhv_spec(mlin, f0, n, gen, T=T)
    Tn = P.shape[-1]
    f = torch.cat([mlin[None, :, :Tn], P.real[None], P.imag[None],
                   torch.log(P.abs()[None] + 1e-5)], 0)
    o = net(f[None])[0]
    return SF.cistft(torch.complex(o[0], o[1]), n).cpu().numpy()


def rms_match(y, g):
    """gt に RMS を揃えるだけ。クリップ処理はしない（試行単位で後段が行う）。"""
    n = min(len(y), len(g))
    y, g = np.asarray(y[:n], np.float64), np.asarray(g[:n], np.float64)
    return y * math.sqrt((g ** 2).mean() / max((y ** 2).mean(), 1e-20))


def norm_trial(clips: dict, g) -> dict:
    """試行内の全系に**共通の**減衰係数を掛ける。

    ⚠ 系ごとにピーク正規化すると RMS 整合が壊れる。実測（旧実装の 30 クリップ）:
    10 試行中 5 試行で RMS 差 > 0.5 dB、最大 4.37 dB。しかも t3 は
    gt > pathA > pathB と**期待される品質順位とレベル順位が一致**しており、
    聴取者は音質を聴かずに順位を付けられた（memory/pesq-blind-band-level と同型）。
    """
    ys = {k: rms_match(v, g) for k, v in clips.items()}
    pk = max(float(np.abs(v).max()) for v in ys.values())
    a = 0.95 / pk if pk > 0.95 else 1.0
    return {k: (v * a).astype(np.float32) for k, v in ys.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = TG.mel_to_linear(dev)
    from rddsp_gpu import build as build_small
    _, te = build_small(80, 12)
    items = te[:N_TRIAL]

    netA, ckA = load("gvoc_nhv", dev)
    netB, ckB = load("gvoc_ship", dev)
    print(f"  pathA gvoc_nhv  step {ckA['step']}  TEST {ckA['test']:.4f}")
    print(f"  pathB gvoc_ship step {ckB['step']}  TEST {ckB['test']:.4f}", flush=True)

    rng = np.random.RandomState(SEED)
    # ⚠ 素の permutation だと隣り合う試行の割り当てが一致しうる（実測: 使用済み seed で
    #   9 回中 2 回が連続同一）。同一だと「さっきと同じ」という情報が漏れる。
    #   6 通りしかないので**全試行で相異なる**ことは要求できない。隣接だけを排除する。
    assigns = []
    for _ in range(len(items)):
        while True:
            a = dict(zip(LETTER, list(rng.permutation(SYSTEMS))))
            if not assigns or a != assigns[-1]:
                assigns.append(a)
                break
    sc = {k: [] for k in SYSTEMS}

    for i, x in enumerate(items):
        assign = assigns[i]
        w = x["gt"].to(dev).float()
        gt = w.cpu().numpy()
        itA = TG.to_gpu_pre(dict(w=w, mel=x["mel"].to(dev), f0=x["f0"]), dev, W)
        clips = {
            "gt": gt,
            "pathA": render_pathA(netA, itA, dev,
                                  torch.Generator(device=dev).manual_seed(999)),
            "pathB": render_pathB(netB, w, dev,
                                  torch.Generator(device=dev).manual_seed(999), W),
        }
        nz = norm_trial(clips, gt)
        for L in LETTER:
            k = assign[L]
            sf.write(OUT / f"t{i}_{L}.wav", nz[k], R.SR)
            sc[k].append(safe_score(torch.tensor(gt),
                                    torch.tensor(clips[k][:len(gt)])))

    key = [f"seed={SEED}  trials={N_TRIAL}", "",
           "手順 1 の耳ゲート（Z0-C 条件1 ではない）。**試行ごとに割り当てを引き直す（隣接は必ず異なる）**。",
           "記録は試行ごとに閉じる: ①各文字の可/不可（必須・分岐に使う）と ②順位。",
           "文字をまたいだ総評はしない。",
           "gt は内部対照: gt を「可」とし、かつ gt より上位の系が無い試行が有効（同点1位も有効）。", ""]
    for i, a in enumerate(assigns):
        key.append(f"t{i}:  " + "  ".join(f"{L}={a[L]}" for L in LETTER))
    key += ["", f"この{len(items)}クリップの PESQ:"]
    key += [f"  {k:6s} {np.mean(v):.4f}" for k, v in sc.items()]
    (OUT / "ANSWER.txt").write_text("\n".join(key) + "\n")
    print("\n".join(f"  {k:6s} {np.mean(v):.4f}" for k, v in sc.items()))
    print(f"\n{N_TRIAL} trials x {len(SYSTEMS)} systems -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
