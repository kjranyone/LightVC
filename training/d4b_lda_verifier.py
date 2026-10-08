"""D4b: G と無関係に学習した独立の話者識別器(凍結 WavLM-large の層の平均+標準偏差 → LDA)で日本語 21 目標の変換音声を測り直す(推論のみ・GPU は埋め込みだけ)。
学習話者 = female-dataset の評価話者(held)以外から --n_spk 人・各 --per 発話(8s)。G・変換器の学習には使っていない識別器 = fooling の経路がない。
較正: 目標の実音声 cen の前半を中心・後半を問いにした本人 top-1。
    uv run python d4b_lda_verifier.py --wav_dir <save_dir> --out ../results/conv_p0/d4b.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.stats import wilcoxon

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent


def load16(path, maxsec=None):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if maxsec:
        x = x[: int(maxsec * sr)]
    return resample_poly(x, 16000, sr).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", nargs="+", required=True, help="複数可(LDA は 1 回だけ学習して各ディレクトリを評価)")
    ap.add_argument("--out", nargs="+", required=True)
    ap.add_argument("--n_spk", type=int, default=1200)
    ap.add_argument("--per", type=int, default=4)
    ap.add_argument("--layers", type=int, nargs="*", default=[4, 6, 8])
    ap.add_argument("--exclude_spk", nargs="*", default=["fe9565ca1f33bf20", "ffb9b5647612b32b"])
    a = ap.parse_args()
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from transformers import WavLMModel
    import train_rvoc as TR
    held = TR.held_speakers() | set(a.exclude_spk)
    allspk = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in held)
    rng = np.random.RandomState(0)
    spks = [allspk[i] for i in rng.permutation(len(allspk))[: a.n_spk]]
    dev = "cuda"
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()

    @torch.no_grad()
    def emb(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
        v = []
        for l in a.layers:
            h = hs[l][0]
            v += [h.mean(0), h.std(0)]
        return torch.cat(v).cpu().numpy()

    X, y = [], []
    for si, s in enumerate(spks):
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        if len(ws) < a.per:
            continue
        for w in [ws[i] for i in np.linspace(0, len(ws) - 1, a.per).astype(int)]:
            x = load16(w, 8.0)
            if len(x) < 16000 * 2:
                continue
            X.append(emb(x)); y.append(si)
        if si % 200 == 0:
            print("train emb", si, flush=True)
    X, y = np.stack(X), np.array(y)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    lda = LDA(solver="eigen", shrinkage=0.3, n_components=128).fit((X - mu) / sd, y)
    tr = lambda E: lda.transform((E - mu) / sd)
    print("LDA fit", X.shape, flush=True)

    for wd_, out_ in zip(a.wav_dir, a.out):
        d = Path(wd_)
        cens = sorted(d.glob("cen_*.wav"), key=lambda f: int(re.match(r"cen_(\d+)", f.name).group(1)))
        S = len(cens)
        assert [int(re.match(r"cen_(\d+)", f.name).group(1)) for f in cens] == list(range(S)), "cen の番号が 0..S-1 の連番でない"
        cen_x = [load16(f) for f in cens]
        Ea = tr(np.stack([emb(x[: len(x) // 2]) for x in cen_x]))
        Eb = tr(np.stack([emb(x[len(x) // 2:]) for x in cen_x]))
        Ec = tr(np.stack([emb(x) for x in cen_x]))
        nz = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
        mc = Ea.mean(0, keepdims=True)
        cal = float(((nz(Eb - mc) @ nz(Ea - mc).T).argmax(1) == np.arange(S)).mean())
        cal_chance = 1 / S
        print("calib real top1", cal, flush=True)
        mc = Ec.mean(0, keepdims=True)
        C = nz(Ec - mc)
        meta = []
        for f in sorted(d.glob("p*_t*_*.wav")):
            m = re.match(r"p(\d+)_t(\d+)_(.+)\.wav", f.name)
            meta.append((int(m.group(1)), int(m.group(2)), m.group(3), f))
        names = sorted({m[2] for m in meta})
        R = {n: ([], []) for n in names}
        for pi, ti, n, f in meta:
            v = nz(tr(emb(load16(f))[None]) - mc)[0]
            se = C @ v
            R[n][0].append(int((se > se[ti]).sum()) + 1)
            R[n][1].append(ti)
        rep = {"n_train_spk": int(len(set(y))), "n_train_clips": int(len(y)), "calib_real_top1": round(cal, 3), "chance": round(cal_chance, 3), "n_targets": S}
        rb = np.array(R["TAB"][0])
        tg = np.array(R["TAB"][1])
        for n in names:
            rk = np.array(R[n][0])
            blk = {"top1": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "mean_rank": round(float(rk.mean()), 2)}
            if n not in ("SRC", "TAB"):
                ma = np.array([rk[tg == t].mean() for t in range(S)]); mb = np.array([rb[tg == t].mean() for t in range(S)])
                blk["cluster_p_improve_vs_TAB"] = round(float(wilcoxon(ma, mb, alternative="less").pvalue), 4)
                blk["targets_better/worse"] = [int((ma < mb).sum()), int((ma > mb).sum())]
            rep[n] = blk
        print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
        Path(out_).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
