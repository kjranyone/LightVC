"""D7: 性別に依存しない単位の事前検査(学習なし・ContentVec の特徴・非因果 = 単位の在庫の検査)。
コードブック(K)を (a) 生の ContentVec 特徴 / (b) 話者ごとの平均を引いた特徴 で作り、日本語男声の単位が目標の女声の参照(20s)の観測済みセルに当たる割合を比べる。
コードブックの話者: 日本語女声(評価話者以外)と日本語 TTS 男声の半分。評価: 日本語 held 女声 20 人(参照 20s / 別発話)・TTS 男声の残り半分。
    uv run python d7_invariant_units.py --out ../results/conv_p0/d7_invariant_units.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n_fem", type=int, default=120)
    ap.add_argument("--ref_sec", type=float, default=20.0)
    a = ap.parse_args()
    import artic_g2_unit as U
    import idloss as ID
    import train_rvoc as TR
    dev = "cuda"
    cv = ID.ContentVec(dev)

    @torch.no_grad()
    def feats(x: np.ndarray) -> np.ndarray:
        f = cv(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy().astype(np.float32)
        return f

    def spk_audio(files, sec):
        xs, tot = [], 0.0
        for w in files:
            if tot >= sec:
                break
            x = load48(w); xs.append(x); tot += len(x) / 48000
        return np.concatenate(xs) if xs else None

    rng = random.Random(0)
    hs = TR.held_speakers()
    allf = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in hs and p.name not in TTS_LEAK)
    fem_tr = rng.sample(allf, a.n_fem)
    mrows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["src"] == "tts_m"]
    by: dict = {}
    for r in mrows:
        by.setdefault(r["spk"], []).append(r["wav"])
    msp = sorted(by)
    rng.shuffle(msp)
    m_tr, m_ev = msp[: len(msp) // 2], msp[len(msp) // 2:]
    pool_raw, pool_norm = [], []
    for s in fem_tr:
        x = spk_audio(sorted((ROOT / "female-dataset" / s).glob("*.wav")), 10.0)
        if x is None:
            continue
        f = feats(x); pool_raw.append(f); pool_norm.append(f - f.mean(0, keepdims=True))
    nf = len(pool_raw)
    for s in m_tr:
        fl = sorted(by[s]); rng.shuffle(fl)
        for rep in range(max(1, a.n_fem // len(m_tr) // 2)):
            x = spk_audio(fl[rep * 6:(rep + 1) * 6], 10.0)
            if x is None:
                continue
            f = feats(x); pool_raw.append(f); pool_norm.append(f - f.mean(0, keepdims=True))
    print("codebook pool: female utt-sets", nf, "male utt-sets", len(pool_raw) - nf, flush=True)
    books = {"raw": U.kmeans(np.concatenate(pool_raw), a.k, dev), "spknorm": U.kmeans(np.concatenate(pool_norm), a.k, dev)}

    def assign(f: np.ndarray, cb: np.ndarray) -> np.ndarray:
        ft = torch.from_numpy(f).to(dev); cbt = torch.from_numpy(np.asarray(cb, np.float32)).to(dev)
        return ((ft[:, None] - cbt[None]) ** 2).sum(-1).argmin(1).cpu().numpy()

    held = [s for s in sorted(hs) if s != "unknown" and s not in TTS_LEAK and (ROOT / "female-dataset" / s).is_dir()]
    tg = []
    for s in held:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= a.ref_sec:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        if not rest:
            continue
        fr = feats(spk_audio(ref, 30.0)); fc = feats(spk_audio(rest, 25.0))
        tg.append({"ref": fr, "self": fc})
    mev = []
    for s in m_ev:
        fl = sorted(by[s])
        mev.append(feats(spk_audio(fl[:8], 10.0)))
    print("targets", len(tg), "male eval speakers", len(mev), flush=True)
    rep: dict = {"k": a.k, "n_targets": len(tg), "male_eval": len(mev)}
    for kind, cb in books.items():
        nz = (lambda f: f - f.mean(0, keepdims=True)) if kind == "spknorm" else (lambda f: f)
        h1s, h5s, s1s, s5s = [], [], [], []
        for i, t in enumerate(tg):
            cnt = np.bincount(assign(nz(t["ref"]), cb), minlength=a.k)
            us = assign(nz(t["self"]), cb)
            s1s.append(float((cnt[us] >= 1).mean())); s5s.append(float((cnt[us] >= 5).mean()))
            for m in (mev[i % len(mev)], mev[(i + 3) % len(mev)]):
                um = assign(nz(m), cb)
                h1s.append(float((cnt[um] >= 1).mean())); h5s.append(float((cnt[um] >= 5).mean()))
        rep[kind] = {"female_self_hit>=1": round(np.mean(s1s), 3), "female_self_hit>=5": round(np.mean(s5s), 3),
                     "ja_male_hit>=1": round(np.mean(h1s), 3), "ja_male_hit>=5": round(np.mean(h5s), 3)}
        print(kind, rep[kind], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
