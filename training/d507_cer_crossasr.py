"""D507(物差しの偏りの検査): 写し合成 CP は入力と同じ声 = 同じ音声認識の誤りが基準(SRC の文字起こし)と揃い、CER が有利になる疑い。
基準と仮説を別の音声認識(Whisper large-v3-turbo / parakeet-tdt_ctc-0.6b-ja)で取り、両方をカナ読み(fugashi + unidic-lite)にそろえて CER を出す。
WW / PP = 同じ認識器(今の物差しの型)・PW / WP = 基準と仮説が別の認識器(誤りの相関が切れる)。CP と変換の差が認識器をまたぐと縮むかを見る。評価専用の環境(asr_eval_env)で走らせる。
    /home/kojirotanaka/asr_eval_env/.venv/bin/python d507_cer_crossasr.py --d ../results/conv_p0/d507 --devset <devff23x>/devset.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", required=True)
    ap.add_argument("--devset", required=True)
    a = ap.parse_args()
    import fugashi
    tg = fugashi.Tagger()

    def kana(s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        s = "".join(ch for ch in s if not (unicodedata.category(ch)[0] in "PSZ"))
        out = "".join((getattr(w.feature, "kana", None) or w.surface) for w in tg(s))
        out = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in out)
        return out.replace("ー", "")

    def edit(x: str, y: str) -> int:
        d = list(range(len(y) + 1))
        for i, cx in enumerate(x, 1):
            prev, d[0] = d[0], i
            for j, cy in enumerate(y, 1):
                cur = min(d[j] + 1, d[j - 1] + 1, prev + (cx != cy))
                prev, d[j] = d[j], cur
        return d[len(y)]

    D = Path(a.d)
    w = json.loads((D / "tx_whisper.json").read_text())
    p_raw = json.loads((D / "tx_parakeet.json").read_text())
    p: dict = {}
    for f, t in p_raw.items():
        m = re.match(r"(p\d+_t\d+)_(.+)\.wav", Path(f).name)
        p.setdefault(m.group(1), {})[m.group(2)] = t
    conds = ["CP", "OP15", "P50", "P49FF"]
    K = {s: {k: {c: kana(v[c]) for c in ["SRC"] + conds if c in v} for k, v in src.items()} for s, src in (("W", w), ("P", p))}
    spk = {f"p{r['pi']:04d}_t{r['ti']:03d}": r["spk"] for r in json.loads(Path(a.devset).read_text())["pairs"]}
    keys = sorted(k for k in K["W"] if k in K["P"] and len(K["W"][k]["SRC"]) >= 4 and len(K["P"][k]["SRC"]) >= 4)
    speechy = lambda s: len(s) >= 10 and Counter(s).most_common(1)[0][1] / len(s) < 0.3
    rep: dict = {}
    rng = np.random.default_rng(507)
    for sub, sel in (("all", lambda k: True), ("speechlike", lambda k: speechy(K["W"][k]["SRC"]))):
        ks = [k for k in keys if sel(k)]
        names = sorted({spk[k] for k in ks})
        cl = np.array([names.index(spk[k]) for k in ks])
        us = np.unique(cl)
        idx = {u: np.flatnonzero(cl == u) for u in us}
        rep[sub] = {"n": len(ks)}
        for sch, (rs, hs) in {"WW": ("W", "W"), "PP": ("P", "P"), "PW": ("P", "W"), "WP": ("W", "P")}.items():
            X = {c: np.array([min(edit(K[rs][k]["SRC"], K[hs][k][c]) / len(K[rs][k]["SRC"]), 1.0) for k in ks]) for c in conds}
            r = {"clipped_mean": {c: round(float(X[c].mean()), 3) for c in conds}, "median": {c: round(float(np.median(X[c])), 3) for c in conds}, "diff_vs_CP": {}}
            for c in conds[1:]:
                d = X[c] - X["CP"]
                bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
                r["diff_vs_CP"][c] = [round(float(d.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
            rep[sub][sch] = r
            print(sub, sch, json.dumps(r, ensure_ascii=False), flush=True)
    (D / "crossasr.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
