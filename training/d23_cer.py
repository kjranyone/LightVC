"""D23: 変換の内容の明瞭さ = 文字誤り率(CER)。元の声(SRC)の文字起こしを基準に、各条件の出力の文字起こしとの編集距離 / 基準の文字数。
音声認識 = faster-whisper large-v3-turbo(日本語・ビーム 5・温度 0)。句読点・空白は除いて比べる。conv_ja --save_dir の pNNN_tNN_<cond>.wav を読む。
    uv run python d23_cer.py --dirs <dir1> <dir2> ... --names w1 w3 w5 --out ../results/conv_p0/d23_cer.json
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if not (unicodedata.category(ch).startswith("P") or ch.isspace() or unicodedata.category(ch).startswith("S")))


def edit(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev, d[j] = d[j], cur
    return d[len(b)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--re", default=r"(p\d+_t\d+)_(.+)\.wav", help="ファイル名 → (組, 条件)。D80 の形式は (m\d+_[0-9a-f]+_t\d+)_(.+)\.wav")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    m = WhisperModel("mobiuslabsgmbh/faster-whisper-large-v3-turbo", device=a.device, compute_type="float16" if a.device == "cuda" else "int8", cpu_threads=8)

    def tr(p: Path) -> str:
        segs, _ = m.transcribe(str(p), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return norm("".join(s.text for s in segs))

    rep: dict = {}
    for d, nm in zip(a.dirs, a.names):
        d = Path(d)
        files = sorted(d.glob("*.wav"))
        by: dict = {}
        for f in files:
            mt = re.match(a.re, f.name)
            if not mt:
                continue
            by.setdefault(mt.group(1), {})[mt.group(2)] = f
        conds = sorted({c for v in by.values() for c in v if c != "SRC"})
        res = {c: [] for c in conds}
        per = {c: {} for c in conds}
        ex = []
        for key, v in sorted(by.items()):
            if "SRC" not in v:
                continue
            ref = tr(v["SRC"])
            if len(ref) < 4:
                continue
            for c in conds:
                if c in v:
                    hyp = tr(v[c])
                    res[c].append(edit(ref, hyp) / len(ref))
                    per[c][key] = round(res[c][-1], 4)
                    if len(ex) < 6 and c == conds[0]:
                        ex.append({"ref": ref, "hyp": hyp, "cond": c})
        rep[nm] = {c: {"cer_mean": round(float(np.mean(x)), 4), "cer_median": round(float(np.median(x)), 4), "n": len(x)} for c, x in res.items()}
        rep[nm]["examples"] = ex
        rep[nm]["per"] = per
        print(nm, {c: rep[nm][c] for c in conds}, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
