"""paper_snv rev26 §5 M20・§9 O10 の全帯域の本物の音楽(段 I14・data/snv_sep/fma_sources.json): FMA(Defferrard ほか 2017・fma_small = 8,000 曲の 30 s の切り出し・MP3)のうち、許諾が CC BY / CC0 / パブリックドメインで、歌と人の声の無いものだけ。
許諾 = fma_metadata の tracks.csv の track.license(Attribution で始まり NC・ND・SA を含まないもの = by・CC0 / Public Domain = pd・他は除く)。
人の声 = musan_sources.py と同じ AudioSet の分類器(MIT/ast-finetuned-audioset-10-10-0.4593)の人の声の 63 類の確率の 10 秒窓(5 秒おき)の最大が thr 未満。
残したものを 48 kHz の mono の wav(data/fma/wav48/<id>.wav・PCM 16)にし、帯域(エネルギーの 99.9 % までの周波数)を記録する。帰属 = 曲名・作者・アルバム・許諾を files に残す。
    cd training && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python fma_sources.py [thr=0.02] [zip=data/fma_dl/fma_small.zip] [meta=data/fma_meta/fma_metadata/tracks.csv] [dev=cuda]
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

import musan_sources as MS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_sep"
WAV = ROOT / "data/fma/wav48"
SR_OUT = 48000
BAD = re.compile(r"noncommercial|no ?deriv|share ?alike|music sharing|sampling plus|art libre", re.I)


def lic_class(text: str) -> str:
    if not isinstance(text, str) or BAD.search(text):
        return "restricted"
    if re.match(r"\s*(creative commons )?attribution\b", text, re.I):
        return "by"
    if re.search(r"public domain|\bcc0\b", text, re.I):
        return "pd"
    return "unknown"


def bandwidth(x: np.ndarray, sr: int) -> float:
    P = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    c = np.cumsum(P) / max(P.sum(), 1e-20)
    return float(np.searchsorted(c, 0.999) * sr / 2 / (len(P) - 1))


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    thr = float(opt.get("thr", 0.02))
    import pandas as pd
    import soundfile as sf
    from scipy.signal import resample_poly
    t = pd.read_csv(ROOT / opt.get("meta", "data/fma_meta/fma_metadata/tracks.csv"), index_col=0, header=[0, 1], low_memory=False)
    t = t[t[("set", "subset")] == "small"]
    zf = zipfile.ZipFile(ROOT / opt.get("zip", "data/fma_dl/fma_small.zip"))
    names = {int(Path(n).stem): n for n in zf.namelist() if n.endswith(".mp3")}
    rows, excluded = [], {}
    WAV.mkdir(parents=True, exist_ok=True)
    for tid, r in t.iterrows():
        lic = r[("track", "license")]
        c = lic_class(lic)
        if c not in ("by", "pd") or tid not in names:
            why = f"license_{c}" if c not in ("by", "pd") else "missing"
            excluded[why] = excluded.get(why, 0) + 1
            continue
        dst = WAV / f"{tid:06d}.wav"
        if not dst.exists():
            tmp = WAV / f"{tid:06d}.tmp.mp3"
            tmp.write_bytes(zf.read(names[tid]))
            try:
                x, sr = sf.read(str(tmp), dtype="float32")
            except Exception as e:
                tmp.unlink()
                excluded["decode"] = excluded.get("decode", 0) + 1
                print("decode fail", tid, e, flush=True)
                continue
            tmp.unlink()
            x = x.mean(1) if x.ndim > 1 else x
            if sr != SR_OUT:
                g = np.gcd(SR_OUT, sr)
                x = resample_poly(x, SR_OUT // g, sr // g).astype(np.float32)
            sf.write(str(dst), np.clip(x, -1, 1), SR_OUT, subtype="PCM_16")
        info = sf.info(str(dst))
        rows.append({"kind": "music", "id": f"{tid:06d}", "path": str(dst), "rel": str(dst.relative_to(ROOT)), "dur": info.frames / info.samplerate,
                     "lic": c, "lic_text": str(lic), "title": str(r[("track", "title")]), "artist": str(r[("artist", "name")]), "artist_id": int(r[("artist", "id")]),
                     "album": str(r[("album", "title")]), "genre": str(r[("track", "genre_top")]), "size": dst.stat().st_size, "mtime": dst.stat().st_mtime_ns})
    print("permissive decoded", len(rows), "excluded", excluded, flush=True)
    MS.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    cache_p = OUT / "voice_scores.json"
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = [r for r in rows if cache.get(r["rel"], {}).get("key") != [r["size"], r["mtime"]]]
    if todo:
        fe = ASTFeatureExtractor.from_pretrained(MS.MODEL)
        m = ASTForAudioClassification.from_pretrained(MS.MODEL).to(opt.get("dev", "cuda")).eval()
        lab = m.config.id2label
        idx = [i for i in sorted(lab) if lab[i] in MS.VOICE]
        for n, r in enumerate(todo):
            x = sf.read(r["path"], dtype="float32")[0]
            x = resample_poly(x, 1, 3).astype(np.float32)
            st = list(range(0, max(1, len(x) - MS.WIN + 1), MS.HOP))
            if len(x) > MS.WIN and st[-1] + MS.WIN < len(x):
                st.append(len(x) - MS.WIN)
            feats = fe([x[s:s + MS.WIN] for s in st], sampling_rate=MS.SR, return_tensors="pt")["input_values"].to(opt.get("dev", "cuda"))
            with torch.no_grad():
                pr = torch.sigmoid(m(feats).logits).float().cpu().numpy()
            v = pr[:, idx]
            top = v.max(0)
            cache[r["rel"]] = {"key": [r["size"], r["mtime"]], "max": float(v.max()), "cls": lab[idx[int(top.argmax())]],
                               "top3": [[lab[idx[j]], round(float(top[j]), 4)] for j in np.argsort(-top)[:3]], "windows": len(st)}
            if (n + 1) % 100 == 0:
                print("scored", n + 1, "/", len(todo), flush=True)
                cache_p.write_text(json.dumps(cache))
        cache_p.write_text(json.dumps(cache))
    for r in rows:
        r["voice"], r["voice_cls"] = cache[r["rel"]]["max"], cache[r["rel"]]["cls"]
    keep = [r for r in rows if r["voice"] < thr]
    excluded["voice_screen"] = len(rows) - len(keep)
    for r in keep:
        x = sf.read(r["path"], dtype="float32")[0]
        r["bw_hz"] = round(bandwidth(x, SR_OUT))
    q = lambda xs: [round(float(np.quantile(xs, a)), 3) for a in (0.05, 0.25, 0.5, 0.75, 0.95)] if xs else []
    print("voice score quantiles", q([r["voice"] for r in rows]), "kept", len(keep), "genres", {g: sum(r["genre"] == g for r in keep) for g in sorted({r["genre"] for r in keep})}, flush=True)
    out = {"pools": {"music": [r["path"] for r in keep]},
           "licenses": {c: sum(r["lic"] == c for r in keep) for c in ("pd", "by")},
           "hours": round(sum(r["dur"] for r in keep) / 3600, 2),
           "bandwidth_hz_quantiles": q([r["bw_hz"] for r in keep]),
           "excluded": excluded,
           "screen": {"model": MS.MODEL, "thr": thr, "window_s": MS.WIN / MS.SR, "hop_s": MS.HOP / MS.SR, "classes": list(MS.VOICE)},
           "files": {r["id"]: {"lic": r["lic"], "text": r["lic_text"], "title": r["title"], "artist": r["artist"], "artist_id": r["artist_id"], "album": r["album"],
                               "genre": r["genre"], "voice": round(r["voice"], 4), "bw_hz": r["bw_hz"]} for r in keep},
           "source": "FMA: A Dataset for Music Analysis (Defferrard, Benzi, Vandergheynst, Bresson 2017, arXiv:1612.01840)・fma_small・音声は各曲の作者の許諾"}
    tmp = OUT / "fma_sources.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    tmp.replace(OUT / "fma_sources.json")
    print("fma sources", len(keep), out["licenses"], out["hours"], "h", "bw", out["bandwidth_hz_quantiles"], "excluded", excluded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
