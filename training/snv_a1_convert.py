"""A1(synth_nam_vc §5・学習なし・外部の実装): SPARC 多言語版で本物の日本語の男声 → Irodori v4.1 の女声を変換し、LEAK 用の SPARC の特徴を書き出す。
事前登録 = results/snv_a1/prereg.yaml。SPARC の依存は学習の環境と分けて ../sparc_env(uv)で動かす。
    cd .. && ../sparc_env/bin/python training/snv_a1_convert.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_a1"
A0 = ROOT / "results/snv_a0"
SR = 16000


def segments(p: Path) -> tuple[np.ndarray, list[np.ndarray], int]:
    y, sr = librosa.load(str(p), sr=None, mono=True)
    y = y.astype(np.float64)
    w = 10 * sr
    tests = []
    s = 6 * w
    while len(tests) < 6 and s + w <= len(y):
        seg = y[s: s + w]
        if np.sqrt(np.mean(seg ** 2)) > 0.01:
            tests.append(seg)
        s += w
    return y[: 3 * w], tests, sr


def to16(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else y.astype(np.float32)


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    for d in ("src", "wav", "feat"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    srcs = {}
    for name, p in (("uirou", ROOT / "外郎売.mp3"), ("namikawa", ROOT / "namikawa.mp3")):
        enr, tests, sr = segments(p)
        sf.write(OUT / f"src/{name}_enroll.wav", to16(enr, sr), SR)
        for i, seg in enumerate(tests):
            q = OUT / f"src/{name}_{i}.wav"
            sf.write(q, to16(seg, sr), SR)
            srcs[f"{name}_{i}"] = q
    its = json.loads((A0 / "wav/items.json").read_text())
    fv = sorted({it["voice"] for it in its if it["sex"] == "F"})
    tcode = {}
    for v in fv:
        y = np.concatenate([to16(*librosa.load(str(A0 / f"wav/{v}_{j}.wav"), sr=None, mono=True)) for j in (0, 1, 2)])
        q = OUT / f"src/tref_{v}.wav"
        sf.write(q, y, SR)
        tcode[v] = m.encode(str(q))
    print("targets encoded", len(tcode), flush=True)
    meta = []
    for s, q in srcs.items():
        sc = m.encode(str(q))
        sf.write(OUT / f"wav/{s}__resynth.wav", m.decode(**copy.deepcopy(sc)), SR)
        for v, tc in tcode.items():
            sf.write(OUT / f"wav/{s}__{v}__std.wav", m.convert(src_code=sc, trg_code=tc), SR)
            c = copy.deepcopy(sc)
            c["pitch"] = c["pitch"] * 2.0
            c["spk_emb"] = tc["spk_emb"]
            sf.write(OUT / f"wav/{s}__{v}__k12.wav", m.decode(**c), SR)
            meta.append({"src": s, "tgt": v})
        print("converted", s, flush=True)
    (OUT / "convert_meta.json").write_text(json.dumps({"pairs": meta, "sources": list(srcs), "targets": fv}, indent=1))
    for it in its:
        if it["role"] not in ("enroll", "test"):
            continue
        c = m.encode(str(A0 / f"wav/{it['voice']}_{it['j']}.wav"))
        n = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
        np.savez(OUT / f"feat/{it['voice']}_{it['j']}.npz", ema=c["ema"][:n], pitch=c["pitch"][:n], loudness=c["loudness"][:n], periodicity=c["periodicity"][:n])
    print("feats done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
