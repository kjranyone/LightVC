from __future__ import annotations

import sys
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

ROOT = Path("..")
FEMALE_TTS = ROOT / "data/female_tts_corpus"
FEMALE_REAL = ROOT / "female-dataset"
MALE_TTS = ROOT / "data/male_tts_corpus"
VCTK = ROOT / "data/vctk_200"
OUT_TSV = ROOT / "data/kansei_vc/golden_mini.tsv"
NEG_DIR = ROOT / "results/golden_negatives"

SIBILANT_KANA = set("さしすせそサシスセソざじずぜぞツつっッシュショシャすず")

COVERAGE_GAPS = [
    "source_male:small_voice", "source_male:fast", "source_male:laugh",
    "source_male:breathy", "source_male:sibilant",
    "target_female:whisper(explicit)",
]

FIELDS = ["golden_id", "category", "role", "style", "path", "sr", "dur_s", "text", "note"]


def _dur_sr(path) -> tuple:
    info = sf.info(str(path))
    return round(info.frames / info.samplerate, 3), info.samplerate


def pick_female_tts(rng, per_style: int = 2) -> list:
    styles = ["breathy", "soft", "neutral", "tension", "warm"]
    rows = []
    all_wavs = list(FEMALE_TTS.rglob("*.wav"))
    rng.shuffle(all_wavs)
    for style in styles:
        got = 0
        for w in all_wavs:
            if got >= per_style:
                break
            if w.stem.endswith(style):
                dur, sr = _dur_sr(w)
                rows.append(dict(golden_id=f"ftts_{style}_{got}", category=f"target_female:{style}",
                                 role="target", style=style, path=str(w), sr=sr, dur_s=dur,
                                 text="", note="tts"))
                got += 1
    return rows


def pick_female_real(rng, pool: int = 400) -> list:
    speakers = sorted([d for d in FEMALE_REAL.iterdir() if d.is_dir()])
    rng.shuffle(speakers)
    cands = []
    for sd in speakers:
        wavs = sorted(sd.glob("*.wav"))
        if not wavs:
            continue
        w = rng.choice(wavs)
        try:
            y, sr = sf.read(str(w), dtype="float32")
            if y.ndim > 1:
                y = y[:, 0]
            dur = len(y) / sr
            if dur < 1.0 or dur > 12.0:
                continue
            rms = float(np.sqrt(np.mean(y ** 2) + 1e-9))
            lab = w.with_suffix(".lab")
            text = lab.read_text().strip() if lab.exists() else ""
            sib = sum(ch in SIBILANT_KANA for ch in text)
            sib_rate = sib / max(len(text), 1)
            cands.append((w, sr, dur, rms, sib, sib_rate, text))
        except Exception:
            continue
        if len(cands) >= pool:
            break

    rows = []
    if not cands:
        return rows

    def add(sel, cat: str, note: str, n: int) -> None:
        for i, c in enumerate(sel[:n]):
            w, sr, dur, rms, sib, sib_rate, text = c
            rows.append(dict(golden_id=f"freal_{cat.split(':')[1]}_{i}", category=cat,
                             role="target", style=cat.split(":")[1], path=str(w), sr=sr,
                             dur_s=round(dur, 3), text=text[:60], note=note))

    by_rms = sorted(cands, key=lambda c: c[3])
    by_sib = sorted(cands, key=lambda c: -c[5])
    by_dur = sorted(cands, key=lambda c: -c[2])
    used = set()

    def take(sorted_list, n: int) -> list:
        out = []
        for c in sorted_list:
            if c[0] in used:
                continue
            out.append(c); used.add(c[0])
            if len(out) >= n:
                break
        return out

    add(take(by_rms, 2), "target_female:small_voice", "proxy=low_rms", 2)
    add(take(by_sib, 2), "target_female:sibilant", "proxy=sibilant_kana", 2)
    add(take(by_dur, 2), "target_female:long_tail", "proxy=long_dur", 2)
    rng.shuffle(cands)
    add(take(cands, 2), "target_female:emotional", "random_real", 2)
    return rows


def pick_male(rng, per_style: int = 2) -> list:
    styles = ["calm_low", "deep", "neutral", "young", "warm"]
    rows = []
    all_wavs = list(MALE_TTS.rglob("*.wav"))
    rng.shuffle(all_wavs)
    for style in styles:
        got = 0
        for w in all_wavs:
            if got >= per_style:
                break
            if w.stem.endswith(style) or f"_{style}" in w.stem:
                dur, sr = _dur_sr(w)
                rows.append(dict(golden_id=f"male_{style}_{got}", category=f"source_male:{style}",
                                 role="source", style=style, path=str(w), sr=sr, dur_s=dur,
                                 text="", note="tts_source"))
                got += 1
    return rows


def pick_vctk_pairs(rng, n_ids: int = 2) -> list:
    wavs = list(VCTK.rglob("*.wav"))
    by_text = {}
    for w in wavs:
        parts = w.stem.split("_")
        if len(parts) == 2:
            spk, tid = parts
            by_text.setdefault(tid, []).append((spk, w))
    shared = {tid: v for tid, v in by_text.items() if len({s for s, _ in v}) >= 2}
    rows = []
    for tid in sorted(shared)[:n_ids]:
        spk_map = {}
        for spk, w in shared[tid]:
            spk_map.setdefault(spk, w)
        for j, (spk, w) in enumerate(sorted(spk_map.items())[:2]):
            dur, sr = _dur_sr(w)
            rows.append(dict(golden_id=f"pair_{tid}_{spk}", category=f"pair_same_text:{tid}",
                             role="pair", style=spk, path=str(w), sr=sr, dur_s=dur,
                             text="", note="vctk_same_text"))
    return rows


def make_negatives(seed_row: dict) -> list:
    NEG_DIR.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(seed_row["path"], dtype="float32")
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 44100:
        y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=44100).astype(np.float32)
        sr = 44100
    rng = np.random.default_rng(0)
    rows = []

    y8 = librosa.resample(y, orig_sr=44100, target_sr=8000)
    muff = librosa.resample(y8, orig_sr=8000, target_sr=44100).astype(np.float32)
    step = 2.0 / (2 ** 4)
    crush = (np.round(y / step) * step).astype(np.float32)
    crush = crush[:: 3].repeat(3)[: len(y)]
    noise = rng.standard_normal(len(y)).astype(np.float32) * 0.02
    jitter = 1.0 + 0.15 * np.sin(np.linspace(0, 80 * np.pi, len(y))).astype(np.float32)
    rough = (y * jitter + noise).astype(np.float32)

    for name, sig, note in [("muffled_8k", muff, "muffled/8k"),
                            ("metallic_crush", crush, "metallic/bitcrush"),
                            ("rough_noise", rough, "rough/tiring")]:
        p = NEG_DIR / f"neg_{name}.wav"
        sf.write(p, np.clip(sig, -1, 1), sr)
        dur, _ = _dur_sr(p)
        rows.append(dict(golden_id=f"neg_{name}", category=f"negative:{name}", role="negative",
                         style=name, path=str(p), sr=sr, dur_s=dur, text="", note=note))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print("=== Golden Mini Set ===")
    rows = []
    rows += pick_female_tts(rng)
    print(f"  female_tts: {len([r for r in rows if r['role']=='target'])}")
    rows += pick_female_real(rng)
    rows += pick_male(rng)
    rows += pick_vctk_pairs(rng)
    seed = next((r for r in rows if r["category"].startswith("target_female:soft")), rows[0])
    rows += make_negatives(seed)

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cats = {}
    for r in rows:
        cats.setdefault(r["role"], []).append(r["category"])
    report = {
        "seed": args.seed, "total": len(rows),
        "by_role": {k: len(v) for k, v in cats.items()},
        "categories": sorted({r["category"] for r in rows}),
        "coverage_gaps_todo": COVERAGE_GAPS,
    }
    (OUT_TSV.parent / "golden_mini_coverage.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    print(f"  total rows: {len(rows)} -> {OUT_TSV}")
    for role, v in cats.items():
        print(f"    {role}: {len(v)}")
    print(f"  COVERAGE GAPS (need human curation): {COVERAGE_GAPS}")


if __name__ == "__main__":
    main()
