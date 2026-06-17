"""
Build an evaluation manifest from a VCTK corpus for parallel validation.

VCTK has same-text utterances across speakers. This script scans the corpus,
finds same-utterance-id pairs across speakers, reads the ground-truth text,
and emits a JSON manifest compatible with evaluate.py ([04-6]).

Usage:
    uv run python build_vctk_manifest.py \
        --vctk-root /path/to/VCTK-Corpus \
        --output eval_manifest.json \
        --max-pairs 100

Expected VCTK layout (standard release):
    VCTK-Corpus/
    ├── wav48/{speaker}/{speaker}_{utt:03d}.wav
    └── txt/{speaker}/{speaker}_{utt:03d}.txt

For each utterance id, the first available speaker is the source and the
second is the reference (any utterance from that speaker). The text file
of the source utterance provides the ground-truth transcription.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def scan_vctk(vctk_root: Path):
    """Return {utt_id: [(speaker, wav_path, txt_path), ...]}."""
    wav_dir = vctk_root / "wav48"
    txt_dir = vctk_root / "txt"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"wav48/ not found under {vctk_root}")

    by_utt: dict[int, list[tuple[str, Path, Path]]] = defaultdict(list)

    for speaker_dir in sorted(wav_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for wav in sorted(speaker_dir.glob("*.wav")):
            stem = wav.stem
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                utt_id = int(parts[1])
            except ValueError:
                continue
            txt = txt_dir / speaker / f"{stem}.txt"
            by_utt[utt_id].append((speaker, wav, txt))

    return by_utt


def read_text(txt_path: Path) -> str | None:
    if not txt_path.is_file():
        return None
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def build_manifest(vctk_root: Path, max_pairs: int, seed: int = 42):
    by_utt = scan_vctk(vctk_root)
    wav_dir = vctk_root / "wav48"
    pairs = []

    # Only keep utterances spoken by >= 2 speakers (parallel).
    parallel_utts = sorted(u for u, spks in by_utt.items() if len(spks) >= 2)
    rng = random.Random(seed)
    rng.shuffle(parallel_utts)

    for utt_id in parallel_utts:
        speakers = by_utt[utt_id]
        src_spk, src_wav, src_txt = speakers[0]
        ref_spk = speakers[1][0]

        # Reference: any utterance from ref_spk (not necessarily this utt_id).
        ref_candidates = sorted((wav_dir / ref_spk).glob("*.wav"))
        if not ref_candidates:
            continue
        ref_wav = rng.choice(ref_candidates)

        text = read_text(src_txt)
        if text is None:
            continue

        pairs.append(
            {
                "source": str(src_wav),
                "reference": str(ref_wav),
                "text": text,
                "utt_id": utt_id,
                "source_speaker": src_spk,
                "reference_speaker": ref_spk,
            }
        )
        if len(pairs) >= max_pairs:
            break

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Build VCTK parallel-pair manifest for evaluate.py"
    )
    parser.add_argument("--vctk-root", required=True, help="Path to VCTK-Corpus root")
    parser.add_argument("--output", required=True, help="Output JSON manifest path")
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = build_manifest(Path(args.vctk_root), args.max_pairs, args.seed)
    manifest = {"pairs": pairs}
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    speakers = sorted({p["source_speaker"] for p in pairs})
    print(
        f"Wrote {len(pairs)} parallel pairs from {len(speakers)} source "
        f"speakers to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
