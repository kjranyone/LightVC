"""
Download LibriTTS or VCTK from HuggingFace datasets and lay them out in the
directory structure expected by encode_corpus.py:

    {output_root}/{speaker_id}/{utterance_id}.wav

Usage:
    uv run python download_corpus.py --dataset libritts --output ../data/libritts
    uv run python download_corpus.py --dataset vctk    --output ../data/vctk

Datasets are streamed via the `datasets` library so the full corpus does not
need to fit in RAM. Audio is written as 16-bit PCM WAV at the original sample
rate (encode_corpus.py handles resampling to 44.1 kHz).

Note: VCTK on HuggingFace is `englishfe/vctk` (or similar mirror); LibriTTS
is `openslr/libritts`. Both require `pip install datasets soundfile`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

DATASETS = {
    "libritts": {
        "repo": "openslr/libritts",
        "split": "train.clean",
        "audio_col": "audio",
        "speaker_col": "speaker_id",
        "utt_col": "id",
    },
    "vctk": {
        "repo": "englishfe/vctk",
        "split": "train",
        "audio_col": "audio",
        "speaker_col": "speaker_id",
        "utt_col": "file_name",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LibriTTS / VCTK corpus")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Which corpus to download",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output root. Will be created if it does not exist.",
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=0,
        help="Cap utterances per speaker (0 = no cap). Useful for quick tests.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=0,
        help="Stop after this many speakers (0 = all).",
    )
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {spec['repo']} ({spec['split']}) -> {out_root}")
    ds = load_dataset(
        spec["repo"],
        split=spec["split"],
        streaming=True,
    )

    counts: dict[str, int] = {}
    speakers_seen = 0
    written = 0

    for ex in tqdm(ds, desc=f"{args.dataset}"):
        spk = str(ex[spec["speaker_col"]])
        if args.max_per_speaker and counts.get(spk, 0) >= args.max_per_speaker:
            continue
        if (
            args.max_speakers
            and spk not in counts
            and speakers_seen >= args.max_speakers
        ):
            break

        if spk not in counts:
            speakers_seen += 1
            (out_root / spk).mkdir(parents=True, exist_ok=True)
        counts[spk] = counts.get(spk, 0) + 1

        audio = ex[spec["audio_col"]]
        wav = audio["array"]
        sr = audio["sampling_rate"]

        utt_id = str(ex.get(spec["utt_col"], f"{spk}_{written:08d}"))
        utt_id = Path(utt_id).stem
        out_path = out_root / spk / f"{utt_id}.wav"
        if out_path.exists():
            continue

        sf.write(str(out_path), wav, sr, subtype="PCM_16")
        written += 1

    print(f"\nDone. {written} utterances across {len(counts)} speakers.")
    print(f"Output: {out_root}")
    print("\nNext step:")
    print(
        f"  uv run python encode_corpus.py --source {out_root} "
        f"--output data/latents_{args.dataset}"
    )


if __name__ == "__main__":
    main()
