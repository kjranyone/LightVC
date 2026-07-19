import argparse
import ast
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf


ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"

BASE_FEMALE_CAPTIONS = {"neutral", "soft", "breathy", "warm", "low_tension"}
CHAR_FEMALE_CAPTIONS = {
    "young_bright",
    "intimate_close",
    "cool_calm",
    "cute_high",
    "mature_deep",
}

FIELDNAMES = [
    "utterance_id",
    "path",
    "path_type",
    "layer",
    "source_type",
    "speaker_id",
    "speaker_gender",
    "text",
    "text_id",
    "text_category",
    "caption_key",
    "caption_text",
    "style_tags",
    "role",
    "persona",
    "scene",
    "relation",
    "phoneme_focus",
    "duration_sec",
    "sample_rate",
    "channels",
    "n_frames",
    "split",
    "text_split",
    "license",
    "quality_status",
    "quality_flags",
    "wav_path",
    "latent_path",
    "feature_path",
    "origin",
    "generation_model",
    "ref_wav",
    "seed",
]


def read_literals(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def as_input_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def rel(path: Path | None) -> str:
    if path is None:
        return ""
    return os.path.relpath(path.resolve(), TRAINING_DIR.resolve())


def speaker_split(speaker_id: str) -> str:
    digest = hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def parse_stem(stem: str, caption_keys: set[str]) -> tuple[str, str]:
    for caption_key in sorted(caption_keys, key=len, reverse=True):
        suffix = f"_{caption_key}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], caption_key
    return "", ""


def feature_path_for(latent_path: Path | None) -> Path | None:
    if latent_path is None:
        return None
    return latent_path.with_name(f"{latent_path.stem}_feat.pt")


def path_index(root: Path, suffix: str, skip_feat: bool = False) -> dict[tuple[str, str], Path]:
    if not root.exists():
        return {}
    items: dict[tuple[str, str], Path] = {}
    for path in sorted(root.rglob(f"*{suffix}")):
        if skip_feat and path.stem.endswith("_feat"):
            continue
        speaker_id = path.parent.name
        items[(speaker_id, path.stem)] = path
    return items


class AudioInfo:
    def __init__(self) -> None:
        self.cache: dict[Path, tuple[float, int, int, int]] = {}

    def get(self, path: Path | None) -> tuple[str, str, str, str]:
        if path is None:
            return "", "", "", ""
        resolved = path.resolve()
        if resolved not in self.cache:
            info = sf.info(str(resolved))
            self.cache[resolved] = (
                info.frames / info.samplerate,
                info.samplerate,
                info.channels,
                info.frames,
            )
        duration, sample_rate, channels, frames = self.cache[resolved]
        return f"{duration:.6f}", str(sample_rate), str(channels), str(frames)


class Builder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.audio_info = AudioInfo()
        self.rows: list[dict[str, str]] = []
        self.issues: list[dict[str, str]] = []
        self.seen_ids: set[str] = set()
        self.female_texts, self.female_categories, self.female_captions = self.load_female_catalog()
        self.male_texts, self.male_categories, self.male_captions = self.load_male_catalog()
        self.live_texts = self.load_live_texts()
        self.female_ref_wavs = self.build_ref_wavs(as_input_path(args.female_real_wavs))

    def load_female_catalog(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        values = read_literals(ROOT / "training" / "generate_female_corpus_fast.py")
        groups = [
            ("TEXTS", "read"),
            ("TEXTS_EMOTIONAL", "emotional"),
            ("TEXTS_LIVE", "live"),
            ("TEXTS_WHISPER", "whisper"),
        ]
        texts: dict[str, str] = {}
        categories: dict[str, str] = {}
        index = 0
        for name, category in groups:
            for text in values.get(name, []):
                text_id = f"t{index:02d}"
                texts[text_id] = text
                categories[text_id] = category
                index += 1
        return texts, categories, values.get("CAPTIONS", {})

    def load_male_catalog(self) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        values = read_literals(ROOT / "training" / "generate_male_corpus.py")
        groups = [
            ("TEXTS_PHONEME_1", "phoneme_basic"),
            ("TEXTS_PHONEME_2", "phoneme_fricative_plosive"),
            ("TEXTS_PHONEME_3", "phoneme_nasal_liquid"),
            ("TEXTS_DAILY", "daily"),
            ("TEXTS_LIVE", "live"),
            ("TEXTS_EMOTIONAL", "emotional"),
            ("TEXTS_LONG", "long"),
        ]
        texts: dict[str, str] = {}
        categories: dict[str, str] = {}
        index = 0
        for name, category in groups:
            for text in values.get(name, []):
                text_id = f"t{index:02d}"
                texts[text_id] = text
                categories[text_id] = category
                index += 1
        return texts, categories, values.get("MALE_CAPTIONS", {})

    def load_live_texts(self) -> dict[str, dict[str, str]]:
        path = as_input_path(self.args.live_texts_tsv)
        if not path.exists():
            return {}
        with path.open(encoding="utf-8", newline="") as f:
            return {row["text_id"]: row for row in csv.DictReader(f, delimiter="\t")}

    def build_ref_wavs(self, root: Path) -> dict[str, Path]:
        refs: dict[str, tuple[float, Path]] = {}
        if not root.exists():
            return {}
        for wav_path in sorted(root.rglob("*.wav")):
            speaker_id = wav_path.parent.name
            try:
                duration, _, _, _ = self.audio_info.get(wav_path)
            except Exception:
                continue
            current = refs.get(speaker_id)
            value = (float(duration), wav_path)
            if current is None or value[0] > current[0]:
                refs[speaker_id] = value
        return {speaker_id: path for speaker_id, (_, path) in refs.items()}

    def issue(self, row: dict[str, str], severity: str, issue: str, detail: str) -> None:
        self.issues.append(
            {
                "severity": severity,
                "issue": issue,
                "utterance_id": row.get("utterance_id", ""),
                "path": row.get("path", ""),
                "detail": detail,
            }
        )

    def finalize_quality(self, row: dict[str, str], flags: list[str]) -> None:
        if row["utterance_id"] in self.seen_ids:
            flags.append("duplicate_utterance_id")
        self.seen_ids.add(row["utterance_id"])
        if row.get("duration_sec"):
            duration = float(row["duration_sec"])
            if duration < self.args.min_duration_sec:
                flags.append("short_duration")
            if duration > self.args.max_duration_sec:
                flags.append("long_duration")
        if any(flag in flags for flag in ("parse_error", "duplicate_utterance_id")):
            row["quality_status"] = "bad"
        elif flags:
            row["quality_status"] = "review"
        else:
            row["quality_status"] = "ok"
        row["quality_flags"] = ";".join(sorted(set(flags)))
        if row["quality_status"] != "ok":
            self.issue(row, row["quality_status"], row["quality_flags"], row.get("path", ""))

    def add_row(self, row: dict[str, str], flags: list[str]) -> None:
        for field in FIELDNAMES:
            row.setdefault(field, "")
        self.finalize_quality(row, flags)
        self.rows.append(row)

    def add_tts_family(
        self,
        wav_root: Path,
        latent_root: Path,
        source_kind: str,
        speaker_gender: str,
        texts: dict[str, str],
        categories: dict[str, str],
        captions: dict[str, str],
        origin: str,
        generation_model: str,
    ) -> None:
        wavs = path_index(wav_root, ".wav")
        latents = path_index(latent_root, ".pt", skip_feat=True)
        caption_keys = set(captions)
        for speaker_id, stem in sorted(set(wavs) | set(latents)):
            wav_path = wavs.get((speaker_id, stem))
            latent_path = latents.get((speaker_id, stem))
            feature_path = feature_path_for(latent_path)
            text_id, caption_key = parse_stem(stem, caption_keys)
            flags: list[str] = []
            if not text_id or not caption_key:
                flags.append("parse_error")
            if text_id and text_id not in texts:
                flags.append("unknown_text_id")
            if latent_path is None:
                flags.append("needs_encode")
            if latent_path is not None and feature_path is not None and not feature_path.exists():
                flags.append("needs_feature")
            duration, sample_rate, channels, frames = self.safe_audio_info(wav_path, flags)
            layer, source_type = self.tts_layer_source(source_kind, text_id, caption_key)
            row = {
                "utterance_id": f"{source_kind}__{speaker_id}__{text_id}__{caption_key}",
                "path": rel(latent_path or wav_path),
                "path_type": "latent_pt" if latent_path else "wav",
                "layer": layer,
                "source_type": source_type,
                "speaker_id": speaker_id,
                "speaker_gender": speaker_gender,
                "text": texts.get(text_id, ""),
                "text_id": text_id,
                "text_category": categories.get(text_id, ""),
                "caption_key": caption_key,
                "caption_text": captions.get(caption_key, ""),
                "duration_sec": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "n_frames": frames,
                "split": speaker_split(speaker_id),
                "license": "MIT",
                "wav_path": rel(wav_path),
                "latent_path": rel(latent_path),
                "feature_path": rel(feature_path) if feature_path else "",
                "origin": origin,
                "generation_model": generation_model,
                "ref_wav": rel(self.female_ref_wavs.get(speaker_id)) if speaker_gender == "F" else "",
            }
            self.add_row(row, flags)

    def tts_layer_source(self, source_kind: str, text_id: str, caption_key: str) -> tuple[str, str]:
        if source_kind == "male_tts":
            return "E", "tts_male_ja"
        if source_kind == "female_tts_live":
            return "B", "tts_jp_live"
        text_index = int(text_id[1:]) if text_id.startswith("t") and text_id[1:].isdigit() else 999
        if text_index <= 9 and caption_key in BASE_FEMALE_CAPTIONS:
            return "A", "tts_base"
        if caption_key in CHAR_FEMALE_CAPTIONS or text_index >= 10:
            return "B", "tts_emotional_live"
        return "A", "tts_base"

    def safe_audio_info(self, wav_path: Path | None, flags: list[str]) -> tuple[str, str, str, str]:
        if wav_path is None:
            flags.append("missing_wav")
            return "", "", "", ""
        try:
            return self.audio_info.get(wav_path)
        except Exception as exc:
            flags.append("invalid_wav")
            return "", "", "", ""

    def add_live_tts(self, wav_root: Path, latent_root: Path) -> None:
        if not wav_root.exists() and not latent_root.exists():
            return
        captions = self.female_captions
        wavs = path_index(wav_root, ".wav")
        latents = path_index(latent_root, ".pt", skip_feat=True)
        for speaker_id, stem in sorted(set(wavs) | set(latents)):
            wav_path = wavs.get((speaker_id, stem))
            latent_path = latents.get((speaker_id, stem))
            feature_path = feature_path_for(latent_path)
            text_id, caption_key = parse_stem(stem, set(captions))
            meta = self.live_texts.get(text_id, {})
            flags: list[str] = []
            if not text_id or not caption_key:
                flags.append("parse_error")
            if text_id and not meta:
                flags.append("unknown_text_id")
            if latent_path is None:
                flags.append("needs_encode")
            if latent_path is not None and feature_path is not None and not feature_path.exists():
                flags.append("needs_feature")
            duration, sample_rate, channels, frames = self.safe_audio_info(wav_path, flags)
            split = "golden" if meta.get("split") == "golden" else speaker_split(speaker_id)
            row = {
                "utterance_id": f"female_tts_live__{speaker_id}__{text_id}__{caption_key}",
                "path": rel(latent_path or wav_path),
                "path_type": "latent_pt" if latent_path else "wav",
                "layer": "B",
                "source_type": "tts_jp_live",
                "speaker_id": speaker_id,
                "speaker_gender": "F",
                "text": meta.get("text", ""),
                "text_id": text_id,
                "text_category": meta.get("category", ""),
                "caption_key": caption_key,
                "caption_text": captions.get(caption_key, ""),
                "style_tags": meta.get("style_tags", ""),
                "role": meta.get("role", ""),
                "persona": meta.get("persona", ""),
                "scene": meta.get("scene", ""),
                "relation": meta.get("relation", ""),
                "phoneme_focus": meta.get("phoneme_focus", ""),
                "duration_sec": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "n_frames": frames,
                "split": split,
                "text_split": meta.get("split", ""),
                "license": "MIT",
                "wav_path": rel(wav_path),
                "latent_path": rel(latent_path),
                "feature_path": rel(feature_path) if feature_path else "",
                "origin": "irodori_tts_japanese_live_vc_texts",
                "generation_model": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
                "ref_wav": rel(self.female_ref_wavs.get(speaker_id)),
            }
            self.add_row(row, flags)

    def add_real_female(self, wav_root: Path, latent_root: Path) -> None:
        wavs = path_index(wav_root, ".wav")
        latents = path_index(latent_root, ".pt", skip_feat=True)
        for speaker_id, stem in sorted(set(wavs) | set(latents)):
            wav_path = wavs.get((speaker_id, stem))
            latent_path = latents.get((speaker_id, stem))
            feature_path = feature_path_for(latent_path)
            flags: list[str] = []
            if latent_path is None:
                flags.append("needs_encode")
            if latent_path is not None and feature_path is not None and not feature_path.exists():
                flags.append("needs_feature")
            duration, sample_rate, channels, frames = self.safe_audio_info(wav_path, flags)
            row = {
                "utterance_id": f"real_female__{speaker_id}__{stem}",
                "path": rel(latent_path or wav_path),
                "path_type": "latent_pt" if latent_path else "wav",
                "layer": "D",
                "source_type": "real_female",
                "speaker_id": speaker_id,
                "speaker_gender": "F",
                "duration_sec": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "n_frames": frames,
                "split": speaker_split(speaker_id),
                "license": "unknown",
                "wav_path": rel(wav_path),
                "latent_path": rel(latent_path),
                "feature_path": rel(feature_path) if feature_path else "",
                "origin": "female-dataset",
            }
            self.add_row(row, flags)

    def add_vctk_pairs(self, root: Path) -> None:
        if not root.exists():
            return
        for split_dir in ("train", "eval"):
            directory = root / split_dir
            if not directory.exists():
                continue
            for pt in sorted(directory.glob("*.pt")):
                if pt.stem.endswith("_feat"):
                    continue
                feature_path = feature_path_for(pt)
                flags: list[str] = []
                if feature_path is not None and not feature_path.exists():
                    flags.append("needs_feature")
                row = {
                    "utterance_id": f"vctk_pair__{split_dir}__{pt.stem}",
                    "path": rel(pt),
                    "path_type": "pair_pt",
                    "layer": "E",
                    "source_type": "source_male",
                    "speaker_gender": "M",
                    "split": "val" if split_dir == "eval" else "train",
                    "license": "CCBY",
                    "latent_path": rel(pt),
                    "feature_path": rel(feature_path) if feature_path else "",
                    "origin": "vctk_phase3_pair",
                }
                self.add_row(row, flags)

    def build(self) -> None:
        self.add_vctk_pairs(as_input_path(self.args.vctk_pairs))
        self.add_real_female(as_input_path(self.args.female_real_wavs), as_input_path(self.args.female_real_latents))
        self.add_tts_family(
            as_input_path(self.args.female_tts_corpus),
            as_input_path(self.args.female_tts_latents),
            "female_tts",
            "F",
            self.female_texts,
            self.female_categories,
            self.female_captions,
            "irodori_tts_v3_female_corpus",
            "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        )
        self.add_live_tts(as_input_path(self.args.female_tts_live_corpus), as_input_path(self.args.female_tts_live_latents))
        self.add_tts_family(
            as_input_path(self.args.male_tts_corpus),
            as_input_path(self.args.male_tts_latents),
            "male_tts",
            "M",
            self.male_texts,
            self.male_categories,
            self.male_captions,
            "irodori_tts_v3_male_corpus",
            "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
        )

    def write(self) -> None:
        out_dir = as_input_path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        canonical_path = out_dir / "canonical_utterances.tsv"
        trainable_path = out_dir / "trainable_utterances.tsv"
        legacy_path = out_dir / "all_utterances.tsv"
        issues_path = out_dir / "manifest_issues.tsv"
        summary_path = out_dir / "manifest_summary.json"
        self.write_tsv(canonical_path, self.rows)
        trainable = [
            row
            for row in self.rows
            if row["path_type"] in ("latent_pt", "pair_pt") and row["quality_status"] == "ok"
        ]
        self.write_tsv(trainable_path, trainable)
        if not self.args.no_legacy_all:
            self.write_tsv(legacy_path, trainable)
        self.write_issues(issues_path)
        summary = self.summary(trainable)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"canonical: {canonical_path} ({len(self.rows)} rows)")
        print(f"trainable: {trainable_path} ({len(trainable)} rows)")
        if not self.args.no_legacy_all:
            print(f"legacy all_utterances: {legacy_path}")
        print(f"issues: {issues_path} ({len(self.issues)} rows)")
        print(f"summary: {summary_path}")

    def write_tsv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def write_issues(self, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["severity", "issue", "utterance_id", "path", "detail"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(self.issues)

    def summary(self, trainable: list[dict[str, str]]) -> dict[str, Any]:
        def count(rows: list[dict[str, str]], field: str) -> dict[str, int]:
            return dict(sorted(Counter(row[field] for row in rows).items()))

        flags = Counter()
        for row in self.rows:
            for flag in row["quality_flags"].split(";"):
                if flag:
                    flags[flag] += 1
        return {
            "path_base": str(TRAINING_DIR),
            "canonical_rows": len(self.rows),
            "trainable_rows": len(trainable),
            "canonical_by_source_type": count(self.rows, "source_type"),
            "trainable_by_source_type": count(trainable, "source_type"),
            "canonical_by_layer": count(self.rows, "layer"),
            "trainable_by_layer": count(trainable, "layer"),
            "canonical_by_split": count(self.rows, "split"),
            "trainable_by_split": count(trainable, "split"),
            "quality_status": count(self.rows, "quality_status"),
            "quality_flags": dict(sorted(flags.items())),
            "issue_count": len(self.issues),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "kansei_vc" / "manifests"))
    parser.add_argument("--female-tts-corpus", default=str(ROOT / "data" / "female_tts_corpus"))
    parser.add_argument("--female-tts-latents", default=str(ROOT / "data" / "female_tts_latents"))
    parser.add_argument("--female-tts-live-corpus", default=str(ROOT / "data" / "female_tts_live_corpus"))
    parser.add_argument("--female-tts-live-latents", default=str(ROOT / "data" / "female_tts_live_latents"))
    parser.add_argument("--female-real-wavs", default=str(ROOT / "female-dataset"))
    parser.add_argument("--female-real-latents", default=str(ROOT / "data" / "female_real_latents"))
    parser.add_argument("--male-tts-corpus", default=str(ROOT / "data" / "male_tts_corpus"))
    parser.add_argument("--male-tts-latents", default=str(ROOT / "data" / "male_tts_latents"))
    parser.add_argument("--vctk-pairs", default=str(ROOT / "data" / "phase3_10k"))
    parser.add_argument("--live-texts-tsv", default=str(ROOT / "data" / "kansei_vc" / "japanese_live_vc_texts.tsv"))
    parser.add_argument("--min-duration-sec", type=float, default=0.7)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    parser.add_argument("--no-legacy-all", action="store_true")
    args = parser.parse_args()
    builder = Builder(args)
    builder.build()
    builder.write()


if __name__ == "__main__":
    main()
