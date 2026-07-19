import argparse
import csv
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def db(value: float) -> float:
    return float(10.0 * np.log10(max(value, 1e-20)))


def ratio_db(numerator: float, denominator: float) -> float:
    return float(10.0 * np.log10(max(numerator, 1e-20) / max(denominator, 1e-20)))


def band_power(power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(mask):
        return 0.0
    return float(power[mask].sum())


def parse_delta(path: Path) -> float | None:
    stem = path.stem
    if stem == "base_no_adapter":
        return 0.0
    marker = "delta_"
    if marker not in stem:
        return None
    value = stem.split(marker, 1)[1].split("_", 1)[0]
    try:
        return float(value)
    except ValueError:
        return None


def analyze_file(path: Path, root: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, always_2d=False)
    y = mono(audio)
    y = y - np.mean(y)

    n_fft = 4096
    hop_length = 512
    spectrum = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    )
    power = spectrum * spectrum + 1e-20
    freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    total_power = float(power.sum())

    band_0_4 = band_power(power, freqs, 0.0, 4000.0)
    band_4_8 = band_power(power, freqs, 4000.0, 8000.0)
    band_8_12 = band_power(power, freqs, 8000.0, 12000.0)
    band_12_16 = band_power(power, freqs, 12000.0, 16000.0)
    band_16_22 = band_power(power, freqs, 16000.0, min(sample_rate / 2.0, 22050.0) + 1.0)
    band_8_16 = band_8_12 + band_12_16
    band_8_22 = band_8_16 + band_16_22

    high_mask = freqs >= 8000.0
    high_spectrum = spectrum[high_mask] + 1e-12
    high_power = power[high_mask]
    high_flatness = float(
        np.exp(np.mean(np.log(high_spectrum), axis=0)).mean()
        / (np.mean(high_spectrum, axis=0).mean() + 1e-12)
    )
    if high_power.shape[1] > 1:
        high_diff = np.diff(np.log(high_power), axis=1)
        high_flux = float(np.sqrt(np.mean(high_diff * high_diff)))
    else:
        high_flux = 0.0

    rms = float(np.sqrt(np.mean(y * y) + 1e-20))
    peak = float(np.max(np.abs(y))) if y.size else 0.0

    return {
        "file": str(path.relative_to(root)),
        "delta_scale": parse_delta(path),
        "sample_rate": int(sample_rate),
        "duration_sec": float(len(y) / sample_rate),
        "peak": peak,
        "rms_db": float(20.0 * np.log10(rms + 1e-20)),
        "centroid_hz": float(librosa.feature.spectral_centroid(S=spectrum, sr=sample_rate).mean()),
        "rolloff_85_hz": float(
            librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate, roll_percent=0.85).mean()
        ),
        "rolloff_95_hz": float(
            librosa.feature.spectral_rolloff(S=spectrum, sr=sample_rate, roll_percent=0.95).mean()
        ),
        "flatness": float(librosa.feature.spectral_flatness(S=spectrum).mean()),
        "hf_flatness": high_flatness,
        "hf_flux": high_flux,
        "band_0_4_total_db": db(band_0_4 / total_power),
        "band_4_8_total_db": db(band_4_8 / total_power),
        "band_8_16_total_db": db(band_8_16 / total_power),
        "band_16_22_total_db": db(band_16_22 / total_power),
        "band_8_22_total_db": db(band_8_22 / total_power),
        "band_8_16_vs_4_8_db": ratio_db(band_8_16, band_4_8),
        "band_16_22_vs_8_16_db": ratio_db(band_16_22, band_8_16),
        "band_8_22_vs_4_8_db": ratio_db(band_8_22, band_4_8),
    }


def add_relative_metrics(rows: list[dict[str, Any]], baseline_name: str | None) -> None:
    baseline = None
    if baseline_name:
        for row in rows:
            if Path(row["file"]).name == baseline_name:
                baseline = row
                break
    if baseline is None and rows:
        baseline = rows[0]
    if baseline is None:
        return

    relative_fields = [
        "centroid_hz",
        "rolloff_85_hz",
        "rolloff_95_hz",
        "flatness",
        "hf_flatness",
        "hf_flux",
        "band_8_16_vs_4_8_db",
        "band_16_22_vs_8_16_db",
        "band_8_22_vs_4_8_db",
    ]
    for row in rows:
        for field in relative_fields:
            row[f"d_{field}"] = float(row[field] - baseline[field])


def sort_key(path: Path) -> tuple[int, float, str]:
    delta = parse_delta(path)
    if delta is None:
        return (1, 0.0, path.name)
    return (0, delta, path.name)


def collect_wavs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob("*.wav"), key=sort_key)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary(path: Path, rows: list[dict[str, Any]], baseline_name: str | None) -> None:
    lines = [
        "# Kansei Audio Analysis",
        "",
        f"- Files: {len(rows)}",
        f"- Baseline: {baseline_name or (Path(rows[0]['file']).name if rows else '')}",
        "",
        "## Delta Safety Curve",
        "",
        "| File | Delta | Centroid Hz | 8-16/4-8 dB | d 8-16/4-8 | 16-22/8-16 dB | HF flatness | HF flux |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    Path(row["file"]).name,
                    format_value(row["delta_scale"]),
                    format_value(row["centroid_hz"]),
                    format_value(row["band_8_16_vs_4_8_db"]),
                    format_value(row["d_band_8_16_vs_4_8_db"]),
                    format_value(row["band_16_22_vs_8_16_db"]),
                    format_value(row["hf_flatness"]),
                    format_value(row["hf_flux"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If `8-16/4-8 dB` drops as delta increases, clarity is being traded for conversion strength.",
            "- If `16-22/8-16 dB` rises while `8-16/4-8 dB` drops, rough high-band noise is becoming more prominent.",
            "- A practical checkpoint should keep the curve stable at least through delta 0.50.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wavs = collect_wavs(input_dir)
    rows = [analyze_file(path, input_dir) for path in wavs]
    add_relative_metrics(rows, args.baseline)

    write_tsv(output_dir / "metrics.tsv", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary(output_dir / "summary.md", rows, args.baseline)

    print(f"analyzed {len(rows)} wav files")
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()

