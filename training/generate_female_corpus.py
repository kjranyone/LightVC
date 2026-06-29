"""
Batch female voice corpus generation using Irodori-TTS VoiceDesign.

Output: female_tts_corpus/{speaker_id}/{text_id}_{caption_id}.wav (44.1kHz mono)

Usage:
  cd Irodori-TTS
  uv run --no-sync python ../LightVC/training/generate_female_corpus.py \
    --female-dir ../LightVC/female-dataset \
    --output-dir ../LightVC/data/female_tts_corpus \
    --n-speakers 5 --texts-per-speaker 2 --captions neutral,soft
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

TEXTS = [
    "こんにちは。今日はとても良い天気ですね。",
    "私は音楽を聴くことが大好きで、特にジャズが好きです。",
    "少しゆっくり話してもよろしいですか？最近少し疲れていて。",
    "この本を読んでみてください。とても面白いですよ。",
    "今日の夕食は何にしましょうか。何かリクエストはありますか？",
    "最近、朝の散歩を日課にしています。気持ちが良いんです。",
    "あの人の声には、どこか懐かしい響きがある気がします。",
    "問題があれば、いつでも相談してくださいね。",
    "夜になると、少し寂しい気持ちになることがあります。",
    "この花の香り、とても好き。思い出すのは子供の頃の夏です。",
]

CAPTIONS = {
    "neutral": "落ち着いた自然な女性の声で、普通の速さで読み上げてください。",
    "soft": "柔らかく穏やかな声で、優しく語りかけるように読み上げてください。",
    "breathy": "息多めの甘い声で、囁くように親密な距離感で読み上げてください。",
    "warm": "温かく包容力のある声で、安心させるようにゆっくりと読み上げてください。",
    "low_tension": "リラックスして力の抜いた声で、少し低めのトーンで読み上げてください。",
}


def select_reference_clips(female_dir, n_speakers):
    speaker_dirs = sorted([d for d in Path(female_dir).iterdir() if d.is_dir()])
    rng = np.random.default_rng(42)
    n = min(n_speakers, len(speaker_dirs))
    selected = list(rng.choice(len(speaker_dirs), size=n, replace=False))

    refs = []
    for idx in selected:
        sd = speaker_dirs[idx]
        wavs = sorted(sd.glob("*.wav"))
        if not wavs:
            continue
        best = max(wavs, key=lambda w: sf.info(str(w)).frames)
        refs.append((sd.name, str(best)))
    return refs


def resample_to_44k(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 44100:
        audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=44100).astype(np.float32)
    return audio, 44100


def run_irodori(text, caption, ref_wav, output_wav, hf_checkpoint, seed=42):
    tmp_wav = "/tmp/irodori_tmp.wav"
    cmd = [
        sys.executable, "infer.py",
        "--hf-checkpoint", hf_checkpoint,
        "--text", text,
        "--ref-wav", str(ref_wav),
        "--caption", caption,
        "--output-wav", tmp_wav,
        "--num-steps", "20",
        "--model-precision", "bf16",
        "--seed", str(seed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not Path(tmp_wav).exists():
        print(f"  ERROR: {result.stderr[:200]}")
        return False

    audio, sr = resample_to_44k(tmp_wav)
    sf.write(str(output_wav), audio, sr)
    Path(tmp_wav).unlink(missing_ok=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--female-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-speakers", type=int, default=100)
    parser.add_argument("--texts-per-speaker", type=int, default=5)
    parser.add_argument("--captions", default="neutral,soft,breathy,warm,low_tension")
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-600M-v3-VoiceDesign")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = select_reference_clips(args.female_dir, args.n_speakers)
    print(f"Selected {len(refs)} speakers")

    caption_keys = [c.strip() for c in args.captions.split(",")]
    texts = TEXTS[:args.texts_per_speaker]

    total = len(refs) * len(texts) * len(caption_keys)
    print(f"Plan: {len(refs)} speakers x {len(texts)} texts x {len(caption_keys)} captions = {total} utterances\n")

    done = 0
    for spk_id, ref_wav in refs:
        spk_dir = output_dir / spk_id
        spk_dir.mkdir(exist_ok=True)

        for ti, text in enumerate(texts):
            for ci, ck in enumerate(caption_keys):
                caption = CAPTIONS[ck]
                out_name = f"t{ti:02d}_{ck}.wav"
                out_path = spk_dir / out_name

                if out_path.exists():
                    done += 1
                    continue

                ok = run_irodori(
                    text, caption, ref_wav, out_path,
                    args.hf_checkpoint, args.seed + done,
                )
                if ok:
                    done += 1
                    info = sf.info(str(out_path))
                    print(f"  [{done}/{total}] {spk_id}/{out_name} ({info.frames/info.samplerate:.1f}s)", flush=True)
                else:
                    print(f"  SKIP: {spk_id}/{out_name}")

    print(f"\nDone: {done}/{total} utterances -> {output_dir}")


if __name__ == "__main__":
    main()
