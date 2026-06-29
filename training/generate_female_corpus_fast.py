"""
Batch female voice corpus generation — single-process (model loaded once).

Usage:
  cd Irodori-TTS
  uv run --no-sync python ../LightVC/training/generate_female_corpus_fast.py \
    --female-dir ../LightVC/female-dataset \
    --output-dir ../LightVC/data/female_tts_corpus \
    --n-speakers 500 --texts-per-speaker 10
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--female-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-speakers", type=int, default=500)
    parser.add_argument("--texts-per-speaker", type=int, default=10)
    parser.add_argument("--captions", default="neutral,soft,breathy,warm,low_tension")
    parser.add_argument("--hf-checkpoint", default="Aratako/Irodori-TTS-600M-v3-VoiceDesign")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    refs = select_reference_clips(args.female_dir, args.n_speakers)
    caption_keys = [c.strip() for c in args.captions.split(",")]
    texts = TEXTS[:args.texts_per_speaker]
    total = len(refs) * len(texts) * len(caption_keys)

    print(f"Speakers: {len(refs)}, Texts: {len(texts)}, Captions: {len(caption_keys)}")
    print(f"Total target: {total} utterances\n")

    print("Loading Irodori-TTS model...")
    from huggingface_hub import hf_hub_download
    from irodori_tts.inference_runtime import (
        InferenceRuntime, RuntimeKey, SamplingRequest,
    )

    checkpoint_path = hf_hub_download(
        repo_id=args.hf_checkpoint, filename="model.safetensors",
    )
    print(f"Checkpoint: {checkpoint_path}")

    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device="cuda",
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision="bf16",
            codec_device="cuda",
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
        )
    )
    print("Model loaded.\n")

    done = 0
    skipped = 0
    t0 = time.time()

    for spk_id, ref_wav in refs:
        spk_dir = output_dir / spk_id
        spk_dir.mkdir(exist_ok=True)

        for ti, text in enumerate(texts):
            for ci, ck in enumerate(caption_keys):
                caption = CAPTIONS[ck]
                out_name = f"t{ti:02d}_{ck}.wav"
                out_path = spk_dir / out_name

                if out_path.exists():
                    skipped += 1
                    done += 1
                    continue

                try:
                    result = runtime.synthesize(
                        SamplingRequest(
                            text=text,
                            caption=caption,
                            ref_wav=ref_wav,
                            ref_latent=None,
                            ref_embed=None,
                            no_ref=False,
                            ref_normalize_db=-16.0,
                            ref_ensure_max=True,
                            num_candidates=1,
                            decode_mode="sequential",
                            seconds=None,
                            duration_scale=1.0,
                            max_ref_seconds=30.0,
                            num_steps=20,
                            cfg_scale_text=3.0,
                            cfg_scale_caption=3.0,
                            cfg_scale_speaker=5.0,
                            cfg_guidance_mode="independent",
                            cfg_scale=None,
                            cfg_min_t=0.5,
                            cfg_max_t=1.0,
                            truncation_factor=None,
                            rescale_k=None,
                            rescale_sigma=None,
                            context_kv_cache=True,
                            speaker_kv_scale=None,
                            speaker_kv_min_t=None,
                            speaker_kv_max_layers=None,
                            speaker_uncond_mode="mask",
                            seed=args.seed + done,
                            t_schedule_mode="linear",
                            sway_coeff=-1.0,
                            trim_tail=True,
                            tail_window_size=20,
                            tail_std_threshold=0.05,
                            tail_mean_threshold=0.1,
                            lora_adapter=None,
                        ),
                        log_fn=None,
                    )

                    audio = result.audio
                    sr = result.sample_rate

                    if hasattr(audio, "numpy"):
                        audio = audio.numpy()
                    if audio.ndim > 1:
                        audio = audio[0] if audio.shape[0] <= 2 else audio[:, 0]

                    if sr != 44100:
                        audio = librosa.resample(
                            audio.astype(np.float64), orig_sr=sr, target_sr=44100
                        ).astype(np.float32)

                    sf.write(str(out_path), audio, 44100)
                    done += 1

                    if done % 50 == 0:
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        eta = (total - done) / rate if rate > 0 else 0
                        print(
                            f"  [{done}/{total}] {spk_id}/{out_name} "
                            f"| {rate:.1f}/s ETA {eta/3600:.1f}h",
                            flush=True,
                        )

                except Exception as e:
                    print(f"  ERROR: {spk_id}/{out_name}: {e}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done}/{total} ({skipped} skipped) in {elapsed/3600:.1f}h")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
