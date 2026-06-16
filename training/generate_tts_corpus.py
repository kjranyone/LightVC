"""
Generate diverse multi-speaker training data using Edge TTS.

Edge TTS provides 300+ neural TTS voices across languages.
We use English voices with diverse characteristics:
- Male/female
- Different accents (US, UK, AU, IN)
- Different voice qualities

Output: {speaker_id}/{utt_id}.wav at 24kHz (Edge TTS native rate)
"""

import asyncio
import os
import random
import edge_tts
import soundfile as sf
import numpy as np

# Diverse English voices from Edge TTS
VOICES = {
    # US English
    "en-US-AriaNeural": "f_us_aria",
    "en-US-AnaNeural": "f_us_ana",
    "en-US-JennyNeural": "f_us_jenny",
    "en-US-MichelleNeural": "f_us_michelle",
    "en-US-AmberNeural": "f_us_amber",
    "en-US-AshleyNeural": "f_us_ashley",
    "en-US-CoraNeural": "f_us_cora",
    "en-US-ElizabethNeural": "f_us_elizabeth",
    "en-US-MonicaNeural": "f_us_monica",
    "en-US-SaraNeural": "f_us_sara",
    "en-US-NancyNeural": "f_us_nancy",
    "en-US-ChristopherNeural": "m_us_christopher",
    "en-US-EricNeural": "m_us_eric",
    "en-US-GuyNeural": "m_us_guy",
    "en-US-RogerNeural": "m_us_roger",
    "en-US-BrandonNeural": "m_us_brandon",
    "en-US-ChristopherNeural": "m_us_chris2",
    "en-US-DavisNeural": "m_us_davis",
    "en-US-JasonNeural": "m_us_jason",
    "en-US-TonyNeural": "m_us_tony",
    # UK English
    "en-GB-SoniaNeural": "f_gb_sonia",
    "en-GB-LibbyNeural": "f_gb_libby",
    "en-GB-MaisieNeural": "f_gb_maisie",
    "en-GB-RyanNeural": "m_gb_ryan",
    "en-GB-ThomasNeural": "m_gb_thomas",
    # AU English
    "en-AU-NatashaNeural": "f_au_natasha",
    "en-AU-WilliamNeural": "m_au_william",
    # CA English
    "en-CA-ClaraNeural": "f_ca_clara",
    "en-CA-LiamNeural": "m_ca_liam",
}

# Diverse text samples for natural speech variation
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on sunny afternoons.",
    "Technology has transformed the way we communicate and interact with each other.",
    "The ancient library held thousands of books collected over many centuries.",
    "Music brings people together regardless of their background or culture.",
    "The mountain trail wound through forests of pine and oak trees.",
    "Innovation requires creativity, persistence, and a willingness to take risks.",
    "The chef prepared a delicious meal using fresh ingredients from the garden.",
    "Children laughed and played in the park on that warm summer evening.",
    "The scientist conducted experiments to understand the fundamental laws of nature.",
    "Every morning she would walk to the cafe and order a small black coffee.",
    "The city skyline glowed with lights as night fell over the bustling streets.",
    "He had always dreamed of traveling the world and experiencing different cultures.",
    "The old wooden boat rocked gently on the calm waters of the mountain lake.",
    "Learning a new language opens doors to understanding other perspectives.",
    "The concert hall was filled with the beautiful sound of the symphony orchestra.",
    "She discovered a hidden path leading to a secret garden behind the old wall.",
    "The autumn leaves created a colorful carpet across the forest floor.",
    "Innovation in renewable energy is crucial for a sustainable future.",
    "The artist painted with bold strokes, creating a vibrant and expressive canvas.",
    "Time passes differently when you are fully absorbed in something you love.",
    "The lighthouse stood tall against the stormy sky, guiding ships to safety.",
    "A good book can transport you to another world entirely.",
    "The recipe called for ingredients that she had never used before.",
    "Rain drummed softly against the window as she sat reading by the fire.",
    "The marathon runner trained every day, regardless of the weather.",
    "Digital privacy is becoming increasingly important in our connected world.",
    "The garden was filled with roses, lavender, and the sound of bees.",
    "He fixed the old radio and suddenly music filled the room again.",
    "The negotiations lasted well into the night before an agreement was reached.",
]


async def generate_voice(voice_name, speaker_id, output_dir, n_utterances=10):
    """Generate n utterances for one speaker."""
    spk_dir = os.path.join(output_dir, speaker_id)
    os.makedirs(spk_dir, exist_ok=True)

    texts = random.sample(TEXTS, min(n_utterances, len(TEXTS)))
    communicate = edge_tts.Communicate

    for i, text in enumerate(texts):
        out_path = os.path.join(spk_dir, f"{speaker_id}_{i:03d}.wav")
        if os.path.exists(out_path):
            continue

        try:
            comm = communicate(text, voice_name)
            await comm.save(out_path)

            # Verify the file is valid
            wav, sr = sf.read(out_path)
            if len(wav) < sr * 1.0:  # skip too short
                os.remove(out_path)
                continue

        except Exception as e:
            print(f"  Error {speaker_id}_{i}: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)


async def main():
    output_dir = "../data/tts_corpus"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating TTS corpus with {len(VOICES)} speakers...")
    print(f"Output: {output_dir}")

    tasks = []
    for voice_name, speaker_id in VOICES.items():
        tasks.append(
            generate_voice(voice_name, speaker_id, output_dir, n_utterances=10)
        )

    await asyncio.gather(*tasks)

    # Summary
    import glob

    speakers = sorted(os.listdir(output_dir))
    total = len(glob.glob(os.path.join(output_dir, "**", "*.wav"), recursive=True))
    print(f"\nDone: {total} utterances from {len(speakers)} speakers")
    for spk in speakers:
        n = len(glob.glob(os.path.join(output_dir, spk, "*.wav")))
        print(f"  {spk}: {n}")


if __name__ == "__main__":
    asyncio.run(main())
