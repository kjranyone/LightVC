"""V2-2: same_text 男女ペア大量生成（Irodori・P の Δlf0 写像教師用）。

80 文（same_text 32 + both_gate 48）× 男 4（TTS p226/p227 + namikawa 実 +
male_tts p232 実音声参照）× 女 8（female-dataset 実話者クローン）。
規約: TTS 生成は GT にしない（P 初期化・alignment 用・V2-3 の実音声 GT と区別）。

    cd Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/v22_pairs.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)

LVC = Path("/home/kojirotanaka/kjranyone/LightVC")
OUT = LVC / "data/same_text_pairs"
CKPT = (Path.home() / ".cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign"
        "/snapshots/e863a3a93e652e09afeff3e84823a206a0a60314/model.safetensors")

MALES = {
    "m226": str(LVC / "data/male_tts_corpus/male_p226/t00_neutral.wav"),
    "m227": str(LVC / "data/male_tts_corpus/male_p227/t00_neutral.wav"),
    "namikawa": str(LVC / "namikawa.mp3"),
    "m232": str(LVC / "data/male_tts_corpus/male_p232/t00_neutral.wav"),
}
FEMALES = {
    "fA": str(LVC / "female-dataset/0005e65d3f11f99d/0005e65d3f11f99d_00002260.wav"),
    "fB": str(LVC / "female-dataset/00218f323fbaddbf/00218f323fbaddbf_00005064.wav"),
    "fC": str(LVC / "female-dataset/000883f1d8ffe583/000883f1d8ffe583_00013321.wav"),
    "fD": str(LVC / "female-dataset/0025e9516c36b547/0025e9516c36b547_00008969.wav"),
    "fE": str(LVC / "female-dataset/0040664b2efd368c/0040664b2efd368c_00017050.wav"),
    "fF": str(LVC / "female-dataset/004c6b19ad8cd958/004c6b19ad8cd958_00011052.wav"),
    "fG": str(LVC / "female-dataset/007c9e6d5c047b7d/007c9e6d5c047b7d_00005162.wav"),
    "fH": str(LVC / "female-dataset/007fc1fa0d86bdbc/007fc1fa0d86bdbc_00005266.wav"),
}


def main() -> int:
    rows = list(csv.DictReader(open(LVC / "data/kansei_vc/japanese_live_vc_texts.tsv"),
                               delimiter="\t"))
    texts = [(r["text_id"], r["text"]) for r in rows
             if r["role"] in ("same_text", "both_gate")]
    print(f"  {len(texts)} texts x {len(MALES)}M x {len(FEMALES)}F", flush=True)
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(CKPT), model_device="cuda",
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision="fp32", codec_device="cuda",
    ))
    n = 0
    for name, ref in FEMALES.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for tid, text in texts:
            req = SamplingRequest(text=text, ref_wav=ref, num_steps=40, seed=1234)
            res = runtime.synthesize(req)
            save_wav(d / f"{tid}.wav", res.audios[0], res.sample_rate)
            n += 1
        print(f"  {name} done ({n})", flush=True)
    for name, ref in MALES.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        for tid, text in texts:
            req = SamplingRequest(text=text, ref_wav=ref, num_steps=40, seed=1234)
            res = runtime.synthesize(req)
            save_wav(d / f"{tid}.wav", res.audios[0], res.sample_rate)
            n += 1
        print(f"  {name} done ({n})", flush=True)
    print(f"-> {OUT}  total {n} files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
