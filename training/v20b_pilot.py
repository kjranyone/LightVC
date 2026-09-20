"""V2-0b: Irodori 600M VoiceDesign による same_text パイロット生成（P 初期化用途の品質確認用）。

男声ソース = namikawa 実音声クローン + TTS 男声（male_tts_corpus 参照）
女声 target = female-dataset 実話者クローン ×2 + caption 設計声（対照）
テキスト = golden same_text 4 節（sibilance/breath/attack/tail guard）

規約準拠: TTS 生成音は GT にしない（P 初期化・alignment 研究用）。

    cd Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/v20b_pilot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/v20b")
CKPT = Path.home() / ".cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign/snapshots/e863a3a93e652e09afeff3e84823a206a0a60314/model.safetensors"

TEXTS = {
    "jp_sib_001": "ささやく声が、すこし静かすぎます。",
    "jp_sib_013": "ふわっと息を吐いて、話し始めた。",
    "jp_plosive_001": "ぱっと立ち止まって、手を叩いた。",
    "jp_plosive_007": "ねえ、もう少しだけ待って。",
}

NAMI = "/home/kojirotanaka/kjranyone/LightVC/namikawa.mp3"
MALE_TTS = "/home/kojirotanaka/kjranyone/LightVC/data/male_tts_corpus/male_p226/t00_neutral.wav"
FEM_A = "/home/kojirotanaka/kjranyone/LightVC/female-dataset/0005e65d3f11f99d/0005e65d3f11f99d_00002260.wav"
FEM_B = "/home/kojirotanaka/kjranyone/LightVC/female-dataset/00218f323fbaddbf/00218f323fbaddbf_00005064.wav"
CAPTION_DESIGN = "明るく可愛い若い女性の声。萌え声で、近い距離感でやわらかく話す。"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(CKPT),
        model_device="cuda",
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision="fp32",
        codec_device="cuda",
    ))
    refs = {
        "m_namikawa": dict(ref_wav=NAMI),
        "m_p226": dict(ref_wav=MALE_TTS),
        "f_realA": dict(ref_wav=FEM_A),
        "f_realB": dict(ref_wav=FEM_B),
        "f_design": dict(no_ref=True, caption=CAPTION_DESIGN),
    }
    for name, group in [("m_namikawa", "male"), ("m_p226", "male"),
                        ("f_realA", "female"), ("f_realB", "female"), ("f_design", "design")]:
        for tid, text in TEXTS.items():
            req = SamplingRequest(text=text, num_steps=40, seed=1234, **refs[name])
            res = runtime.synthesize(req)
            out = OUT / f"{group}_{name}_{tid}.wav"
            save_wav(out, res.audios[0], res.sample_rate)
            print(f"  {out.name}  {res.audios[0].shape[-1]/res.sample_rate:.2f}s", flush=True)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
