"""双子クローン: 0aaa4d1a55dddc9a_00010059.wav を参照に Irodori クローンを2誘導分生成。

    cd ~/kjranyone/Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/twin_clone.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/twin_0aaa4d1a")
CKPT = Path.home() / ".cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign/snapshots/e863a3a93e652e09afeff3e84823a206a0a60314/model.safetensors"
REF = "/home/kojirotanaka/kjranyone/LightVC/female-dataset/0aaa4d1a55dddc9a/0aaa4d1a55dddc9a_00010059.wav"

TEXTS = {
    "jp_sib_001": "ささやく声が、すこし静かすぎます。",
    "jp_plosive_007": "ねえ、もう少しだけ待って。",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = InferenceRuntime.from_key(RuntimeKey(
        checkpoint=str(CKPT),
        model_device="cuda",
        codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision="fp32",
        codec_device="cuda",
    ))
    for twin_id, seed in [("twinA", 1234), ("twinB", 5678)]:
        for tid, text in TEXTS.items():
            req = SamplingRequest(text=text, num_steps=40, seed=seed, ref_wav=REF)
            res = runtime.synthesize(req)
            out = OUT / f"{twin_id}_{tid}.wav"
            save_wav(out, res.audios[0], res.sample_rate)
            print(f"  {out.name}  {res.audios[0].shape[-1]/res.sample_rate:.2f}s", flush=True)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
