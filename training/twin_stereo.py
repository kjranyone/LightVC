"""双子ステレオ合成: 0aaa4d1a55dddc9a 参照クローン2誘導を L(twinA)/R(twinB) に配置。

台本: sibilance / breath / plosive / tail / 長文 のガード込み5文。
各文ごとに stereo wav（L=twinA, R=twinB）を出力。同一seedペアで誘導差を両耳で比較できる。

    cd ~/kjranyone/Irodori-TTS && HF_HUB_OFFLINE=1 .venv/bin/python ../LightVC/training/twin_stereo.py
"""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"))
from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
)

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/twin_0aaa4d1a")
CKPT = Path.home() / ".cache/huggingface/hub/models--Aratako--Irodori-TTS-600M-v3-VoiceDesign/snapshots/e863a3a93e652e09afeff3e84823a206a0a60314/model.safetensors"
REF = "/home/kojirotanaka/kjranyone/LightVC/female-dataset/0aaa4d1a55dddc9a/0aaa4d1a55dddc9a_00010059.wav"

SCRIPT = {
    "g1_sib": "（ささやき）しゅっ、しゅっと息が漏れるくらい、静かに話してみるね。",
    "g2_breath": "ふぅ……深呼吸して、ゆっくり落ち着くの。",
    "g3_plosive": "ぱっぱっぱっと手を叩いて、元気に挨拶してみよう。",
    "g4_tail": "ねえ、もう少しだけ、こうしていたいなぁ……。",
    "g5_long": "今日はね、ずっと話したかったことがあったの。ゆっくりでいいから、最後まで聞いてほしいな。",
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
    for tid, text in SCRIPT.items():
        wavs = {}
        for twin_id, seed in [("twinA", 1234), ("twinB", 5678)]:
            mono_path = OUT / f"st_{twin_id}_{tid}.wav"
            if not mono_path.exists():
                req = SamplingRequest(text=text, num_steps=40, seed=seed, ref_wav=REF)
                res = runtime.synthesize(req)
                sf.write(mono_path, res.audios[0].reshape(-1), res.sample_rate)
            wavs[twin_id], sr = sf.read(mono_path)
        n = max(len(wavs["twinA"]), len(wavs["twinB"]))
        st = np.zeros((n, 2), dtype=np.float32)
        st[: len(wavs["twinA"]), 0] = wavs["twinA"]
        st[: len(wavs["twinB"]), 1] = wavs["twinB"]
        out = OUT / f"stereo_{tid}.wav"
        sf.write(out, st, sr)
        print(f"  {out.name}  {n/sr:.2f}s", flush=True)
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
