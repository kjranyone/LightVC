"""D165: 実女声の発話と同じ文を男声 TTS(data/male_tts_v4l の話者・錨 t10 + 同じ caption)に読ませ、男声 → 実女声のパラレルの組を作る。
入力 = d165_real_parallel.py --stage select の items.json([{item, ti, spk, text, ...}])。出力 = <out>/p{item:04d}_t{ti:03d}_SRC.wav。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_male_for_real.py --items <s>/d165/items.json --out <s>/d165
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_male_corpus_v4l import ANCHOR, IRODORI

SRC = Path(__file__).resolve().parent.parent / "data/male_tts_v4l"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision=a.precision, codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    man = json.loads((SRC / "manifest.json").read_text())
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    for it in json.loads(Path(a.items).read_text()):
        wp = od / f"p{it['item']:04d}_t{it['ti']:03d}_SRC.wav"
        if wp.exists():
            continue
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=man["speakers"][it["spk"]]["caption"], num_steps=a.steps, seed=a.seed + it["item"],
                                          ref_wav=str(SRC / it["spk"] / f"t{ANCHOR:02d}.wav")), log_fn=None)
        sf.write(wp, r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate)
        print(it["item"], it["spk"], it["text"][:20], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
