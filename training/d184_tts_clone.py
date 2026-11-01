"""D184(評価の較正だけ・製品の設計 / 学習 / 推論に使わない・出力を学習の目標にしない): 大きなゼロショットの生成器(Irodori-TTS-v4-Large)に、
D165 の項目の文を目標の参照(items.json の ref = 表と同じ参照のクリップ)を手本に読ませる(TTSZ)。統制 = 評価するクリップそのものを手本に(TTSSELF)。caption なし。
出力 = <out>/p{item:04d}_t{ti:03d}_{TTSZ|TTSSELF}.wav → 独立の中心で d4b / d146。事前登録 = results/conv_p0/d184_prereg.yaml。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/d184_tts_clone.py --items <s>/d165/items.json --out <s>/d165
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_male_corpus_v4l import IRODORI


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--conds", default="TTSZ,TTSSELF")
    a = ap.parse_args()
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision=a.precision, codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out)
    for it in json.loads(Path(a.items).read_text()):
        for c in a.conds.split(","):
            wp = od / f"p{it['item']:04d}_t{it['ti']:03d}_{c}.wav"
            if wp.exists():
                continue
            refs = it["ref"] if c == "TTSZ" else [it["clip"]]
            r = rt.synthesize(SamplingRequest(text=it["text"], ref_wavs=None if len(refs) == 1 else refs, ref_wav=refs[0] if len(refs) == 1 else None,
                                              num_steps=a.steps, seed=a.seed + it["item"]), log_fn=None)
            sf.write(wp, r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate)
            print(it["item"], c, len(refs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
