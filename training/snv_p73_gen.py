"""段 P73 step b(results/snv_p73/prereg.yaml): 別の生成器の声を参照にして Irodori-TTS-v4.1-Small が読んだ声(声は別の生成器・演じ手は Irodori)。
参照 = data/snv_p73/gm/<声>.flac(sttts-gui に同梱の Gemini TTS の合成の声・実在の人の録音ではない)。文 = snv_e1_gen.text_pool(E1 と別の種)。caption は与えない。測りの入力としてだけ使う(学習に使わない)。
出力 = data/snv_p73/gm_ir/<声>/tNN.wav と data/snv_p73/gm_ir/manifest.json。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_p73_gen.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
import snv_e1_gen as E1

OUT = ROOT / "data/snv_p73/gm_ir"
N_UTT = 5


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    refs = sorted((ROOT / "data/snv_p73/gm").glob("*.flac"))
    rng = random.Random(7301)
    pool = E1.text_pool()
    man = {"model": "Aratako/Irodori-TTS-v4.1-Small", "steps": 40, "voices": [{"voice": p.stem, "ref": str(p.relative_to(ROOT)), "seed": 73010000 + 1000 * i,
                                                                                 "files": [{"t": f"t{j:02d}", "text": t} for j, t in enumerate(rng.sample(pool, N_UTT))]} for i, p in enumerate(refs)]}
    (OUT / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
    sys.path.insert(0, str(E1.IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(man["model"])) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim", model_precision="fp32", codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    for v in man["voices"]:
        d = OUT / v["voice"]
        d.mkdir(exist_ok=True)
        for j, f in enumerate(v["files"]):
            p = d / f"{f['t']}.wav"
            if p.exists():
                continue
            r = rt.synthesize(SamplingRequest(text=f["text"], ref_wav=str(ROOT / v["ref"]), num_steps=man["steps"], seed=v["seed"] + j), log_fn=None)
            sf.write(p, r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate)
        print(v["voice"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
