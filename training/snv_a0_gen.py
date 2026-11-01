"""A0(synth_nam_vc §5・学習なし): 物差しの較正用の声の組を Irodori-TTS-v4.1-Small で作る。
声 = data/pair_tts_v4l の pfNNN(女 24)/ pmNNN(男 12)の caption + t00 の参照。1 声 8 本・文は 1〜15 から声ごとに回す(参照の文 0 は使わない)。
役: j0〜2 = 登録(素の読み)・j3〜5 = 検査(素の読み)・j6 = 検査(👂 囁き)・j7 = 検査(🫶 優しく)。--extra で女声だけ j8〜13 = 追加の検査(素の読み・A0c の 15 s の検査)。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_a0_gen.py --out ../LightVC/results/snv_a0/wav [--extra]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
PAIR = ROOT / "data/pair_tts_v4l"
ROLES = ("enroll", "enroll", "enroll", "test", "test", "test", "whisper", "tender")
PREFIX = {"whisper": "👂", "tender": "🫶"}


def items(extra: bool = False) -> list:
    man = json.loads((PAIR / "manifest.json").read_text())
    texts = [man["speakers"]["pm000"]["files"][f"t{k:02d}"]["text"] for k in range(1, 16)]
    voices = [f"pf{k:03d}" for k in range(24)] + [f"pm{k:03d}" for k in range(12)]
    roles = list(enumerate(ROLES)) if not extra else [(j, "test_extra") for j in range(8, 14)]
    out = []
    for v, name in enumerate(voices):
        if extra and not name.startswith("pf"):
            continue
        for j, role in roles:
            ti = (v + j) % len(texts)
            out.append({"voice": name, "sex": "F" if name.startswith("pf") else "M", "j": j, "role": role, "text_id": ti + 1,
                        "text": PREFIX.get(role, "") + texts[ti], "caption": man["speakers"][name]["caption"],
                        "seed": 20261101 + 100 * v + j})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--extra", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(a.ckpt)) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    its = items(a.extra)
    (od / ("items_extra.json" if a.extra else "items.json")).write_text(json.dumps(its, ensure_ascii=False, indent=1))
    for it in its:
        p = od / f"{it['voice']}_{it['j']}.wav"
        if p.exists():
            continue
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], ref_wav=str(PAIR / it["voice"] / "t00.wav"),
                                          num_steps=a.steps, seed=it["seed"]), log_fn=None)
        y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        sf.write(p, y, r.sample_rate)
        print(it["voice"], it["j"], it["role"], f"{len(y) / r.sample_rate:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
