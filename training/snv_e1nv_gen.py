"""A4 の学習データ(synth_nam_vc §5): E1 の 600 声(data/snv_e1)に音素の無い発声を演じさせる。Irodori-TTS-v4.1-Small・N0r と同じ作法
(参照 = その声の t00 の 1 本・説明文 = 動作と状況だけ・絵文字 前 2 + 後ろ 1 / 混ざりは 2 つずつ・本文 CFG 5.5・話者 CFG 4.0・説明文 CFG 3.0・40 step)。
種類 = 喘ぎ・悲鳴・吐息・息切れ・泣き声・笑い・混ざり × 2 本(--ks で作る本の番号を選ぶ・番号ごとに全ての声を回す)。文は N0 の 4 通りから声ごとに回す。乱数は N0 / N0r と別。
出力 = data/snv_e1nv/{voice}/{cat}_{k}.wav と manifest.json。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_e1nv_gen.py
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
E1 = ROOT / "data/snv_e1"
OUT = ROOT / "data/snv_e1nv"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snv_n0_gen import PER_CAT, TEXTS
from snv_n0r_gen import ACT, recipe_text


def items() -> list:
    man = json.loads((E1 / "manifest.json").read_text())
    out = []
    for v, row in enumerate(man["voices"]):
        for c, (cat, ts) in enumerate(TEXTS.items()):
            for k in range(PER_CAT):
                ti = (v + 2 * k + c) % len(ts)
                out.append({"voice": row["voice"], "sex": row["sex"], "cat": cat, "k": k, "text_id": ti, "text": recipe_text(ts[ti], cat),
                            "caption": ACT[cat], "seed": 20261202 + 1000 * v + 10 * c + k})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ks", default="0")
    a = ap.parse_args()
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(a.ckpt)) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    OUT.mkdir(parents=True, exist_ok=True)
    its = items()
    (OUT / "manifest.json").write_text(json.dumps({"model": a.ckpt, "steps": 40, "recipe": "snv_n0r_gen(r4)", "items": its}, ensure_ascii=False, indent=0))
    ks = [int(k) for k in a.ks.split(",")]
    for it in [it for k in ks for it in its if it["k"] == k][: a.limit or None]:
        p = OUT / it["voice"] / f"{it['cat']}_{it['k']}.wav"
        if p.exists():
            continue
        p.parent.mkdir(exist_ok=True)
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], ref_wav=str(E1 / it["voice"] / "t00.wav"), num_steps=40, seed=it["seed"],
                                          cfg_scale_text=5.5, cfg_scale_speaker=4.0, cfg_scale_caption=3.0), log_fn=None)
        y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        sf.write(p, y, r.sample_rate)
        print(it["voice"], it["cat"], it["k"], f"{len(y) / r.sample_rate:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
