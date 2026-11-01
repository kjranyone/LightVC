"""E0(synth_nam_vc §6・学習なし): 種の結合 P1 の対を Irodori-TTS-v4.1-Small で作る。
1 項目 = 目標役の女声 Y(種 s・長さは継続時間の予測)・操作者役の男声 XC(同じ文・同じ種 s・同じ長さ)・対照 XI(同じ文・種 s + 1000・同じ長さ)。
声 = data/pair_tts_v4l の pmNNN / pfNNN の t00(caption + 参照)を無作為に組む。trim_tail なし(長さを揃える)。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_e0_gen.py --out ../LightVC/results/snv_e0/wav
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
PAIR = ROOT / "data/pair_tts_v4l"
STYLES = ("", "", "👂", "🫶")


def items(n: int, seed: int) -> list:
    man = json.loads((PAIR / "manifest.json").read_text())
    texts = [f["text"] for f in man["speakers"]["pm000"]["files"].values()]
    ms = sorted(k for k in man["speakers"] if k.startswith("pm"))
    fs = sorted(k for k in man["speakers"] if k.startswith("pf"))
    rng = random.Random(seed)
    out = []
    for i in range(n):
        m, f = rng.choice(ms), rng.choice(fs)
        out.append({"item": i, "m": m, "f": f, "m_cap": man["speakers"][m]["caption"], "f_cap": man["speakers"][f]["caption"],
                    "style": STYLES[i % len(STYLES)], "text": STYLES[i % len(STYLES)] + texts[i % len(texts)], "seed": 20261101 + i})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="Aratako/Irodori-TTS-v4.1-Small")
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
    its = items(a.n, 1101)
    (od / "items.json").write_text(json.dumps(its, ensure_ascii=False, indent=1))

    def gen(it: dict, who: str, seed: int, seconds: float | None) -> tuple:
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it[f"{who}_cap"], ref_wav=str(PAIR / it[who] / "t00.wav"), num_steps=a.steps,
                                          seed=seed, seconds=seconds, trim_tail=False), log_fn=None)
        return r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate

    for it in its:
        p = od / f"i{it['item']:03d}"
        if (p.parent / f"{p.name}_XI.wav").exists():
            continue
        y, sr = gen(it, "f", it["seed"], None)
        sec = (len(y) + 0.5) / sr
        xc, _ = gen(it, "m", it["seed"], sec)
        xi, _ = gen(it, "m", it["seed"] + 1000, sec)
        assert len(xc) == len(y) == len(xi), (len(y), len(xc), len(xi))
        sf.write(f"{p}_Y.wav", y, sr)
        sf.write(f"{p}_XC.wav", xc, sr)
        sf.write(f"{p}_XI.wav", xi, sr)
        print(it["item"], it["m"], it["f"], it["style"] or "-", f"{sec:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
