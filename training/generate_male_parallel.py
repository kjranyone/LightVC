"""男声 → 女声の変換を解析で採点するためのパラレル文: data/male_tts_v4l の男性話者(錨 t10 + 設計文で同じ声)に、女声 TTS コーパスと同じ 10 文(generate_female_corpus_fast.TEXTS)を読ませる。
変換の出力は、目標の女性が同じ文を読んだ本物と DTW で合わせて比べられる。出力 = <out>/<spk>/fNN.wav(48kHz)と manifest.json。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_male_parallel.py --out ../LightVC/data/male_tts_v4l_par --n_spk 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_female_corpus_fast import TEXTS as FTEXTS
from generate_male_corpus_v4l import ANCHOR, IRODORI

SRC = Path(__file__).resolve().parent.parent / "data/male_tts_v4l"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=24)
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261010)
    a = ap.parse_args()
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision=a.precision, codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    src_man = json.loads((SRC / "manifest.json").read_text())
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    mpath = od / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": "Aratako/Irodori-TTS-v4-Large", "steps": a.steps, "texts": FTEXTS, "source": str(SRC), "speakers": {}}
    for si, spk in enumerate(sorted(src_man["speakers"])[: a.n_spk]):
        cap = src_man["speakers"][spk]["caption"]
        anchor = SRC / spk / f"t{ANCHOR:02d}.wav"
        sd = od / spk; sd.mkdir(exist_ok=True)
        ent = man["speakers"].setdefault(spk, {"caption": cap, "anchor": str(anchor), "files": {}})
        t0 = time.time()
        for ti, text in enumerate(FTEXTS):
            wp = sd / f"f{ti:02d}.wav"
            if wp.exists():
                continue
            r = rt.synthesize(SamplingRequest(text=text, caption=cap, num_steps=a.steps, seed=a.seed + si * 100 + ti, ref_wav=str(anchor)), log_fn=None)
            y = r.audio.detach().float().cpu().numpy().reshape(-1)
            sf.write(wp, y.astype(np.float32), r.sample_rate)
            ent["files"][f"f{ti:02d}"] = {"text": text, "sr": r.sample_rate, "sec": round(len(y) / r.sample_rate, 3), "seed": r.used_seed}
            mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print(spk, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
