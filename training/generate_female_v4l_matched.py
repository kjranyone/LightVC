"""D328: dev の男声の操作者(data/male_tts_v4l の m000〜m029・Irodori-TTS-v4-Large)と同じ TTS・同じ作り方・同じ文で女声の操作者 f000〜f029 を作る(内容と TTS をそろえた男声の分の計測用・dev の操作者は m000〜m047 の 48 人 = --n_spk 48)。
設計文は generate_male_corpus_v4l.py と同じ組み合わせで『日本人女性』に替える。話者 fNNN ↔ mNNN。作る文 = 錨 t10 + dev の組で mNNN が使った文 + 操作者の参照用に名前順の残り 7 文。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_female_v4l_matched.py --dev <dev> --out ../LightVC/data/female_tts_v4l_matched
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
sys.path.insert(0, str(IRODORI))
sys.path.insert(0, str(Path(__file__).parent))
from generate_male_corpus_v4l import AGES, ANCHOR, PITCH, STYLE, TIMBRE


def captions(n: int, seed: int) -> list[str]:
    combos = list(itertools.product(AGES, PITCH, TIMBRE, STYLE))
    random.Random(seed).shuffle(combos)
    return [f"{a}の日本人女性。{p}{t}で、{s}静かな部屋でマイクに近い、クリアな録音。" for a, p, t, s in combos[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=30)
    ap.add_argument("--n_ref", type=int, default=7)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261019)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    texts = json.loads((ROOT / "data/male_tts_v4l/manifest.json").read_text())["texts"]
    used: dict = {}
    for r in json.loads((Path(a.dev) / "devset.json").read_text())["pairs"]:
        used.setdefault(r["spk"], []).extend(r["used"])
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    mpath = od / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": "Aratako/Irodori-TTS-v4-Large", "precision": "fp32", "steps": a.steps, "texts": texts, "speakers": {}, "matched_to": "data/male_tts_v4l"}
    caps = captions(a.n_spk, a.seed)
    for si, cap in enumerate(caps):
        spk, msp = f"f{si:03d}", f"m{si:03d}"
        u = list(dict.fromkeys(used.get(msp, [])))
        rest = [f"t{i:02d}" for i in range(len(texts)) if f"t{i:02d}" not in u and i != ANCHOR][: a.n_ref]
        order = [ANCHOR] + [int(k[1:]) for k in u + rest]
        sd = od / spk
        sd.mkdir(exist_ok=True)
        ent = man["speakers"].setdefault(spk, {"caption": cap, "seed": a.seed + si, "matched": msp, "files": {}})
        t0 = time.time()
        for ti in order:
            wp = sd / f"t{ti:02d}.wav"
            if wp.exists():
                continue
            anchor = sd / f"t{ANCHOR:02d}.wav"
            req = dict(text=texts[ti], caption=cap, num_steps=a.steps, seed=a.seed + si * 100 + ti)
            req.update(no_ref=True) if ti == ANCHOR else req.update(ref_wav=str(anchor))
            r = rt.synthesize(SamplingRequest(**req), log_fn=None)
            y = r.audio.detach().float().cpu().numpy().reshape(-1)
            sf.write(wp, y.astype(np.float32), r.sample_rate)
            ent["files"][f"t{ti:02d}"] = {"text": texts[ti], "sr": r.sample_rate, "sec": round(len(y) / r.sample_rate, 3), "seed": r.used_seed, "ref": None if ti == ANCHOR else anchor.name}
            mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print(spk, cap, len(order), f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
