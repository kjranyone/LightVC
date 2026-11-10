"""段 D2 の下見(results/snv_d2_pilot/prereg.yaml・paper_snv §5 M1): 同じ声・同じ文を、演技(E1 の説明文)と落ち着いた朗読(声質を書かない説明文)で読ませ、
声ごとに標準化した Y の 1 声を抜いた型の R²(内容が軌跡を決める割合)を比べる。
    cd ../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_d2_pilot.py --gen
    cd training && uv run --no-sync python snv_d2_pilot.py --measure
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
E1 = ROOT / "data/snv_e1"
OUT = ROOT / "data/snv_d2_pilot"
READ = "原稿を落ち着いた調子で淡々と読み上げている。静かな部屋でマイクに近い、クリアな録音。"
SEED = 20261107


def plan() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snv_e1_gen as G
    import train_snv_p1 as T
    e1 = json.loads((E1 / "manifest.json").read_text())
    dm = json.loads((ROOT / "data/snv_d/manifest.json").read_text())
    used = {f["text"] for v in e1["voices"] for f in v["files"]} | {f["text"] for v in dm["voices"] for f in v["files"]}
    pool = [t for t in G.text_pool() if t not in used and 12 <= len(t) <= 40]
    rng = random.Random(SEED)
    texts = rng.sample(pool, 20)
    tr, _ = T.split([v["voice"] for v in e1["voices"]])
    byv = {v["voice"]: v for v in e1["voices"]}
    fs = [v for v in tr if byv[v]["sex"] == "F"]
    ms = [v for v in tr if byv[v]["sex"] == "M"]
    voices = rng.sample(fs, 6) + rng.sample(ms, 2)
    return {"model": "Aratako/Irodori-TTS-v4.1-Small", "steps": 40, "texts": texts, "read_caption": READ,
            "voices": [{"voice": v, "sex": byv[v]["sex"], "caption": byv[v]["caption"], "seed": SEED + 100 * i, "ref": f"{v}/t00.wav"} for i, v in enumerate(voices)]}


def gen() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mp = OUT / "manifest.json"
    if not mp.exists():
        mp.write_text(json.dumps(plan(), ensure_ascii=False, indent=1))
    man = json.loads(mp.read_text())
    sys.path.insert(0, str(IRODORI))
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(man["model"])) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    for v in man["voices"]:
        for style, cap in (("act", v["caption"]), ("read", man["read_caption"])):
            d = OUT / style / v["voice"]
            d.mkdir(parents=True, exist_ok=True)
            for j, t in enumerate(man["texts"]):
                p = d / f"s{j:02d}.wav"
                if p.exists():
                    continue
                r = rt.synthesize(SamplingRequest(text=t, caption=cap, ref_wav=str(E1 / v["ref"]), no_ref=False, num_steps=man["steps"], seed=v["seed"] + j), log_fn=None)
                sf.write(p, r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate)
        print(v["voice"], flush=True)
    return 0


def measure() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snv_p0_front as P
    import snv_p_feat as F
    import train_snv_p1 as T
    from snv_diag.i3_share_probe import pool, r2_loo
    man = json.loads((OUT / "manifest.json").read_text())

    def vstat(v: str) -> tuple:
        z = np.concatenate([T.load_utt(p)["y"] for p in sorted((T.FEAT / f"e1/{v}").glob("t*.npz"))], 1)
        return z.mean(1, keepdims=True), z.std(1, keepdims=True) + 1e-6

    st = {v["voice"]: vstat(v["voice"]) for v in man["voices"]}
    res: dict = {"prereg": "results/snv_d2_pilot/prereg.yaml", "texts": len(man["texts"]), "voices": len(man["voices"])}
    for style in ("act", "read"):
        r2s, dur = [], []
        for j in range(len(man["texts"])):
            xs = []
            for v in man["voices"]:
                x = F.load48(OUT / style / v["voice"] / f"s{j:02d}.wav")
                x = x[: P.n_frames(len(x)) * P.H]
                d = P.analyze(x)
                y = T.to_y(d["env"], d["per"], d["f0"])
                m, s = st[v["voice"]]
                xs.append(pool((y - m) / s))
                dur.append(len(x) / P.SR)
            r2s.append(r2_loo(xs))
        res[style] = {"R2_mean": round(float(np.mean(r2s)), 4), "R2_per_text": [round(float(r), 3) for r in r2s], "dur_mean_s": round(float(np.mean(dur)), 2), "dur_cv": round(float(np.std(dur) / np.mean(dur)), 3)}
        print(style, json.dumps({k: v for k, v in res[style].items() if k != "R2_per_text"}), flush=True)
    a, b = res["act"]["R2_mean"], res["read"]["R2_mean"]
    res["P_read"] = bool(b >= 2 * a and b >= 0.3)
    (ROOT / "results/snv_d2_pilot/measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("P_read", res["P_read"], flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--measure", action="store_true")
    a = ap.parse_args()
    raise SystemExit(gen() if a.gen else measure())
