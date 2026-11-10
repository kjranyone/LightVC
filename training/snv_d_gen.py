"""段 D(paper_snv §5 M1・§6 段 D): 内容を共有する Irodori のコーパス。E1 の 600 声(data/snv_e1)が、E1 で使っていない台本の文を、各文 K 声ずつ読む(各声 M 文)。
声の固定 = その声の E1 の t00 を参照・説明文 = E1 と同じ caption・Irodori-TTS-v4.1-Small・40 step(E1 と同じ作法)。文の同一性(content)は M3 の演奏の事前分布の条件にだけ使う(音素・文字列は使わない)。
K と M は段 T の T3(results/snv_t/verdict.yaml)から決める。出力 = data/snv_d/{voice}/sNN.wav と data/snv_d/manifest.json。透かしは外さない。
    cd training && uv run --no-sync python snv_d_gen.py --plan --K 8 --M 20       # 台帳だけ
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_d_gen.py --shard 0 --nshards 3
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
E1 = ROOT / "data/snv_e1"
OUT = ROOT / "data/snv_d"
SEED = 20261106


def plan(K: int, M: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snv_e1_gen as G
    e1 = json.loads((E1 / "manifest.json").read_text())
    used = {f["text"] for v in e1["voices"] for f in v["files"]}
    pool = [t for t in G.text_pool() if t not in used]
    voices = e1["voices"]
    n_text = len(voices) * M // K
    rng = random.Random(SEED)
    assert n_text <= len(pool), f"未使用の文 {len(pool)} が足りない(要 {n_text})"
    texts = rng.sample(pool, n_text)
    cap = {v["voice"]: M for v in voices}
    files: dict = {v["voice"]: [] for v in voices}
    for ci, t in enumerate(texts):
        avail = [v for v, c in cap.items() if c > 0]
        if len(avail) < K:
            break
        for v in rng.sample(avail, K):
            files[v].append({"s": f"s{len(files[v]):02d}", "text": t, "content": ci})
            cap[v] -= 1
    out = [{"voice": v["voice"], "sex": v["sex"], "caption": v["caption"], "seed": v["seed"] + 100, "ref": f"{v['voice']}/t00.wav", "files": files[v["voice"]]} for v in voices]
    cross = sum(1 for ci in range(len(texts)) if len({v["sex"] for v in out for f in v["files"] if f["content"] == ci}) == 2)
    return {"model": e1["model"], "steps": e1["steps"], "K": K, "M": M, "n_text": len(texts), "n_text_cross_sex": cross, "n_pool_unused": len(pool), "voices": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mp = OUT / "manifest.json"
    if a.plan:
        man = plan(a.K, a.M)
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        n = sum(len(v["files"]) for v in man["voices"])
        print(f"plan K {man['K']} M {man['M']} texts {man['n_text']} (cross-sex {man['n_text_cross_sex']}) utts {n} unused pool {man['n_pool_unused']}", flush=True)
        return 0
    man = json.loads(mp.read_text())
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(man["model"])) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    for i, v in enumerate(man["voices"]):
        if i % a.nshards != a.shard:
            continue
        d = OUT / v["voice"]
        d.mkdir(exist_ok=True)
        for j, f in enumerate(v["files"]):
            p = d / f"{f['s']}.wav"
            if p.exists():
                continue
            r = rt.synthesize(SamplingRequest(text=f["text"], caption=v["caption"], ref_wav=str(E1 / v["ref"]), no_ref=False,
                                              num_steps=man["steps"], seed=v["seed"] + j), log_fn=None)
            y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
            sf.write(p, y, r.sample_rate)
        print(v["voice"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
