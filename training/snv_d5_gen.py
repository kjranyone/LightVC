"""段 D5(paper_snv rev41 §5 M1 rev41 (a)・§6 段 D5・学習なしの下見): 生成器の参照の型。説明文だけで作った声の t00 を信号処理で動かした音を、Irodori-TTS-v4.1-Small の話者の参照に渡す。実在の人の音声は使わない。
元 = E1 の学習の男声 24 声(t00 の高さの中央値の順に等間隔)。型: g0 = 加工しない t00(対照)/ w = WORLD で高さを目標の中央値 {80, 95, 110, 125} Hz へ・周波数の軸を 1/α_v(α_v ∈ {1.0, 1.12, 1.25})に /
r = 再標本化だけで速度を r ∈ {0.9, 0.8, 0.7} に。説明文: c1 = 元の E1 の説明文・c2 = 声質を書かない文。各声 3 文(E1・段 D・段 D3 で使っていない文・元の声ごとに全ての型で同じ文)。
出力 = data/snv_d5/refs/{元}_{型}.wav・data/snv_d5/{声}/tNN.wav・data/snv_d5/manifest.json(作り方の台帳)。透かしは外さない。
    cd training && uv run --no-sync python snv_d5_gen.py --refs                          # 参照の加工と台帳(pyworld)
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_d5_gen.py    # 生成
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
OUT = ROOT / "data/snv_d5"
N_BASE = 24
N_UTT = 3
F0_T = (80.0, 95.0, 110.0, 125.0)
ALPHA_V = (1.0, 1.12, 1.25)
SLOW = (0.9, 0.8, 0.7)
REC = "静かな部屋でマイクに近い、クリアな録音。"
CAP2 = "落ち着いて話している。" + REC


def refs() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import librosa
    import snv_d5_world as W
    import snv_e1_gen as G
    e1 = {v["voice"]: v for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]}
    used = {f["text"] for v in e1.values() for f in v["files"]}
    for m in ("data/snv_d/manifest.json", "data/snv_d3/manifest.json"):
        if (ROOT / m).exists():
            used |= {f["text"] for v in json.loads((ROOT / m).read_text())["voices"] for f in v["files"]}
    pool = [t for t in G.text_pool() if t not in used]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    males = [v for v in split["train"] if v.startswith("em")]
    med = {}
    for v in males:
        f0 = np.load(ROOT / f"data/snv_p_feat/e1/{v}/t00.npz")["f0"]
        med[v] = float(np.median(f0[f0 > 0]))
    order = sorted(males, key=lambda v: med[v])
    base = [order[int(round(i * (len(order) - 1) / (N_BASE - 1)))] for i in range(N_BASE)]
    (OUT / "refs").mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261110)
    voices = []
    for i, b in enumerate(base):
        x, sr = librosa.load(str(ROOT / f"data/snv_e1/{b}/t00.wav"), sr=None, mono=True)
        m0 = W.f0_median(x, sr)
        cell = i % 12
        ft, av, r = F0_T[cell % 4], ALPHA_V[cell // 4], SLOW[i % 3]
        made = {"g0": (x, {}), "w": (W.shift(x, sr, ft / m0, av), {"f0_target": ft, "alpha_v": av}), "r": (W.slow(x, r), {"slow": r, "f0_target": m0 * r, "alpha_v": 1 / r})}
        texts = rng.sample(pool, N_UTT)
        for typ, (y, par) in made.items():
            rp = OUT / "refs" / f"{b}_{typ}.wav"
            sf.write(rp, y, sr)
            for cap in (("c1",) if typ == "g0" else ("c1", "c2")):
                voices.append({"voice": f"{b}_{typ}_{cap}", "base": b, "type": typ, "cap": cap, "caption": e1[b]["caption"] if cap == "c1" else CAP2, "ref": str(rp.relative_to(ROOT)),
                               "base_f0": round(m0, 1), "ref_f0": round(W.f0_median(y, sr), 1), "seed": e1[b]["seed"] + 500, **par,
                               "files": [{"t": f"t{j:02d}", "text": texts[j]} for j in range(N_UTT)]})
        print(b, round(m0, 1), "w", ft, av, "r", r, flush=True)
    return {"model": "Aratako/Irodori-TTS-v4.1-Small", "steps": 40, "n_text_pool": len(pool), "voices": voices}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg_speaker", type=float, default=None)
    ap.add_argument("--out", default="data/snv_d5")
    a = ap.parse_args()
    mp = OUT / "manifest.json"
    if a.refs:
        man = refs()
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print("refs", len(man["voices"]), "voices", flush=True)
        return 0
    man = json.loads(mp.read_text())
    out = ROOT / a.out
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(man["model"])) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    kw = {} if a.cfg_speaker is None else {"cfg_scale_speaker": a.cfg_speaker}
    for v in man["voices"]:
        d = out / v["voice"]
        d.mkdir(parents=True, exist_ok=True)
        for j, f in enumerate(v["files"]):
            p = d / f"{f['t']}.wav"
            if p.exists():
                continue
            r = rt.synthesize(SamplingRequest(text=f["text"], caption=v["caption"], ref_wav=str(ROOT / v["ref"]), no_ref=False, num_steps=man["steps"], seed=v["seed"] + j, **kw), log_fn=None)
            sf.write(p, r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate)
        print(v["voice"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
