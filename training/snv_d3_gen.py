"""段 D3(paper_snv rev27 §5 M1 rev27・§6 段 D3・学習なし): 操作者の母集団への族の覆いの下見の声。Irodori-TTS-v4.1-Small の VoiceDesign の男声を E1 と同じ作法で作る(t00 = 説明文だけ・t01 と t02 = 説明文 + t00 の参照・40 step)。
(v) dvNNN = 操作者の母集団へ向けた声の属性の説明文 48 声(録音の文は E1 と同じ)・(r) drNNN = 同じ 48 の属性と種に録音の文 4 通りを回した 48 声。文 = E1 と段 D で使っていない台本の文。
出力 = data/snv_d3/{voice}/tNN.wav と data/snv_d3/manifest.json(作り方の台帳)。透かしは外さない。
    cd training && uv run --no-sync python snv_d3_gen.py --plan                       # 台帳だけ
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_d3_gen.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
OUT = ROOT / "data/snv_d3"
N_VOICE = 48
N_UTT = 3
REC_E1 = "静かな部屋でマイクに近い、クリアな録音。"
REC_VAR = ["自宅の部屋で一般的なマイクを使って録音した、少し部屋の響きがある音。",
           "マイクから少し離れて録音した、自然な部屋の音。",
           "放送のナレーション収録のような、低音のしっかりした録音。",
           "ヘッドセットのマイクで録音した、少しこもった音。"]
OP_ATTR = {
    "age": ["二十代後半", "三十代", "四十代", "五十代", "六十代"],
    "pitch": ["落ち着いた普通の高さの", "やや低めの", "低めの", "かなり低めの"],
    "quality": ["太く響く", "こもった", "少ししゃがれた", "息の混じった", "張りのある", "渋く落ち着いた", "柔らかく穏やかな", "硬く引き締まった"],
    "style": ["原稿をナレーションのように読み上げている", "物語を朗読している", "マイクに向かって落ち着いて話している", "友人と自然に会話している"],
}


def plan() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snv_e1_gen as G
    used = {f["text"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"] for f in v["files"]}
    dman = ROOT / "data/snv_d/manifest.json"
    if dman.exists():
        used |= {f["text"] for v in json.loads(dman.read_text())["voices"] for f in v["files"]}
    pool = [t for t in G.text_pool() if t not in used]
    combos = list(itertools.product(*OP_ATTR.values()))
    random.Random(20261109).shuffle(combos)
    rng = random.Random(4)
    out = []
    for i, (a, p, q, st) in enumerate(combos[:N_VOICE]):
        texts = rng.sample(pool, N_UTT)
        base = f"{a}の日本人男性。{p}{q}声で、{st}。"
        for arm, rec in (("dv", REC_E1), ("dr", REC_VAR[i % len(REC_VAR)])):
            out.append({"voice": f"{arm}{i:03d}", "arm": arm, "sex": "M", "caption": base + rec, "seed": 20261109 + 1000 * i,
                        "files": [{"t": f"t{j:02d}", "text": texts[j], "ref": None if j == 0 else "t00"} for j in range(N_UTT)]})
    return {"model": "Aratako/Irodori-TTS-v4.1-Small", "steps": 40, "n_text_pool": len(pool), "voices": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mp = OUT / "manifest.json"
    if a.plan or not mp.exists():
        man = plan()
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print("plan", len(man["voices"]), "voices", "text pool", man["n_text_pool"], flush=True)
        if a.plan:
            return 0
    man = json.loads(mp.read_text())
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download(man["model"])) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    for v in man["voices"]:
        d = OUT / v["voice"]
        d.mkdir(exist_ok=True)
        for j, f in enumerate(v["files"]):
            p = d / f"{f['t']}.wav"
            if p.exists():
                continue
            ref = None if f["ref"] is None else str(d / f"{f['ref']}.wav")
            r = rt.synthesize(SamplingRequest(text=f["text"], caption=v["caption"], ref_wav=ref, no_ref=ref is None,
                                              num_steps=man["steps"], seed=v["seed"] + j), log_fn=None)
            y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
            sf.write(p, y, r.sample_rate)
        print(v["voice"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
