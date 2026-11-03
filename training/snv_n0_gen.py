"""N0(synth_nam_vc §5・学習なし): 参照に無い演奏(音素の無い発声)への外挿を測る物差しの較正用の組を Irodori-TTS-v4.1-Small で作る。
声 = A0 と同じ data/pair_tts_v4l の pfNNN(女 24・目標)/ pmNNN(男 12・操作者)の caption + t00 の参照。
発声 = 絵文字の演技指定で 7 種(喘ぎ 🥵・悲鳴 😱・吐息 😮‍💨・息切れ 🌬️・泣き声 😭・笑い 🤭・発話との混ざり)× 2 本。文は間投詞だけ(露骨な語なし)・4 通りから声ごとに回す。
出力 = results/snv_n0/wav/{voice}_{cat}_{k}.wav と items.json。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_n0_gen.py --out ../LightVC/results/snv_n0/wav
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
TEXTS = {
    "moan": ["んっ…あっ…んんっ…🥵", "あっ…ぁあ…んぅ…🥵", "はぁ…んっ、あぁっ…🥵", "ん、んぅ…あっ…🥵"],
    "scream": ["きゃあああっ！😱", "いやああああっ！😱", "きゃっ！いやぁっ！😱", "うわあああっ！😱"],
    "sigh": ["はぁ…😮‍💨", "ふぅ…😮‍💨", "はぁぁ…😮‍💨", "ふぅ…はぁ…😮‍💨"],
    "pant": ["はぁっ、はぁっ、はぁっ…🌬️", "はっ、はっ、はぁっ…🌬️", "ぜぇ、はぁ、はぁ…🌬️", "はぁ、はぁ…はぁっ…🌬️"],
    "cry": ["うっ…ひっく…うぅ…😭", "うぅっ…ぐすっ…😭", "ひっく、ひっく…うわぁぁん😭", "う、うぅ…ひっく…😭"],
    "laugh": ["ふふっ…あははっ🤭", "くすくす…ふふっ🤭", "あはははっ🤭", "ふふ、ふふふっ🤭"],
    "mixed": ["そこ…だめっ…あっ…🥵", "待って…はぁ…はぁ…🌬️もう無理…", "やだ、こわい…きゃあっ😱", "ありがとう…うっ…ひっく😭"],
}
PER_CAT = 2


def items() -> list:
    man = json.loads((PAIR / "manifest.json").read_text())
    voices = [f"pf{k:03d}" for k in range(24)] + [f"pm{k:03d}" for k in range(12)]
    out = []
    for v, name in enumerate(voices):
        for c, (cat, ts) in enumerate(TEXTS.items()):
            for k in range(PER_CAT):
                ti = (v + 2 * k) % len(ts)
                out.append({"voice": name, "sex": "F" if name.startswith("pf") else "M", "cat": cat, "k": k, "text_id": ti, "text": ts[ti],
                            "caption": man["speakers"][name]["caption"], "seed": 20261102 + 1000 * v + 10 * c + k})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="Aratako/Irodori-TTS-v4.1-Small")
    ap.add_argument("--limit", type=int, default=0)
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
    its = items()
    (od / "items.json").write_text(json.dumps(its, ensure_ascii=False, indent=1))
    for it in its[: a.limit or None]:
        p = od / f"{it['voice']}_{it['cat']}_{it['k']}.wav"
        if p.exists():
            continue
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], ref_wav=str(PAIR / it["voice"] / "t00.wav"),
                                          num_steps=a.steps, seed=it["seed"]), log_fn=None)
        y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        sf.write(p, y, r.sample_rate)
        print(it["voice"], it["cat"], it["k"], f"{len(y) / r.sample_rate:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
