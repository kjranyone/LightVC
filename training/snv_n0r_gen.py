"""N0r(synth_nam_vc §5・学習なし): N0 を Irodori の演技の作法で作り直す(オーナー 2026-11-02「その作法で N0 作り直して」)。
作法(results/snv_n0/verdict.yaml の n0d_recipe・男 6 声の事前検査で最良の r4): 参照 = t00 の 1 本・説明文 = 動作と状況だけ(声質を書かない)・
絵文字 = 純粋な種類は前に 2 つ + 後ろに 1 つ、混ざりは文の中の絵文字を 2 つずつに・本文 CFG 5.5・話者 CFG 4.0・説明文 CFG 3.0・40 step。
声・文・乱数は N0 と同じ(snv_n0_gen.items)。出力 = results/snv_n0r/wav/{voice}_{cat}_{k}.wav と items.json。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_n0r_gen.py --out ../LightVC/results/snv_n0r/wav
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snv_n0_gen import items

EMO = {"moan": "🥵", "scream": "😱", "sigh": "😮‍💨", "pant": "🌬️", "cry": "😭", "laugh": "🤭"}
ACT = {"moan": "吐息混じりに甘く喘いでいる。", "scream": "恐怖で甲高く悲鳴を上げている。", "sigh": "深く長いため息をついている。",
       "pant": "走った直後のように激しく息を切らしている。", "cry": "しゃくり上げながら泣いている。", "laugh": "くすくすと楽しそうに笑っている。",
       "mixed": "感情があふれて、吐息や泣き声が混ざりながら話している。"}


def recipe_text(text: str, cat: str) -> str:
    if cat in EMO:
        e = EMO[cat]
        return e * 2 + text.replace(e, "") + e
    for e in sorted(EMO.values(), key=len, reverse=True):
        text = text.replace(e, e * 2)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
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
    its = [it | {"text": recipe_text(it["text"], it["cat"]), "caption": ACT[it["cat"]]} for it in items()]
    (od / "items.json").write_text(json.dumps(its, ensure_ascii=False, indent=1))
    for it in its[: a.limit or None]:
        p = od / f"{it['voice']}_{it['cat']}_{it['k']}.wav"
        if p.exists():
            continue
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], ref_wav=str(PAIR / it["voice"] / "t00.wav"), num_steps=40, seed=it["seed"],
                                          cfg_scale_text=5.5, cfg_scale_speaker=4.0, cfg_scale_caption=3.0), log_fn=None)
        y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        sf.write(p, y, r.sample_rate)
        print(it["voice"], it["cat"], it["k"], f"{len(y) / r.sample_rate:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
