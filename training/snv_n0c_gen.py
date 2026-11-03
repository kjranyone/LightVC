"""N0c(synth_nam_vc §5・学習なし): 発声の種類の物差しを声に依らない形で作るための組。参照なし(説明文だけ)の Irodori-TTS-v4.1-Small で、
E1 の 48 声(女 24・男 24)の説明文を、声質の後ろを種類ごとの動作 + 近接の録音に替えて生成する(N0b 事前検査で参照なしだけが種類を分けた・0.71 対 参照つき 0.41〜0.54)。
参照なしなので同じ説明文でも本数ごとに声は変わる = 本人らしさには使わない。種類 = 喘ぎ・悲鳴・吐息・息切れ・泣き声・笑い の 6 種(混ざりは他の種類を含むので除く)× 4 文 × 3 本。
出力 = results/snv_n0c/wav/{voice}_{cat}_{t}_{k}.wav と items.json。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_n0c_gen.py --out ../LightVC/results/snv_n0c/wav
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snv_n0_gen import TEXTS

CATS = ("moan", "scream", "sigh", "pant", "cry", "laugh")
ACT = {"moan": "吐息混じりに甘く喘いでいる", "scream": "恐怖で甲高く悲鳴を上げている", "sigh": "深く長いため息をついている",
       "pant": "走った直後のように激しく息を切らしている", "cry": "しゃくり上げながら泣いている", "laugh": "くすくすと楽しそうに笑っている"}
REC = "ASMR のようにマイクのすぐ近くで、息の音まではっきり聞こえる録音。"
PER_TEXT = 1


def caption(cap: str, cat: str) -> str:
    s = cap.split("。")
    return s[0] + "。" + s[1].split("で、")[0] + "で、" + ACT[cat] + "。" + REC


def items() -> list:
    man = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    rng = random.Random(20261102)
    vs = rng.sample([v for v in man["voices"] if v["sex"] == "F"], 24) + rng.sample([v for v in man["voices"] if v["sex"] == "M"], 24)
    out = []
    for vi, v in enumerate(vs):
        for c, cat in enumerate(CATS):
            for t, text in enumerate(TEXTS[cat]):
                for k in range(PER_TEXT):
                    out.append({"voice": v["voice"], "sex": v["sex"], "cat": cat, "t": t, "k": k, "text": text,
                                "caption": caption(v["caption"], cat), "seed": 31000000 + 1000 * vi + 100 * c + 10 * t + k})
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
        p = od / f"{it['voice']}_{it['cat']}_{it['t']}_{it['k']}.wav"
        if p.exists():
            continue
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], no_ref=True, num_steps=a.steps, seed=it["seed"]), log_fn=None)
        y = r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32)
        sf.write(p, y, r.sample_rate)
        print(it["voice"], it["cat"], it["t"], it["k"], f"{len(y) / r.sample_rate:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
