"""腕 64 の学習用の日本語の男声のコーパス(オーナー 10-30『irodori v4 large でいいよ』): Irodori-TTS-v4-Large(bf16・40 step)。
話者 mtNNN = 設計文(年齢・高さ・声質・話し方・録音)。1 文目 = 設計文だけ(錨・候補 1 本)・残り = 錨を参照 + 同じ設計文・候補 2 本(同じ文の別の読み)。
文 = data/male_tts_v4l_train/texts.json(prep_male_train_texts.py・学習の話者の書き起こし)を話者ごとに重ならず割り当てる。評価の声(m・g・pm・f・pf)とは別の種。
出力 = <out>/<spk>/uNN_cJ.wav と manifest.json。透かし(SilentCipher)は Irodori の既定のまま。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_male_train_v4l.py --out ../LightVC/data/male_tts_v4l_train --n_spk 475
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
sys.path.insert(0, str(ROOT.parent / "Irodori-TTS"))

AGES = ["二十代前半", "二十代後半", "三十代", "四十代", "五十代"]
PITCH = ["かなり低めの", "低めの", "落ち着いた普通の高さの", "やや高めの"]
TIMBRE = ["穏やかで柔らかい声", "はっきりとした明るい声", "少しかすれたハスキーな声", "張りのある力強い声", "少し鼻にかかった声", "低く響く深い声", "淡々とした平坦な声",
          "優しく温かみのある声", "少しこもった声", "息の混じった柔らかい声", "若々しく軽い声", "渋くて落ち着いた声"]
STYLE = ["友人と自然に会話している。", "落ち着いて丁寧に話している。", "くだけた口調で気楽に話している。", "独り言のようにぼそぼそと話している。", "感情を込めて台詞を読んでいる。",
         "少し照れながら話している。"]
REC = ["静かな部屋でマイクに近い、クリアな録音。", "静かな部屋でマイクに近い、クリアな録音。", "普通の部屋で、マイクから少し離れた録音。", "少し響く部屋での録音。"]
N_TXT = 9


def captions(n: int, seed: int) -> list[str]:
    combos = list(itertools.product(AGES, PITCH, TIMBRE, STYLE, range(len(REC))))
    random.Random(seed).shuffle(combos)
    return [f"{a}の日本人男性。{p}{t}で、{s}{REC[r]}" for a, p, t, s, r in combos[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=475)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261030)
    a = ap.parse_args()
    od = Path(a.out)
    texts = [t["text"] for t in json.loads((od / "texts.json").read_text())["texts"]]
    assert len(texts) >= a.n_spk * N_TXT, (len(texts), a.n_spk * N_TXT)
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="bf16", codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    mpath = od / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": "Aratako/Irodori-TTS-v4-Large", "precision": "bf16", "steps": a.steps, "speakers": {}}
    t_all = time.time()
    for si, cap in enumerate(captions(a.n_spk, a.seed)):
        spk = f"mt{si:03d}"
        sd = od / spk
        sd.mkdir(exist_ok=True)
        ent = man["speakers"].setdefault(spk, {"caption": cap, "files": {}})
        t0 = time.time()
        anchor = sd / "u00_c0.wav"
        for ti in range(N_TXT):
            text = texts[si * N_TXT + ti]
            nc = 1 if ti == 0 else 2
            if all((sd / f"u{ti:02d}_c{j}.wav").exists() for j in range(nc)):
                continue
            req = dict(text=text, caption=cap, num_steps=a.steps, seed=a.seed + si * 100 + ti, num_candidates=nc, decode_mode="batch")
            req.update(no_ref=True) if ti == 0 else req.update(ref_wav=str(anchor))
            r = rt.synthesize(SamplingRequest(**req), log_fn=None)
            for j, y in enumerate(r.audios[:nc]):
                y = y.detach().float().cpu().numpy().reshape(-1)
                sf.write(sd / f"u{ti:02d}_c{j}.wav", y.astype(np.float32), r.sample_rate)
                ent["files"][f"u{ti:02d}_c{j}"] = {"text": text, "sr": r.sample_rate, "sec": round(len(y) / r.sample_rate, 3), "seed": r.used_seed,
                                                    "ref": None if ti == 0 else anchor.name}
        mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print(spk, f"{time.time() - t0:.0f}s", f"total {(time.time() - t_all) / 60:.1f} min", cap, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
