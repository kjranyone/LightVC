"""D549: 男声 → 女声の最終の関門(オーナー判断 10-23: 男声の録音は Irodori-TTS v4 でよい)の未使用の男声の操作者を合成する。録音のお願い(data/real_male_eval/README)と同じ形
= <out>/<spk>/t00.wav〜t39.wav(generate_male_corpus_v4l.TEXTS = README の 40 文)+ free.wav(別の 10 文の独り語りを 0.4 s の無音でつなぐ・約 35 s・操作者の参照)。
声の設計文 = generate_male_corpus_v4l.captions の同じ並び(種 20261009)の 49 番目以降(m000〜m047 = 開発の組・学習の交差の標本と重ならない)・話者 = g000〜・種は 20261023 系。
錨 = t10 を設計文だけで作り、残りは錨を参照に同じ設計文で読ませる(generate_male_corpus_v4l と同じ)。学習・開発・調整に使わない(関門で一度だけ)。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/d549_male_eval_tts.py --out ../LightVC/data/male_tts_v4_eval --n_spk 24
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
from generate_male_corpus_v4l import ANCHOR, TEXTS, captions

FREE = [
    "最近は仕事が忙しくて、なかなか自分の時間が取れないんですよ。",
    "それでも週末だけは、できるだけ好きなことをするようにしています。",
    "この前は久しぶりに友達と山に登って、頂上でお弁当を食べました。",
    "天気が良くて景色もきれいで、本当に気持ちよかったです。",
    "帰りに温泉に寄ったんですけど、それがまた最高でしたね。",
    "来月は少し遠出をして、海の近くの町に行ってみようと思っています。",
    "美味しい魚料理のお店があるらしいので、今から楽しみなんです。",
    "普段はあまり料理をしないんですが、最近はカレーだけ練習しています。",
    "スパイスを自分で混ぜるのが意外と面白くて、はまってしまいました。",
    "今度うまくできたら、みんなにも食べてもらいたいですね。",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=24)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--precision", default="fp32", help="GPU が足りないとき bf16(10-23: 他のプロセスが 5.6 GB を使い fp32 の Large は参照つきで収まらない)")
    ap.add_argument("--codec_device", default="cuda", help="GPU を他のプロセスと分けるとき cpu(モデルは開発の組と同じ fp32 のまま)")
    ap.add_argument("--offset", type=int, default=0, help="2 組目以降の関門(10-24 gate_mf_v3): 話者 = g<offset>〜・設計文 = 並びの 49 + offset 番目以降で --exclude の書き出しの設計文と重ならないもの")
    ap.add_argument("--exclude", nargs="*", default=[], help="設計文を重ねない書き出しの manifest.json(学習の操作者・開発・前の関門)")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision=a.precision, codec_device=a.codec_device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    mpath = od / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": "Aratako/Irodori-TTS-v4-Large", "steps": a.steps, "texts": TEXTS, "free": FREE, "speakers": {},
                                                                "note": "男声 → 女声の最終の関門専用(学習・開発・調整に使わない)", "codec_device": a.codec_device, "precision": a.precision}
    used = {v.get("caption") for p in a.exclude for v in json.loads(Path(p).read_text()).get("speakers", {}).values()}
    caps = [c for c in captions(360, 20261009)[48 + a.offset:] if c not in used][:a.n_spk] if a.offset else captions(48 + a.n_spk, 20261009)[48:]
    assert len(caps) == a.n_spk
    order = [ANCHOR] + [i for i in range(len(TEXTS)) if i != ANCHOR]
    for si, cap in enumerate(caps, a.offset):
        spk = f"g{si:03d}"
        sd = od / spk
        sd.mkdir(exist_ok=True)
        ent = man["speakers"].setdefault(spk, {"caption": cap, "seed": a.seed + si, "files": {}})
        t0 = time.time()
        anchor = sd / f"t{ANCHOR:02d}.wav"

        def synth(text: str, seed: int, ref: bool) -> tuple:
            req = dict(text=text, caption=cap, num_steps=a.steps, seed=seed)
            req.update(ref_wav=str(anchor)) if ref else req.update(no_ref=True)
            r = rt.synthesize(SamplingRequest(**req), log_fn=None)
            return r.audio.detach().float().cpu().numpy().reshape(-1), r.sample_rate, r.used_seed

        for ti in order:
            wp = sd / f"t{ti:02d}.wav"
            if wp.exists():
                continue
            y, sr, us = synth(TEXTS[ti], a.seed * 7 + si * 100 + ti, ti != ANCHOR)
            sf.write(wp, y.astype(np.float32), sr)
            ent["files"][f"t{ti:02d}"] = {"text": TEXTS[ti], "sr": sr, "sec": round(len(y) / sr, 3), "seed": us}
            mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        fp = sd / "free.wav"
        if not fp.exists():
            parts, sr0 = [], None
            for fi, text in enumerate(FREE):
                y, sr, _ = synth(text, a.seed * 11 + si * 100 + fi, True)
                sr0 = sr0 or sr
                assert sr == sr0
                parts += [y, np.zeros(int(0.4 * sr), np.float32)]
            yy = np.concatenate(parts).astype(np.float32)
            sf.write(fp, yy, sr0)
            ent["files"]["free"] = {"texts": FREE, "sr": sr0, "sec": round(len(yy) / sr0, 3)}
            mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
        print(spk, cap, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
