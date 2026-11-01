"""学習用の対の TTS コーパス: 同じ文を男声と女声が読む(Irodori-TTS-v4-Large・作り方は generate_male_corpus_v4l.py と同じ = 設計文の錨 + 錨を参照)。
評価の文(male_tts_v4l の 40 文)・評価の声(m000〜m047・f000〜f047)とは別の文 16 本・別の種の設計文。話者 = pm000〜(男声)・pf000〜(女声)・同じ番号は同じ年齢・高さ・声質・話し方の語。
入力の側のデータ(操作者)。目標の側には使わない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_pair_v4l.py --out ../LightVC/data/pair_tts_v4l --n_spk 60
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
sys.path.insert(0, str(Path(__file__).parent))
from generate_male_corpus_v4l import AGES, PITCH, STYLE, TIMBRE

TEXTS = [
    "明日の天気が良ければ、海の近くまでドライブしようと思ってるんだ。",
    "この前借りた本、すごく面白かったから最後まで一気に読んじゃった。",
    "すみません、このパンはまだ温かいうちに食べたほうがいいですか。",
    "来週の発表の資料は、金曜日までにまとめて送りますね。",
    "あれ、鍵をどこに置いたか全然思い出せないんだけど。",
    "久しぶりに実家に帰ったら、庭の木がずいぶん大きくなっていました。",
    "そんなに急がなくても大丈夫だよ、まだ時間はたっぷりあるから。",
    "お店の人がとても親切で、道に迷った私を駅まで案内してくれました。",
    "今夜は冷えるらしいので、暖かい格好をして出かけてください。",
    "ねえ、さっきの映画の最後の場面、どういう意味だったと思う。",
    "毎朝コーヒーを淹れるのが、一日の中でいちばん好きな時間です。",
    "このあいだの旅行の写真、あとでまとめて送るね。",
    "申し訳ありませんが、担当の者がただいま席を外しております。",
    "雨が上がったら、少し散歩してから晩ごはんの買い物に行こうか。",
    "子どもたちが公園で遊ぶ声が、窓の外から聞こえてきます。",
    "思っていたより簡単だったので、次はもっと難しいのに挑戦したい。",
]
ANCHOR = 0


def captions(n: int, seed: int) -> list[tuple]:
    combos = list(itertools.product(AGES, PITCH, TIMBRE, STYLE))
    random.Random(seed).shuffle(combos)
    return combos[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=60)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261020)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    mpath = od / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": "Aratako/Irodori-TTS-v4-Large", "precision": "fp32", "steps": a.steps, "texts": TEXTS, "speakers": {}}
    order = [ANCHOR] + [i for i in range(len(TEXTS)) if i != ANCHOR]
    for si, (ag, pt, tb, sty) in enumerate(captions(a.n_spk, a.seed)):
        for g, word in (("pm", "男性"), ("pf", "女性")):
            spk = f"{g}{si:03d}"
            cap = f"{ag}の日本人{word}。{pt}{tb}で、{sty}静かな部屋でマイクに近い、クリアな録音。"
            sd = od / spk
            sd.mkdir(exist_ok=True)
            ent = man["speakers"].setdefault(spk, {"caption": cap, "seed": a.seed + si, "files": {}})
            t0 = time.time()
            for ti in order:
                wp = sd / f"t{ti:02d}.wav"
                if wp.exists():
                    continue
                anchor = sd / f"t{ANCHOR:02d}.wav"
                req = dict(text=TEXTS[ti], caption=cap, num_steps=a.steps, seed=a.seed + si * 100 + ti)
                req.update(no_ref=True) if ti == ANCHOR else req.update(ref_wav=str(anchor))
                r = rt.synthesize(SamplingRequest(**req), log_fn=None)
                y = r.audio.detach().float().cpu().numpy().reshape(-1)
                sf.write(wp, y.astype(np.float32), r.sample_rate)
                ent["files"][f"t{ti:02d}"] = {"text": TEXTS[ti], "sr": r.sample_rate, "sec": round(len(y) / r.sample_rate, 3), "seed": r.used_seed, "ref": None if ti == ANCHOR else anchor.name}
            mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
            print(spk, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
