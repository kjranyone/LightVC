"""男声の元の声のコーパス(変換の入力の試験用)を Irodori-TTS-v4-Large で作る(オーナー指示 2026-10-09: v3 の male_tts_corpus を作り直す)。
話者 = 日本人男性の声の設計文(年齢・声の高さ・声質・話し方の組み合わせ)。話者ごとに設計文だけで 1 文目(錨)を作り、残りは錨を参照音声 + 同じ設計文で読ませる(同じ人の声に揃える)。
文 = 自然な会話文 40 文(五十音の読み上げなし)。出力 = <out>/<spk>/tNN.wav(48kHz)と manifest.json(設計文・種・文・長さ)。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/generate_male_corpus_v4l.py --out ../LightVC/data/male_tts_v4l --n_spk 48
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

IRODORI = Path(__file__).resolve().parent.parent.parent / "Irodori-TTS"
sys.path.insert(0, str(IRODORI))

TEXTS = [
    "今日は朝から雨が降っていて、少し肌寒いですね。",
    "駅前に新しいカフェができたので、今度一緒に行きませんか。",
    "昨日の会議では、来月の予定について話し合いました。",
    "すみません、この近くにコンビニはありますか。",
    "週末は家でゆっくり映画を観て過ごしました。",
    "このアプリ、思ったより使いやすくて便利だよ。",
    "電車が遅れているので、少し到着が遅くなりそうです。",
    "最近ランニングを始めたんだけど、なかなか続かなくてさ。",
    "その件については、明日までに確認しておきます。",
    "子供の頃は、夏休みになると毎日川で遊んでいました。",
    "えっ、本当に？それはちょっと驚いたな。",
    "ありがとう、すごく助かったよ。",
    "ちょっと待って、今そっちに行くから。",
    "うーん、どっちがいいか迷うなあ。",
    "なるほど、そういうことだったんですね。",
    "お疲れさまです。今日も一日よく頑張りましたね。",
    "このラーメン屋、スープが濃厚でおすすめなんですよ。",
    "明日の天気予報では、午後から晴れるそうです。",
    "ごめん、さっきの話もう一回だけ聞いてもいい？",
    "新しいゲームを買ったので、夜にみんなで遊ぼう。",
    "資料は共有フォルダに入れておいたので、確認してください。",
    "久しぶりに実家に帰ったら、母がすごく喜んでいました。",
    "あれ、鍵どこに置いたっけ。さっきまでここにあったのに。",
    "そろそろお昼にしようか。何か食べたいものある？",
    "この問題は、もう少し時間をかけて考えたほうがいいと思います。",
    "図書館は午後八時まで開いているので、まだ間に合いますよ。",
    "いやあ、今日は本当に暑いね。冷たいものが飲みたいよ。",
    "来週の土曜日、みんなでバーベキューをする予定なんだ。",
    "申し訳ありませんが、担当者はただいま席を外しております。",
    "このまままっすぐ進んで、二つ目の信号を右に曲がってください。",
    "趣味は写真で、休みの日はよく海まで撮りに行きます。",
    "はは、それは面白いね。もっと詳しく教えてよ。",
    "体調はどう？無理しないで、ゆっくり休んでね。",
    "部屋の掃除をしていたら、懐かしい手紙が出てきました。",
    "猫が膝の上で寝ちゃって、動けなくなっちゃった。",
    "次の駅で降りて、そこから歩いて十分くらいです。",
    "この曲を聴くと、学生時代のことを思い出すんだよね。",
    "準備ができたら声をかけてください。すぐに始めます。",
    "よし、今日はここまでにして、続きはまた明日やろう。",
    "最後まで聞いてくれて、本当にありがとうございました。",
]

AGES = ["二十代前半", "二十代後半", "三十代", "四十代", "五十代"]
PITCH = ["低めの", "落ち着いた普通の高さの", "やや高めの"]
TIMBRE = ["穏やかで柔らかい声", "はっきりとした明るい声", "少しかすれたハスキーな声", "張りのある力強い声", "少し鼻にかかった声", "低く響く深い声", "淡々とした平坦な声", "優しく温かみのある声"]
STYLE = ["友人と自然に会話している。", "落ち着いて丁寧に話している。", "くだけた口調で気楽に話している。"]
ANCHOR = 10


def captions(n: int, seed: int) -> list[str]:
    combos = list(itertools.product(AGES, PITCH, TIMBRE, STYLE))
    random.Random(seed).shuffle(combos)
    return [f"{a}の日本人男性。{p}{t}で、{s}静かな部屋でマイクに近い、クリアな録音。" for a, p, t, s in combos[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_spk", type=int, default=48)
    ap.add_argument("--n_text", type=int, default=len(TEXTS))
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261009)
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision=a.precision, codec_device="cuda", codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    mpath = od / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {"model": "Aratako/Irodori-TTS-v4-Large", "precision": a.precision, "steps": a.steps, "texts": TEXTS, "speakers": {}}
    caps = captions(a.n_spk, a.seed)
    order = [ANCHOR] + [i for i in range(a.n_text) if i != ANCHOR]
    for si, cap in enumerate(caps):
        spk = f"m{si:03d}"
        sd = od / spk; sd.mkdir(exist_ok=True)
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
        print(spk, cap, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
