"""E1 試作(synth_nam_vc §5 A2 の学習データ): Irodori-TTS-v4.1-Small の VoiceDesign の声のコーパス。
声 = 属性の組から作る caption(女 480・男 120・組は重複なし)。t00 = caption だけ・t01〜t09 = caption + t00 の参照の複製(声を固定)。
文 = female-dataset の書き起こし(.lab)から普通の台詞(prep_male_train_texts.clean と同じ規則 + 露骨な性的表現の語を除く)・評価の 16 文(pair_tts_v4l)は除く。
出力 = data/snv_e1/{voice}/tNN.wav と data/snv_e1/manifest.json(作り方の台帳)。透かしは外さない。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_e1_gen.py --plan        # 台帳だけ
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_e1_gen.py --shard 0 --nshards 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
OUT = ROOT / "data/snv_e1"
PAIR = ROOT / "data/pair_tts_v4l"
N_UTT = 10
NG = re.compile(r"ちゅ|ぢゅ|じゅる|じゅぷ|じゅぽ|れろ|ぴちゃ|くちゅ|はぁ|はあっ|んっ|あっ|あぁ|ん[ー〜～]|[♡♥]|[a-zA-Zａ-ｚＡ-Ｚ]")
ADULT = re.compile(r"ちんぽ|ちんちん|ちんこ|おちん|まんこ|おまんこ|精液|精子|ザーメン|射精|勃起|挿入|中出し|膣|子宮|乳首|おっぱい|クリトリス|セックス|エッチ|えっち|性器|陰茎|亀頭|肉棒|淫|犯し|犯す|舐め|しゃぶ|喘|イク|イッ|イかせ|絶頂|オナ|ペニス|ちくび|お尻の穴|アナル|処女|童貞|孕|種付け|発情|愛液|濡れ|パンツ|下着|裸")
KANJI = re.compile(r"[一-鿿]")
STRIP = str.maketrans("", "", "（）()「」『』【】〈〉《》\"' 　\n\t")
REC = "静かな部屋でマイクに近い、クリアな録音。"
F_ATTR = {
    "age": ["十代後半", "二十歳前後", "二十代前半", "二十代後半", "三十代", "四十代", "五十代"],
    "pitch": ["かなり高めの", "やや高めの", "落ち着いた普通の高さの", "やや低めの", "低めの"],
    "quality": ["甘く可愛らしい", "透き通った", "息混じりの柔らかい", "少しかすれたハスキーな", "はっきりとした明るい", "張りのある力強い",
                "穏やかで優しい", "少し鼻にかかった", "舌足らずで幼い", "落ち着いた大人っぽい", "芯のある凛とした", "ふんわりとした", "低く響く深い", "淡々とした平坦な"],
    "style": ["友人と自然に会話している", "落ち着いて丁寧に話している", "くだけた口調で気楽に話している", "優しく語りかけている", "元気よく話している"],
}
M_ATTR = {
    "age": ["十代後半", "二十代前半", "二十代後半", "三十代", "四十代", "五十代", "六十代"],
    "pitch": ["やや高めの", "落ち着いた普通の高さの", "やや低めの", "低めの", "かなり低めの"],
    "quality": ["はっきりとした明るい", "張りのある力強い", "穏やかで柔らかい", "少しかすれたハスキーな", "低く響く深い", "淡々とした平坦な",
                "少し鼻にかかった", "爽やかな", "渋く落ち着いた", "優しく温かみのある"],
    "style": ["友人と自然に会話している", "落ち着いて丁寧に話している", "くだけた口調で気楽に話している", "淡々と読み上げている", "元気よく話している"],
}


def clean(s: str) -> str | None:
    s = s.translate(STRIP)
    if not 22 <= len(s) <= 70 or NG.search(s) or ADULT.search(s) or len(KANJI.findall(s)) < 3:
        return None
    if sum(c in "…ー〜～っッ・" for c in s) / len(s) >= 0.12:
        return None
    return s


def text_pool() -> list[str]:
    man = json.loads((PAIR / "manifest.json").read_text())
    held = {f["text"] for f in man["speakers"]["pm000"]["files"].values()}
    seen = set()
    for lab in sorted((ROOT / "female-dataset").glob("*/*.lab")):
        s = clean(lab.read_text(encoding="utf-8"))
        if s and s not in seen and s not in held:
            seen.add(s)
    pool = sorted(seen)
    random.Random(20261101).shuffle(pool)
    return pool


def captions(attr: dict, sex: str, n: int, seed: int) -> list[str]:
    combos = list(itertools.product(*attr.values()))
    random.Random(seed).shuffle(combos)
    who = "女性" if sex == "F" else "男性"
    return [f"{a}の日本人{who}。{p}{q}声で、{s}。{REC}" for a, p, q, s in combos[:n]]


def plan() -> dict:
    pool = text_pool()
    voices = [("F", c) for c in captions(F_ATTR, "F", 480, 1)] + [("M", c) for c in captions(M_ATTR, "M", 120, 2)]
    rng = random.Random(3)
    out = []
    for i, (sex, cap) in enumerate(voices):
        name = f"{'ef' if sex == 'F' else 'em'}{i:03d}"
        texts = rng.sample(pool, N_UTT)
        out.append({"voice": name, "sex": sex, "caption": cap, "seed": 20261101 + 1000 * i,
                    "files": [{"t": f"t{j:02d}", "text": texts[j], "ref": None if j == 0 else "t00"} for j in range(N_UTT)]})
    return {"model": "Aratako/Irodori-TTS-v4.1-Small", "steps": 40, "n_text_pool": len(pool), "voices": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
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
    for i, v in enumerate(man["voices"]):
        if i % a.nshards != a.shard:
            continue
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
