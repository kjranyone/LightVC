"""段 P73 step c(results/snv_p73/prereg.yaml): 別の生成器 Qwen3-TTS(Apache-2.0)の、説明文で作った男声。測りの入力(学習に足すかは段の判定の後)。
t00 = Qwen3-TTS-12Hz-1.7B-VoiceDesign が説明文から作る(実在の人の音声を参照にしない)。t01〜t04 = Qwen3-TTS-12Hz-1.7B-Base が t00 を参照に読む(公式の design-then-clone)。
文 = data/snv_p73/qw/manifest.json(snv_diag/p73_sets.py qw_plan が snv_e1_gen.text_pool から作る・この機械の中だけ)。出力 = data/snv_p73/qw/<声>/tNN.wav。
    ../qwen_tts_env/bin/python training/snv_p73_qw_gen.py
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p73/qw"


def main() -> int:
    man = json.loads((OUT / "manifest.json").read_text())
    todo = [v for v in man["voices"] if not (OUT / v["voice"] / "t00.wav").exists()]
    if todo:
        m = Qwen3TTSModel.from_pretrained(man["design"], device_map="cuda:0", dtype=torch.bfloat16)
        for v in todo:
            torch.manual_seed(v["seed"])
            w, sr = m.generate_voice_design(text=v["files"][0]["text"], language="Japanese", instruct=v["instruct"])
            (OUT / v["voice"]).mkdir(exist_ok=True)
            sf.write(OUT / v["voice"] / "t00.wav", w[0], sr)
            print("design", v["voice"], flush=True)
        del m
        torch.cuda.empty_cache()
    m = Qwen3TTSModel.from_pretrained(man["clone"], device_map="cuda:0", dtype=torch.bfloat16)
    for v in man["voices"]:
        d = OUT / v["voice"]
        if all((d / f"{f['t']}.wav").exists() for f in v["files"]):
            continue
        pr = m.create_voice_clone_prompt(ref_audio=str(d / "t00.wav"), ref_text=v["files"][0]["text"])
        for j, f in enumerate(v["files"][1:], 1):
            torch.manual_seed(v["seed"] + j)
            w, sr = m.generate_voice_clone(text=f["text"], language="Japanese", voice_clone_prompt=pr)
            sf.write(d / f"{f['t']}.wav", w[0], sr)
        print("clone", v["voice"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
