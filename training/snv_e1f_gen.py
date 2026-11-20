"""段 P55(paper_snv rev130 §5 M41): 速く話す学習の声。E1 の学習の声(results/snv_i8_m13/split.json の train・564 声)の t01〜t04 を、Irodori-TTS v4.1 Small の duration_scale(< 1 で速い)で読み直す。
声は E1 と同じ(説明文・t00 を参照・種は E1 の種 + 1000 + 文の番号)。duration_scale は (声, 文) の crc32 で SCALES から決める。出力 = data/snv_e1f/<声>/<文>.wav・台帳 = data/snv_e1f/manifest.json(台詞の文は書かない・E1 の台帳の (声, 文) で引く)。
pilot = 下見の 8 声 × 2 文 × PILOT_SCALES を data/snv_e1f_pilot に作る。measure = 下見の書き起こしの CER と有声 1 秒あたりの c0 の山(数値だけを出す)。
    cd ../../Irodori-TTS && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python ../LightVC/training/snv_e1f_gen.py gen [--shard 0 --nshards 2] | pilot
    cd training && uv run --no-sync python snv_e1f_gen.py measure
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
OUT = ROOT / "data/snv_e1f"
PILOT = ROOT / "data/snv_e1f_pilot"
SCALES = (0.8, 0.65, 0.55)
PILOT_SCALES = (0.55, 0.5)
PILOT_VOICES = ("em480", "em519", "em542", "em566", "ef000", "ef016", "ef032", "ef048")
TEXTS = ("t01", "t02", "t03", "t04")


def e1() -> dict:
    return {v["voice"]: v for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]}


def plan() -> list[dict]:
    m = e1()
    voices = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())["train"]
    out = []
    for v in voices:
        files = {f["t"]: f for f in m[v]["files"]}
        for t in TEXTS:
            sc = SCALES[zlib.crc32(f"e1f|{v}|{t}".encode()) % len(SCALES)]
            out.append({"voice": v, "t": t, "scale": sc, "seed": m[v]["seed"] + 1000 + int(t[1:]), "text": files[t]["text"], "caption": m[v]["caption"]})
    return out


def pilot_plan() -> list[dict]:
    m = e1()
    out = []
    for v in PILOT_VOICES:
        files = {f["t"]: f for f in m[v]["files"]}
        for t in TEXTS[:2]:
            for sc in PILOT_SCALES:
                out.append({"voice": v, "t": t, "scale": sc, "seed": m[v]["seed"] + 1000 + int(t[1:]), "text": files[t]["text"], "caption": m[v]["caption"]})
    return out


def runtime():
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4.1-Small")) / "model.safetensors"
    return InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim", model_precision="fp32", codec_device="cuda",
                                                codec_precision="fp32", codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))


def synth(items: list[dict], dst_of) -> int:
    import soundfile as sf
    rt = runtime()
    from irodori_tts.inference_runtime import SamplingRequest
    n = 0
    for i, it in enumerate(items):
        p = dst_of(it)
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], ref_wav=str(ROOT / "data/snv_e1" / it["voice"] / "t00.wav"), num_steps=40, seed=it["seed"], duration_scale=it["scale"]), log_fn=None)
        tmp = p.with_suffix(".tmp.wav")
        sf.write(tmp, r.audio.detach().float().cpu().numpy().reshape(-1), r.sample_rate)
        tmp.rename(p)
        n += 1
        if (i + 1) % 100 == 0:
            print(i + 1, "/", len(items), flush=True)
    return n


def measure() -> int:
    import librosa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snv_p0_front as P
    import train_snv_p1 as T
    from faster_whisper import WhisperModel
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    rows = []
    for it in pilot_plan():
        p = PILOT / it["voice"] / f"{it['t']}_{it['scale']}.wav"
        x = librosa.load(str(p), sr=P.SR, mono=True)[0]
        n = P.n_frames(len(x))
        f = P.analyze(x[: n * P.H])
        c0, v = T.to_y(f["env"], f["per"], f["f0"])[0] * 10, f["f0"] > 0
        pk, _ = find_peaks(np.where(v, uniform_filter1d(c0, 5), -1e9), prominence=3.0, distance=12)
        hyp = kana("".join(s.text for s in asr.transcribe(to16(x.astype(np.float64), P.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
        rows.append({"voice": it["voice"], "t": it["t"], "scale": it["scale"], "sec": round(len(x) / P.SR, 2), "rate": round(float(len(pk) / (v.sum() / 200.0 + 1e-9)), 2), "cer": round(float(cer(kana(it["text"]), hyp)), 4)})
    summ = {str(sc): {k: round(float(np.median([r[k] for r in rows if r["scale"] == sc])), 3) for k in ("sec", "rate", "cer")} | {"cer_max": round(float(max(r["cer"] for r in rows if r["scale"] == sc)), 3)} for sc in PILOT_SCALES}
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    (ROOT / "results/snv_p55").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/snv_p55/pilot.json").write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "pilot", "measure"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "measure":
        return measure()
    if a.mode == "pilot":
        print("pilot", synth(pilot_plan(), lambda it: PILOT / it["voice"] / f"{it['t']}_{it['scale']}.wav"), flush=True)
        return 0
    items = plan()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.shard == 0:
        (OUT / "manifest.json").write_text(json.dumps({"stage": "paper_snv 段 P55(§5 M41・速く話す学習の声)", "model": "Aratako/Irodori-TTS-v4.1-Small", "steps": 40, "scales": SCALES, "texts": "data/snv_e1/manifest.json の同じ (声, 文)",
                                                       "files": [{k: it[k] for k in ("voice", "t", "scale", "seed")} for it in items]}, ensure_ascii=False, indent=1))
    mine = items[a.shard::a.nshards]
    print("shard", a.shard, "/", a.nshards, "発話", len(mine), flush=True)
    print("done", synth(mine, lambda it: OUT / it["voice"] / f"{it['t']}.wav"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
