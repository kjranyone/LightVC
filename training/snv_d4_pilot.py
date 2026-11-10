"""段 D4 の下見(paper_snv rev31 §9 O12・学習なし): 試験の本物の声は学習の声より速く(有声 1 秒あたりの c0 の山 外郎売 7.8〜8.9・namikawa_3〜5 10.6〜12.1 対 Irodori 6.9)、JVNV を速めると声を跨ぐ変換だけが落ちる(results/snv_i15/rate_probe.json)。
Irodori-TTS v4.1 Small の duration_scale(生成の長さの倍率・< 1 で速い)で、学習の声が速く話す音を作れるかを測る: E1 の声 8 声(男 4・女 4・各声の t00 を参照)× 段 D の文 3 本 × duration_scale 1.0 / 0.8 / 0.65。
量 = 有声 1 秒あたりの c0 の山(snv_diag/i10_real_decomp の mode_rate と同じ: 5 フレームの平均・prominence 3・間隔 12 フレーム)・large-v3 の書き起こしの CER(文に対して)。出力 = data/snv_d4_pilot/{声}/{文}_{倍率}.wav と results/snv_d4/pilot.json。
    cd ../../Irodori-TTS && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python ../LightVC/training/snv_d4_pilot.py gen
    cd training && uv run --no-sync python snv_d4_pilot.py measure
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
OUT = ROOT / "data/snv_d4_pilot"
VOICES = ("em480", "em519", "em542", "em566", "ef000", "ef016", "ef032", "ef048")
SCALES = (1.0, 0.8, 0.65)


def plan() -> list[dict]:
    d = json.loads((ROOT / "data/snv_d/manifest.json").read_text())
    e1 = {v["voice"]: v for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"]}
    by = {v["voice"]: v for v in d["voices"]}
    out = []
    for v in VOICES:
        for f in by[v]["files"][:3]:
            for sc in SCALES:
                out.append({"voice": v, "s": f["s"], "text": f["text"], "content": f["content"], "scale": sc, "caption": e1[v]["caption"], "seed": e1[v]["seed"] + 77})
    return out


def gen() -> int:
    import soundfile as sf
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4.1-Small")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim", model_precision="fp32", codec_device="cuda",
                                              codec_precision="fp32", codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    for it in plan():
        p = OUT / it["voice"] / f"{it['s']}_{it['scale']}.wav"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it["caption"], ref_wav=str(ROOT / "data/snv_e1" / it["voice"] / "t00.wav"), num_steps=40, seed=it["seed"], duration_scale=it["scale"]), log_fn=None)
        sf.write(p, r.audio.detach().float().cpu().numpy().reshape(-1), r.sample_rate)
    print("gen done", flush=True)
    return 0


def measure() -> int:
    import librosa
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import snv_p0_front as P
    import train_snv_p1 as T
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    rows = []
    for it in plan():
        p = OUT / it["voice"] / f"{it['s']}_{it['scale']}.wav"
        x = librosa.load(str(p), sr=P.SR, mono=True)[0]
        n = P.n_frames(len(x))
        f = P.analyze(x[: n * P.H])
        y = T.to_y(f["env"], f["per"], f["f0"])
        from scipy.ndimage import uniform_filter1d
        from scipy.signal import find_peaks
        c0 = y[0] * 10
        v = f["f0"] > 0
        pk, _ = find_peaks(np.where(v, uniform_filter1d(c0, 5), -1e9), prominence=3.0, distance=12)
        rate = len(pk) / (v.sum() / 200.0 + 1e-9)
        hyp = kana("".join(s.text for s in asr.transcribe(to16(x.astype(np.float64), P.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
        rows.append({"voice": it["voice"], "s": it["s"], "scale": it["scale"], "sec": round(len(x) / P.SR, 2), "rate": round(float(rate), 2), "cer": round(float(cer(kana(it["text"]), hyp)), 4)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    summ = {str(sc): {"rate_median": round(float(np.median([r["rate"] for r in rows if r["scale"] == sc])), 2), "cer_median": round(float(np.median([r["cer"] for r in rows if r["scale"] == sc])), 4),
                      "sec_median": round(float(np.median([r["sec"] for r in rows if r["scale"] == sc])), 2)} for sc in SCALES}
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    o = ROOT / "results/snv_d4/pilot.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(gen() if "gen" in sys.argv[1:] else measure())
