"""段 P73(results/snv_p73/prereg.yaml): 測りの組の台帳 data/snv_p73/sets.json を作る(p18_nnc.py の sets=x:<組>・tpool=x:<組>)。
gm = sttts-gui に同梱の Gemini TTS の合成の 8 声(各 1 片を前半 = 源・後半 = 登録に切る)/ gm_ir = 同じ声を参照に Irodori が読んだ声(snv_p73_gen.py・源 t00・t01・登録 t02〜t04)/
bl = E1 の検証の男声を 24 kHz に落として戻した音(帯域の対照・EMA は元の音のもの)/ jvs_m・jvs_f = JVNV の男 2 人・女 2 人の、同じ感情の中の組(源 regular_01・05・登録 02〜04)。
    cd training && uv run --no-sync python snv_diag/p73_sets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/snv_p73"
SP = "data/snv_e1_feat/sparc"
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")


QW_ATTR = {"age": ["二十代", "三十代", "四十代", "五十代"], "pitch": ["低い", "やや低い", "普通の高さの", "かなり低い"],
           "style": ["落ち着いてはっきり話す", "くだけた口調で気楽に話す", "ぼそぼそと小さな声で話す", "早口で話す", "低くかすれた声でゆっくり話す", "淡々と読み上げる", "明るく元気に話す", "疲れた様子で気だるく話す"]}


def qw_plan() -> None:
    import random
    import sys
    sys.path.insert(0, str(ROOT / "training"))
    import snv_e1_gen as E1
    rng = random.Random(7302)
    pool = E1.text_pool()
    voices = [{"voice": f"qm{i:02d}", "seed": 73020000 + 100 * i,
               "instruct": f"{QW_ATTR['age'][i % 4]}の日本人男性。{QW_ATTR['pitch'][(i // 4 + i) % 4]}声で、{QW_ATTR['style'][i % 8]}。普通の部屋で録った自然な話し声。",
               "files": [{"t": f"t{j:02d}", "text": t} for j, t in enumerate(rng.sample(pool, 5))]} for i in range(16)]
    (D / "qw").mkdir(parents=True, exist_ok=True)
    if not (D / "qw/manifest.json").exists():
        (D / "qw/manifest.json").write_text(json.dumps({"design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "clone": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "voices": voices}, ensure_ascii=False, indent=1))
    rng = random.Random(7303)
    vx = [v | {"voice": f"vm{i:02d}", "seed": 7303000 + 100 * i, "files": [{"t": f"t{j:02d}", "text": t} for j, t in enumerate(rng.sample(pool, 5))]} for i, v in enumerate(voices)]
    (D / "vx").mkdir(parents=True, exist_ok=True)
    (D / "vx/manifest.json").write_text(json.dumps({"model": "openbmb/VoxCPM2", "voices": vx}, ensure_ascii=False, indent=1))


def main() -> int:
    import sys
    if "qw_plan" in sys.argv[1:]:
        qw_plan()
        return 0
    sets: dict = json.loads((D / "sets.json").read_text()) if (D / "sets.json").exists() else {}
    w = lambda wav, ema: {"wav": wav, "ema": ema}
    gm = []
    for p in sorted((D / "gm").glob("*.flac")):
        x, sr = sf.read(p, dtype="float32")
        d = D / "gm" / p.stem
        d.mkdir(exist_ok=True)
        h = len(x) // 2
        k = h + int(np.argmin(np.convolve(x[h - sr:h + sr] ** 2, np.ones(sr // 20), "same"))) - sr
        sf.write(d / "a.wav", x[:k], sr)
        sf.write(d / "b.wav", x[k:], sr)
        f0 = librosa.yin(x, fmin=60, fmax=500, sr=sr)
        gm.append({"voice": p.stem, "f0": round(float(np.median(f0)), 1), "src": [w(f"data/snv_p73/gm/{p.stem}/a.wav", f"{SP}/p73/gm/{p.stem}_a.npz")], "enr": [w(f"data/snv_p73/gm/{p.stem}/b.wav", f"{SP}/p73/gm/{p.stem}_b.npz")]})
    sets["gm"] = gm
    if (D / "gm_ir/manifest.json").exists():
        f = lambda v, t: w(f"data/snv_p73/gm_ir/{v}/{t}.wav", f"{SP}/p73/gm_ir/{v}_{t}.npz")
        sets["gm_ir"] = [{"voice": v["voice"], "src": [f(v["voice"], "t00"), f(v["voice"], "t01")], "enr": [f(v["voice"], t) for t in ("t02", "t03", "t04")]} for v in json.loads((D / "gm_ir/manifest.json").read_text())["voices"]]
    val = [v for v in json.loads((ROOT / "results/snv_p19_x/split.json").read_text())["val"] if v.startswith("em")]
    for v in val:
        for t in ("t00", "t01", "t02", "t05", "t06"):
            q = D / f"bl/{v}/{t}.wav"
            if not q.exists():
                q.parent.mkdir(parents=True, exist_ok=True)
                x, sr = sf.read(ROOT / f"data/snv_e1/{v}/{t}.wav", dtype="float32")
                sf.write(q, librosa.resample(librosa.resample(x, orig_sr=sr, target_sr=24000), orig_sr=24000, target_sr=sr), sr)
    b = lambda v, t: w(f"data/snv_p73/bl/{v}/{t}.wav", f"{SP}/e1/{v}/{t}.npz")
    sets["bl"] = [{"voice": v, "src": [b(v, "t05"), b(v, "t06")], "enr": [b(v, t) for t in ("t00", "t01", "t02")]} for v in val]
    j = lambda sp, e, n: w(f"data/jvnv/jvnv_v1/{sp}/{e}/regular/{sp}_{e}_regular_{n}.wav", f"{SP}/p73/jvs/{sp}_{e}_regular_{n}.npz")
    for nm, sps in (("jvs_m", ("M1", "M2")), ("jvs_f", ("F1", "F2"))):
        sets[nm] = [{"voice": f"{sp}_{e}", "src": [j(sp, e, "01"), j(sp, e, "05")], "enr": [j(sp, e, n) for n in ("02", "03", "04")]} for sp in sps for e in EMOS]
    if (D / "qw/manifest.json").exists():
        f = lambda v, t: w(f"data/snv_p73/qw/{v}/{t}.wav", f"{SP}/p73/qw/{v}_{t}.npz")
        sets["qw"] = [{"voice": v["voice"], "src": [f(v["voice"], "t01"), f(v["voice"], "t02")], "enr": [f(v["voice"], t) for t in ("t03", "t04", "t00")]} for v in json.loads((D / "qw/manifest.json").read_text())["voices"]
                      if all((D / "qw" / v["voice"] / f"t0{j}.wav").exists() for j in range(5))]
    if (D / "vx/manifest.json").exists():
        f = lambda v, t: w(f"data/snv_p73/vx/{v}/{t}.wav", f"{SP}/p73/vx/{v}_{t}.npz")
        sets["vx"] = [{"voice": v["voice"], "src": [f(v["voice"], "t01"), f(v["voice"], "t02")], "enr": [f(v["voice"], t) for t in ("t03", "t04", "t00")]} for v in json.loads((D / "vx/manifest.json").read_text())["voices"]
                      if all((D / "vx" / v["voice"] / f"t0{j}.wav").exists() for j in range(5))]
    (D / "sets.json").write_text(json.dumps(sets, ensure_ascii=False, indent=1))
    print({k: len(v) for k, v in sets.items()}, {v["voice"]: v["f0"] for v in gm})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
