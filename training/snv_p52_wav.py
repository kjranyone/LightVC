"""段 P52 step b(paper_snv §5 M39): 粗さを散らした擬似の操作者の版の波形を作る。
snv_p38_wav.py と同じ声ごとの (r, α, 部屋) と同じ鎖の乱数で、E1 と E1-NV の全ての発話を WORLD の分析合成に通す。違いは合成の前の粗さだけ(snv_d5_world.shift_rough):
発話ごとの乱数で 非周期性 a ~ U[0, 0.5](4 kHz より下)・スペクトルの傾き t ~ U[−3, +1] dB/oct・確率 0.3 で包絡のケプストラムを 40 次で切る(帯域幅)。その後に録音の鎖 v2。
出力 = data/snv_p_wav_x52/{e1,e1nv}/<声>/<発話>.wav(48 kHz・16 bit・元と同じ長さ)・台帳 manifest.json(発話ごとの粗さ)。加工は入力側だけに使う(重みの当てる先はこの版の声の成分と足した音の成分から)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_p52_wav.py [check]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import snv_d5_chain as CH
import snv_d5_world as W
import snv_p0_front as P
import snv_p16_feat as X

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p_wav_x52"


def rough_of(voice: str, kind: str, name: str) -> dict:
    g = np.random.default_rng(CH.seed_of(f"p52|{voice}|{kind}|{name}"))
    return {"ap": round(float(g.uniform(0.0, 0.5)), 4), "tilt": round(float(g.uniform(-3.0, 1.0)), 4), "bw": bool(g.random() < 0.3)}


def work(job: tuple) -> tuple:
    import librosa
    kind, voice, name, p = job
    torch.set_num_threads(1)
    q = rough_of(voice, kind, name)
    dst = OUT / kind / voice / f"{name}.wav"
    if dst.exists():
        return f"{kind}/{voice}/{name}", q, ""
    x = librosa.load(str(X.SRC[kind] / voice / f"{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
    x = W.shift_rough(x, P.SR, p["r"], p["alpha"], q["ap"], q["tilt"], q["bw"])
    x = CH.apply(x, p["room"], np.random.default_rng(CH.seed_of(f"p16|{voice}|{kind}|{name}")))
    pk = float(np.abs(x).max())
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    sf.write(tmp, np.clip(x, -1.0, 1.0), P.SR, subtype="PCM_16")
    tmp.rename(dst)
    return f"{kind}/{voice}/{name}", q, (f"尖頭値 {pk:.2f} > 1(切った)" if pk > 1.0 else "")


def check(n: int = 24) -> None:
    files = sorted(OUT.glob("e1/*/t05.wav"))
    g = np.random.default_rng(0)
    d = []
    for q in [files[i] for i in g.choice(len(files), min(n, len(files)), replace=False)]:
        x = sf.read(str(q), dtype="float32")[0]
        y = sf.read(str(ROOT / "data/snv_p_wav_x16/e1" / q.parent.name / "t05.wav"), dtype="float32")[0]
        f, r = P.analyze(x[: P.n_frames(len(x)) * P.H]), P.analyze(y[: P.n_frames(len(y)) * P.H])
        L = min(len(f["f0"]), len(r["f0"]))
        d.append((float(np.mean((f["f0"][:L] > 0) == (r["f0"][:L] > 0))), len(x) == len(y)))
    print("check", len(d), "粗さのない版との有声の一致の最小", round(min(v[0] for v in d), 4), "中央値", round(float(np.median([v[0] for v in d])), 4), "長さの一致", sum(v[1] for v in d), flush=True)
    assert all(v[1] for v in d) and float(np.median([v[0] for v in d])) > 0.85, "粗さの版が版と大きく食い違う"


def main() -> int:
    if "check" in sys.argv[1:]:
        check()
        return 0
    man = json.loads((ROOT / "data/snv_p_feat_x16/manifest.json").read_text())
    files = [(kind, q.parent.name, q.stem) for kind in X.SRC for q in sorted((X.FEAT / kind).glob("*/*.npz"))]
    jobs = [(kind, v, name, man["voices"][v]) for kind, v, name in files]
    OUT.mkdir(parents=True, exist_ok=True)
    print("発話", len(jobs), "声", len(man["voices"]), flush=True)
    rough, notes = {}, []
    with ProcessPoolExecutor(14) as ex:
        for i, (key, q, e) in enumerate(ex.map(work, jobs, chunksize=8)):
            rough[key] = q
            if e:
                notes.append(f"{key}: {e}")
            if (i + 1) % 1000 == 0:
                print(i + 1, "/", len(jobs), "切った発話", len(notes), flush=True)
    (OUT / "manifest.json").write_text(json.dumps({"stage": "paper_snv 段 P52 step b(§5 M39 の粗さを散らした版の波形)", "use": "入力側だけ(前段の重みの学習の声)", "base": "data/snv_p_wav_x16 と同じ声ごとの r・α・部屋・鎖の乱数", "rough": {"ap": "U[0, 0.5]", "tilt_db_oct": "U[-3, 1]", "bw_p": 0.3}, "chain": CH.VER, "voices": man["voices"], "utterances": rough}, ensure_ascii=False, indent=0))
    (OUT / "clipped.json").write_text(json.dumps(notes, ensure_ascii=False, indent=0))
    print("done", len(jobs), "切った発話", len(notes), flush=True)
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
