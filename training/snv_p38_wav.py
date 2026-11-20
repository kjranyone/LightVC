"""段 P38(paper_snv §5 M29 の版を前段の重みの学習の声に): 擬似の操作者の版の波形を保存する。
snv_p16_feat.py と同じ声ごとの (r, α, 部屋) と同じ乱数で、E1 と E1-NV の全ての発話を WORLD の分析合成(高さ r 倍・声道 α 倍)→ 録音の鎖 v2 に通し、
その音を data/snv_p_wav_x16/{e1,e1nv}/<声>/<発話>.wav(48 kHz・16 bit)に置く(data/snv_p_feat_x16 の特徴の元の音と同じもの)。
加工は入力側だけに使う(重みの当てる先は、この版の声の成分と足した音の成分から計算する)。台帳 = data/snv_p_wav_x16/manifest.json。
段 P55: kinds=e1f = 速く話す学習の声(data/snv_e1f・特徴は data/snv_p_feat/e1f)にも同じ声ごとの (r, α, 部屋) で版を作る(台帳は書き換えず、check は行わない)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_p38_wav.py [check] [kinds=e1f]
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
OUT = ROOT / "data/snv_p_wav_x16"
SRCS = X.SRC | {"e1f": ROOT / "data/snv_e1f"}


def work(job: tuple) -> str:
    import librosa
    kind, voice, name, p = job
    torch.set_num_threads(1)
    dst = OUT / kind / voice / f"{name}.wav"
    if dst.exists():
        return ""
    x = librosa.load(str(SRCS[kind] / voice / f"{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
    x = W.shift(x, P.SR, p["r"], p["alpha"])
    x = CH.apply(x, p["room"], np.random.default_rng(CH.seed_of(f"p16|{voice}|{kind}|{name}")))
    pk = float(np.abs(x).max())
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    sf.write(tmp, np.clip(x, -1.0, 1.0), P.SR, subtype="PCM_16")
    tmp.rename(dst)
    return f"{kind}/{voice}/{name}: 尖頭値 {pk:.2f} > 1(切った)" if pk > 1.0 else ""


def check(n: int = 24) -> None:
    files = sorted(OUT.glob("e1/*/t05.wav"))
    g = np.random.default_rng(0)
    d = []
    for q in [files[i] for i in g.choice(len(files), min(n, len(files)), replace=False)]:
        x = sf.read(str(q), dtype="float32")[0]
        f = P.analyze(x[: P.n_frames(len(x)) * P.H])
        r = np.load(ROOT / "data/snv_p_feat_x16/e1" / q.parent.name / "t05.npz")
        L = min(len(f["f0"]), len(r["f0"]))
        v = (f["f0"][:L] > 0) & (r["f0"][:L] > 0)
        d.append((float(np.mean((f["f0"][:L] > 0) == (r["f0"][:L] > 0))), float(np.abs(f["env"][:, :L][:, v] - r["env"][:, :L][:, v]).mean()), len(f["f0"]) == len(r["f0"])))
    print("check", len(d), "有声の一致の最小", round(min(x[0] for x in d), 4), "有声のフレームの包絡の差の平均の最大", round(max(x[1] for x in d), 4), "フレーム数の一致", sum(x[2] for x in d), flush=True)
    assert min(x[0] for x in d) > 0.97 and all(x[2] for x in d), "保存した波形が data/snv_p_feat_x16 の特徴の元の音と合わない"


def main() -> int:
    if "check" in sys.argv[1:]:
        check()
        return 0
    man = json.loads((ROOT / "data/snv_p_feat_x16/manifest.json").read_text())
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    kinds = opt["kinds"].split(",") if "kinds" in opt else list(X.SRC)
    files = [(kind, q.parent.name, q.stem) for kind in kinds for q in sorted((X.FEAT / kind).glob("*/*.npz"))]
    jobs = [(kind, v, name, man["voices"][v]) for kind, v, name in files]
    OUT.mkdir(parents=True, exist_ok=True)
    if "kinds" not in opt:
        (OUT / "manifest.json").write_text(json.dumps({"stage": "paper_snv 段 P38(§5 M29 の版の波形)", "use": "入力側だけ(前段の重みの学習の声)", "same_audio_as": "data/snv_p_feat_x16", "chain": CH.VER, "voices": man["voices"]}, ensure_ascii=False, indent=1))
    print("発話", len(jobs), "声", len(man["voices"]), flush=True)
    notes = []
    with ProcessPoolExecutor(14) as ex:
        for i, e in enumerate(ex.map(work, jobs, chunksize=8)):
            if e:
                notes.append(e)
            if (i + 1) % 1000 == 0:
                print(i + 1, "/", len(jobs), "切った発話", len(notes), flush=True)
    print("done", len(jobs), "切った発話", len(notes), flush=True)
    if "kinds" in opt:
        (OUT / f"clipped_{'_'.join(kinds)}.json").write_text(json.dumps(notes, ensure_ascii=False, indent=0))
        return 0
    (OUT / "clipped.json").write_text(json.dumps(notes, ensure_ascii=False, indent=0))
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
