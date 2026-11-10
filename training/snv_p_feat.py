"""P 線(synth_nam_vc §8)の特徴: P0a の前段(snv_p0_front・48 kHz・200 fps・先読み 0)で f0・谷の値・包絡 25・周期性 4 を書き出す。
組 = e1(data/snv_e1 の 600 声 × 10)・e1nv(data/snv_e1nv の 4,221)・d(data/snv_d = paper_snv 段 D の内容を共有するコーパス)・a0(results/snv_a0/wav = 目標と自己の評価)・n0r(results/snv_n0r/wav)・
real(本物の男声 = 外郎売・namikawa の登録 30 s と 10 s の区間 = snv_a1_convert.segments・48 kHz へ)・vctk(VCTK の男声 6 人 × 2 の開発の入力)。
出力 = data/snv_p_feat/{組}/{名}.npz(f0 [T]・ap [T]・env [25, T]・per [4, T]・float32)・real と vctk は wav も 48 kHz で置く。
    cd training && uv run --no-sync python snv_p_feat.py [組 ...]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p_feat"
VCTK = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
VCTK_M = ("p226", "p227", "p232", "p237", "p241", "p243")
SETS = ("e1", "e1nv", "d", "a0", "n0r", "real", "vctk")


def load48(p: Path) -> np.ndarray:
    return librosa.load(str(p), sr=P.SR, mono=True)[0].astype(np.float32)


def jobs_for(name: str) -> list:
    if name == "e1":
        return [(f"e1/{p.parent.name}/{p.stem}", p) for p in sorted((ROOT / "data/snv_e1").glob("e*/t*.wav"))]
    if name == "e1nv":
        return [(f"e1nv/{p.parent.name}/{p.stem}", p) for p in sorted((ROOT / "data/snv_e1nv").glob("e*/*.wav"))]
    if name == "d":
        return [(f"d/{p.parent.name}/{p.stem}", p) for p in sorted((ROOT / "data/snv_d").glob("e*/s*.wav"))]
    if name == "a0":
        return [(f"a0/{p.stem}", p) for p in sorted((ROOT / "results/snv_a0/wav").glob("*.wav"))]
    if name == "n0r":
        return [(f"n0r/{p.stem}", p) for p in sorted((ROOT / "results/snv_n0r/wav").glob("*.wav"))]
    if name == "real":
        from snv_a1_convert import segments
        out = []
        (OUT / "real").mkdir(parents=True, exist_ok=True)
        for spk, mp3 in (("uirou", ROOT / "外郎売.mp3"), ("namikawa", ROOT / "namikawa.mp3")):
            enr, tests, sr = segments(mp3)
            for tag, seg in [("enroll", enr)] + [(str(i), s) for i, s in enumerate(tests)]:
                q = OUT / f"real/{spk}_{tag}.wav"
                if not q.exists():
                    sf.write(q, librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=P.SR), P.SR)
                out.append((f"real/{spk}_{tag}", q))
        return out
    if name == "vctk":
        out = []
        (OUT / "vctk").mkdir(parents=True, exist_ok=True)
        for spk in VCTK_M:
            ws = sorted((VCTK / spk).glob("*.wav"))
            for w in ws[20:22] + ws[30:40]:
                q = OUT / f"vctk/{spk}_{w.stem}.wav"
                if not q.exists():
                    sf.write(q, load48(w), P.SR)
                out.append((f"vctk/{spk}_{w.stem}", q))
        return out
    raise SystemExit(f"unknown set {name}")


def work(job: tuple) -> str:
    key, src = job
    torch.set_num_threads(1)
    dst = OUT / f"{key}.npz"
    if dst.exists():
        return key
    x = load48(src)
    T = P.n_frames(len(x))
    f = P.analyze(x[: T * P.H])
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    return key


def main() -> int:
    names = sys.argv[1:] or list(SETS)
    for name in names:
        js = jobs_for(name)
        done = 0
        with ProcessPoolExecutor(11) as ex:
            for _ in ex.map(work, js, chunksize=4):
                done += 1
                if done % 500 == 0:
                    print(name, done, "/", len(js), flush=True)
        print(name, "done", len(js), flush=True)
    (OUT / "front.json").write_text(json.dumps({k: getattr(P, k) for k in ("SR", "H", "W_YIN", "LP_HZ", "LP_ORDER", "K_CAND", "C_MAX", "BETA", "LAMBDA", "GAMMA", "U_COST", "TMIN", "TMAX", "E_FLOOR", "NFFT_P")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
