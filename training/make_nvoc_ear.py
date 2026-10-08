"""nvoc(出力部)の耳の盲検: 元音声 vs BigVGAN v2 44k(合格錨・非因果・参照専用)vs nvoc(Rust 製品経路の CLI で生成)。

事前登録 results/nvoc1/prereg.yaml の ear_entry_gate を満たした重みだけに使う(logmel ≤ 0.26・PESQ ≥ 3.6・hf ≥ −2.0)。
発話 = 除外話者 held21 のうち有声率の高い 3 発話(6s)。各試行は A/B/C の 3 本(元音声に RMS 整合 → 試行内共通の減衰)。
出力: results/earbattery/<dir>/nvoc_<stem>/{A,B,C}.wav・鍵 _key_聴取後に開く.json・listen.html(保存先 /save_nvoc)。

    CUDA_VISIBLE_DEVICES=0 uv run python make_nvoc_ear.py --export ../results/nvoc5/export
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_nvoc as E
import nvoc as N
from render_d1_ab import norm_trial

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"


def rust(export: Path, inp: Path, out: Path) -> str:
    r = subprocess.run(["cargo", "run", "--release", "-q", "-p", "lightvc-core", "--example", "nvoc_resynth", "--",
                        "--model", str(export), "--in", str(inp), "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stderr.strip().splitlines()[-1]


def bigvgan_fn(dev: str):
    import json as _j
    import librosa
    import bigvgan
    from bigvgan.env import AttrDict
    from bigvgan.meldataset import get_mel_spectrogram
    snaps = Path.home() / ".cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots"
    snap = sorted(snaps.iterdir())[-1]
    voc = bigvgan.BigVGAN(AttrDict(_j.loads((snap / "config.json").read_text())), use_cuda_kernel=False)
    voc.load_state_dict(torch.load(snap / "bigvgan_generator.pt", map_location="cpu")["generator"])
    voc.remove_weight_norm()
    voc = voc.eval().to(dev)

    def run(x: np.ndarray) -> np.ndarray:
        x44 = librosa.resample(x.astype(np.float64), orig_sr=N.SR, target_sr=44100).astype(np.float32)
        with torch.no_grad():
            y44 = voc(get_mel_spectrogram(torch.from_numpy(x44)[None].to(dev), voc.h)).squeeze().cpu().numpy()
        return librosa.resample(y44.astype(np.float64), orig_sr=44100, target_sr=N.SR)[:len(x)]
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--dir", default="nvoc_ab")
    ap.add_argument("--eb", default=str(EB), help="出力先の親(試験時は別の場所に)")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    export = Path(a.export).resolve()
    man = json.loads((export / "manifest.json").read_text())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eb = Path(a.eb).resolve()
    out = eb / a.dir
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_src"
    tmp.mkdir(exist_ok=True)
    items = sorted(E.held_items(), key=lambda it: -float((it["f0"][:6 * N.SR // N.HOP] > 0).mean()))[:a.n]
    bv = bigvgan_fn(dev)
    key: dict = {"export": str(export), "export_step": man["const"].get("step"), "nosrc": man["const"].get("NOSRC", False),
                 "trials": {}}
    for it in items:
        x = it["x"][:6 * N.SR].astype(np.float32)
        soundfile.write(tmp / f"{it['stem']}.wav", x, N.SR, subtype="FLOAT")
        log = rust(export, tmp / f"{it['stem']}.wav", tmp / f"{it['stem']}_nvoc.wav")
        y = soundfile.read(tmp / f"{it['stem']}_nvoc.wav")[0].astype(np.float64)
        clips = {"source": x.astype(np.float64), "bigvgan_v2": bv(x).astype(np.float64), "nvoc": y}
        n = min(len(v) for v in clips.values())
        clips = {k: v[:n] for k, v in clips.items()}
        met = {k: E.metrics(v, clips["source"], 0, dev) for k, v in clips.items() if k != "source"}
        normed = norm_trial(clips, clips["source"])
        names = list(normed)
        trial = f"nvoc_{it['stem']}"
        random.Random(trial).shuffle(names)
        td = out / trial
        td.mkdir(exist_ok=True)
        key["trials"][trial] = {"utt": it["stem"], "map": {}, "rust_log": log,
                                "metrics_vs_source": {k: {m: round(v, 4) for m, v in d.items()} for k, d in met.items()}}
        for i, nm in enumerate(names):
            soundfile.write(td / f"{'ABC'[i]}.wav", normed[nm], N.SR)
            key["trials"][trial]["map"]["ABC"[i]] = nm
        print(trial, log, flush=True)
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    page_key = {t: {"map": {L: "" for L in v["map"]}} for t, v in key["trials"].items()}
    (out / "_key_page.json").write_text(json.dumps(page_key, ensure_ascii=False))
    if eb == EB.resolve():
        subprocess.run([sys.executable, "make_ab_page.py", "--dir", a.dir, "--order", "nvoc", "--save", "/save_nvoc", "--no-err",
                        "--key", "_key_page.json",
                        "--title", "出力部(ボコーダ)の盲検",
                        "--intro", "各試行は同じ発話の 3 本で、1 本は元の録音(加工なし)です。<b>それぞれ『合格/不合格』</b>を付け、気づき(こもり・ざらつき・コーラス・声質)はメモへ。音量は揃えてあります。"],
                       cwd=Path(__file__).parent, check=True)
    print("ear page ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
