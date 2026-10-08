"""A2-VC 段 S1 の判定: held21 の写し合成(c_in 条件 m12_a085 / p5_a110)・c_in の使用量(c_in = 0 の比較)・学習済み重みの実音声未来不変性。
--ear で盲検(元音声 / BigVGAN v2 / A2 写し合成 m12_a085)を results/earbattery/<dir> に作る。

    CUDA_VISIBLE_DEVICES=0 uv run python eval_a2vc.py --ckpt ../results/a2vc_s1/snap/ema_20k.pt --out ../results/a2vc_s1/eval_20k.json [--ear]
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
import a2vc as A
import eval_nvoc as E
import nvoc as N
import train_a2vc as TA
import zsvc as Z
from eval_zsvc import contrast

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"


def render(model: A.A2Circuit, front: Z.ZSVC, it: dict, cond: str, dev: str, zero_cin: bool = False) -> np.ndarray:
    pert = np.zeros_like(it["perts"][cond]) if zero_cin else it["perts"][cond]
    nz = torch.randn(1, len(it["x"]), generator=torch.Generator().manual_seed(0)).to(dev)
    with torch.no_grad():
        return TA.run(model, front, torch.from_numpy(it["seg"])[None].to(dev), torch.from_numpy(pert)[None].to(dev),
                      torch.from_numpy(it["f0a"])[None].to(dev), nz)[0].cpu().numpy()


def score(ys: list, ev: list, dev: str) -> dict:
    rows = []
    for y, it in zip(ys, ev):
        m = E.metrics(y, it["x"], N.DELAY, dev)
        m["contrast"] = contrast(y[N.DELAY:], it["f0"])
        rows.append(m)
    return {k: round(float(np.nanmean([r[k] for r in rows])), 3) for k in rows[0]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ear", action="store_true")
    ap.add_argument("--dir", default="a2vc_s1_ab")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = A.A2Circuit(ch=st["ch"], d_ctrl=TA.D_CTRL).to(dev).eval()
    model.load_state_dict(st["ema"])
    front = Z.ZSVC().to(dev).eval()
    items = E.held_items()
    ev = TA.eval_set(items)
    rep: dict = {"ckpt": a.ckpt, "step": int(st["step"]), "held_n": len(ev)}
    for cond, _, _ in TA.EVAL_COND:
        rep[cond] = score([render(model, front, it, cond, dev) for it in ev], ev, dev)
    rep["m12_a085_cin0"] = score([render(model, front, it, "m12_a085", dev, zero_cin=True) for it in ev], ev, dev)
    rep["ship_gate_trained"] = bool(TA.ship_gate(model, front))
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    if not a.ear:
        return 0
    from make_nvoc_ear import bigvgan_fn
    from render_d1_ab import norm_trial
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    bv = bigvgan_fn(dev)
    order = sorted(range(len(ev)), key=lambda i: -float((ev[i]["f0"][:6 * N.SR // N.HOP] > 0).mean()))[:a.n]
    key: dict = {"ckpt": a.ckpt, "step": rep["step"], "cond": "m12_a085(c_in = −12 半音・α 0.85 で嘘にした入力・制御 = 正解の包絡・パルス = 正解の f0)", "trials": {}}
    for i in order:
        it = ev[i]
        n = 6 * N.SR
        x = it["x"][:n].astype(np.float64)
        y = render(model, front, it, "m12_a085", dev)[N.DELAY:N.DELAY + n].astype(np.float64)
        clips = {"source": x, "bigvgan_v2": bv(it["x"][:n]).astype(np.float64), "a2vc": y}
        m = min(len(v) for v in clips.values())
        clips = {k: v[:m] for k, v in clips.items()}
        met = {k: E.metrics(v, clips["source"], 0, dev) for k, v in clips.items() if k != "source"}
        normed = norm_trial(clips, clips["source"])
        names = list(normed)
        trial = f"a2_{items[i]['stem']}"
        random.Random(trial).shuffle(names)
        td = out / trial
        td.mkdir(exist_ok=True)
        key["trials"][trial] = {"utt": items[i]["stem"], "map": {},
                                "metrics_vs_source": {k: {q: round(v, 4) for q, v in d.items()} for k, d in met.items()}}
        for j, nm in enumerate(names):
            soundfile.write(td / f"{'ABC'[j]}.wav", normed[nm], N.SR)
            key["trials"][trial]["map"]["ABC"[j]] = nm
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (out / "_key_page.json").write_text(json.dumps({t: {"map": {L: "" for L in v["map"]}} for t, v in key["trials"].items()}, ensure_ascii=False))
    subprocess.run([sys.executable, "make_ab_page.py", "--dir", a.dir, "--order", "a2", "--save", "/save_a2vc", "--no-err",
                    "--key", "_key_page.json", "--title", "A2 回路の写し合成の盲検",
                    "--intro", "各試行は同じ発話の 3 本で、1 本は元の録音(加工なし)です。<b>それぞれ『合格/不合格』</b>を付け、気づき(ガビガビ・ブザー・こもり・ざらつき・機械的な声)はメモへ。音量は揃えてあります。"],
                   cwd=Path(__file__).parent, check=True)
    print("ear page ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
