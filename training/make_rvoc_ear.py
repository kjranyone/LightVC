"""出力部 rvoc の写し合成の耳(合格線の較正): 元音声 vs BigVGAN v2 44k(合格錨) vs rvoc(PyTorch の EMA 重み・f0 = harvest・包絡は元音声の分析)。
問いは「元の録音と並べて、合格か不合格か」(基準のある問い)。重みは数値の合格線(PESQ ≥ 3.2)に届いていない: 線が耳に対して意味を持つかの較正のため、
重みと数値を鍵に残す。Rust 経路とのビット一致は未確認なので、これは PyTorch の生成器の音。
    uv run python make_rvoc_ear.py --ckpt ../results/diag_rvoc1s/snap/ema_110k.pt --env_smooth 0.25 --dir rvoc_ab
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
import rvoc as R
import train_rvoc as TR
from make_nvoc_ear import bigvgan_fn
from render_d1_ab import norm_trial

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--env_smooth", type=float, default=None, help="省略時は ckpt の env_smooth(無ければ 0)")
    ap.add_argument("--dir", default="rvoc_ab")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--save", default="/save_rvoc")
    ap.add_argument("--hi", action="store_true", help="f0 を f0hi の教師(1kHz 超を含む)にする")
    ap.add_argument("--hi_eval", type=int, default=0, help="data/hi_eval の 1kHz 超の区間(評価話者・置換フレームの多い順)を追加する本数")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=st["cfg"]["ch"], kernels=tuple(st["cfg"]["kernels"]), dils=tuple(st["cfg"]["dils"]), d_cond=st["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(st["ema"])
    gen.eval()
    front = TR.Front("pae", TR.ckpt_env_smooth(st, a.env_smooth)).to(dev)
    items = TR.held(dev, a.hi)
    held_e = {it["stem"]: it for it in E.held_items()} if False else None
    order = sorted(range(len(items)), key=lambda i: -float((items[i]["f0_h"][:6 * N.SR // N.HOP] > 0).mean()))[:a.n]
    if a.hi_eval:
        import json as _j
        from hi_probe import item as hi_item
        idx = sorted(_j.loads((ROOT / "data/hi_eval/index.json").read_text())["items"], key=lambda r: -r["hi_frames"])[:a.hi_eval]
        for r in idx:
            z = np.load(ROOT / "data/hi_eval" / f"{r['name']}.npz")
            items.append(hi_item(z["x"].astype(np.float32), z["f0"]))
            order.append(len(items) - 1)
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    bv = bigvgan_fn(dev)
    key: dict = {"ckpt": a.ckpt, "env_smooth": TR.ckpt_env_smooth(st, a.env_smooth), "step": st.get("step"), "trials": {}}
    for i in order:
        it = items[i]
        x = it["x"][:6 * N.SR].astype(np.float32)
        with torch.no_grad():
            y = TR.render(gen, front, it, it["f0_h"], dev)[N.DELAY:][:len(x)].astype(np.float64)
        clips = {"source": x.astype(np.float64), "bigvgan_v2": bv(x).astype(np.float64), "rvoc": y}
        n = min(len(v) for v in clips.values())
        clips = {k: v[:n] for k, v in clips.items()}
        met = {k: E.metrics(v, clips["source"], 0, dev) for k, v in clips.items() if k != "source"}
        normed = norm_trial(clips, clips["source"])
        names = list(normed)
        trial = f"rvoc_{i:02d}"
        random.Random(trial).shuffle(names)
        td = out / trial
        td.mkdir(exist_ok=True)
        key["trials"][trial] = {"item": i, "map": {}, "metrics_vs_source": {k: {m: round(v, 4) for m, v in d.items()} for k, d in met.items()}}
        for j, nm in enumerate(names):
            soundfile.write(td / f"{'ABC'[j]}.wav", normed[nm], N.SR)
            key["trials"][trial]["map"]["ABC"[j]] = nm
        print(trial, key["trials"][trial]["metrics_vs_source"], flush=True)
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (out / "_key_page.json").write_text(json.dumps({t: {"map": {L: "" for L in v["map"]}} for t, v in key["trials"].items()}, ensure_ascii=False))
    subprocess.run([sys.executable, "make_ab_page.py", "--dir", a.dir, "--order", "rvoc", "--save", a.save, "--no-err", "--key", "_key_page.json",
                    "--title", "出力部(ボコーダ)の写し合成",
                    "--intro", "各試行は同じ発話の 3 本で、1 本は元の録音(加工なし)です。<b>それぞれ『合格/不合格』</b>を付け、気づき(こもり・ざらつき・コーラス・声質)はメモへ。音量は揃えてあります。"],
                   cwd=Path(__file__).parent, check=True)
    print("ear page ->", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
