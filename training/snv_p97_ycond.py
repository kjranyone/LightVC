"""段 P97 step b(paper_snv): 描画器の学びの条件に混ぜる「模型の出力」を作る。
E1 + E1-NV の各発話について、擬似の操作者の版(data/snv_p_wav_x16・data/snv_p_feat_x16 = 高さ r 倍・声道 α 倍・録音の鎖 v2)を源にして、製品の経路の K3G(門つきの前段 → 3 組の平均と錨)で同じ声へ変換した出力 Y(29 行)と、描くときの高さ(門つきの前段の読み ÷ r)を data/snv_p97_y/{e1,e1nv}/<声>/<名前>.npz に置く。
操作者の登録 = 同じ版の他の普通の発話(8 本まで)・目標の構造 = 加工しない参照 3 本(源の発話を含まない)。教師は加工しない元の音(時刻は版と揃っている)。
    cd training && SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_p97_ycond.py 0/4
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
XF = ROOT / "data/snv_p_feat_x16"
XW = ROOT / "data/snv_p_wav_x16"
OUT = ROOT / "data/snv_p97_y"
K3G = "ens:snv_p24_c+mask:snv_p45_gate_t80+fb:anch|snv_p30_c+mask:snv_p45_gate_t80+fb:anch|snv_p91e_c+mask:snv_p45_gate_t80+fb:anch"


def main() -> int:
    i, n = (int(v) for v in sys.argv[1].split("/"))
    man = json.loads((XF / "manifest.json").read_text())["voices"]
    voices = sorted(p.name for p in (T.FEAT / "e1").iterdir())[i::n]
    if len(sys.argv) > 2:
        voices = voices[: int(sys.argv[2])]
    EV.System.render = lambda self, src, yo, k: None
    S = EV.Ens(K3G, EV.load_gen())
    lead = next(p for p in S.parts if p.envc)
    for vi, v in enumerate(voices):
        r = float(man[v]["r"])
        k = -12.0 * math.log2(r)
        ld = lambda kind, name: T.load_utt(XF / kind / v / f"{name}.npz") | {"wav_path": str(XW / kind / v / f"{name}.wav")}
        norm = sorted(p.stem for p in (XF / "e1" / v).glob("t*.npz"))
        nv = sorted(p.stem for p in (XF / "e1nv" / v).glob("*.npz")) if (XF / "e1nv" / v).exists() else []
        xs = {("e1", q): ld("e1", q) for q in norm} | {("e1nv", q): ld("e1nv", q) for q in nv}
        todo = [key for key in xs if not (OUT / key[0] / v / f"{key[1]}.npz").exists()]
        if not todo:
            continue
        for u in xs.values():
            f = lead.prep(u)
            for p in S.parts:
                p.pcache[id(u)] = (u, f)
        refs = {q: EV.utt(f"e1/{v}/{q}") for q in norm[:6]}
        sa = S.structure(f"t|{v}|a", [refs[q] for q in norm[:3]])
        sb = S.structure(f"t|{v}|b", [refs[q] for q in norm[3:6]])
        for kind, q in todo:
            u = xs[(kind, q)]
            stat = [xs[("e1", z)] for z in norm if not (kind == "e1" and z == q)][:8]
            _, yo = S.convert(u, stat, sb if kind == "e1" and q in norm[:3] else sa, k, with_y=True)
            f0 = lead.prep(u)["f0"] / r
            dst = OUT / kind / v / f"{q}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, y=yo[0, :29].cpu().numpy().astype(np.float32), f0=f0.astype(np.float32))
        for p in S.parts:
            p.pcache.clear(), p.cache.clear(), p.scache.clear(), p.hmu.clear(), p.hc0.clear()
        torch.cuda.empty_cache()
        print(vi, len(voices), v, len(todo), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
