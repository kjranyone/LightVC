"""D672: 現状の試聴(オーナーの依頼 10-29・盲検でない・判定に使わない)。開発の組 dev2 / dev3 から k = 5 / 7 / 12 を 1 例ずつ(Random(672)・選り好みなし)。
音声 = 入力(SRC 8 s)・目標(d37/cen_TTT の音量が最大の 8 s)・今の候補(Rust の構音の描画 RUST{k}R = artic_render2・γ 0.5)・前の候補(v5 の生成器の経路 OURS{k}RY58GQ3 / OURS12RY58G)・錨 GU{k}。
音量 = 例の中で 5 本を同じ RMS(−23 dBFS・有音の部分)・ピーク 0.95 を超えるなら 5 本に共通の減衰。出力 = results/earbattery/now_v6_dev/{eNN_*.wav, listen.html}。
    uv run python d672_now_listen.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from d566_mf_v3_ear import active_rms, best8, load

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
OUT = ROOT / "results/earbattery/now_v6_dev"
SR = 48000
PREV = {5: "OURS5RY58GQ3", 7: "OURS7RY58GQ3", 12: "OURS12RY58G"}
GU = {5: "GU5_a112", 7: "GU7_a112", 12: "GU12_a120"}
NAMES = [("in", "入力(男声)"), ("tgt", "目標(本人の録音)"), ("now", "今の候補(構音の描画)"), ("prev", "前の候補(v5・生成器)"), ("gu", "錨(一律シフト)")]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(672)
    items = []
    for s in ("dev2", "dev3"):
        for k in (5, 7, 12):
            D = H / s / "d570"
            p = rng.choice(sorted(f.name[:10] for f in (D / f"RUST{k}R").glob("p*.wav")))
            xs = {"in": load(H / s / "d56" / f"{p}_SRC.wav")[: 8 * SR], "tgt": best8(load(H / s / "d37" / f"cen_{int(p[7:10]):03d}.wav")),
                  "now": load(D / f"RUST{k}R" / f"{p}_RUST{k}R.wav"), "prev": load(D / PREV[k] / f"{p}_{PREV[k]}.wav"), "gu": load(D / GU[k] / f"{p}_{GU[k]}.wav")}
            m = min(len(x) for x in xs.values())
            xs = {n: x[:m] * (10 ** (-23 / 20) / active_rms(x[:m])) for n, x in xs.items()}
            pk = max(float(np.abs(x).max()) for x in xs.values())
            g = 0.95 / pk if pk > 0.95 else 1.0
            eid = f"e{len(items) + 1:02d}"
            for n, x in xs.items():
                sf.write(OUT / f"{eid}_{n}.wav", (x * g).astype(np.float32), SR, subtype="PCM_16")
            items.append({"id": eid, "set": s, "k": k, "pair": p})
    (OUT / "items.json").write_text(json.dumps(items, indent=1))
    rows = "".join(f"<section><h2>{it['id']}・+{it['k']} 半音</h2>" + "".join(f"<div class=r><span>{lab}</span><audio controls preload=none src='{it['id']}_{n}.wav'></audio></div>" for n, lab in NAMES) + "</section>" for it in items)
    (OUT / "listen.html").write_text(PAGE.replace("__ROWS__", rows), encoding="utf-8")
    print("items", len(items), "->", OUT, flush=True)
    return 0


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>現状の試聴 v6</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1c1a;--mut:#6b6b66;--line:#e2e1dc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#33322f}}
body{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;margin:0;padding:16px;max-width:760px}
h1{font-size:20px;margin:0 0 4px}p{color:var(--mut);margin:0 0 16px}
section{border-top:1px solid var(--line);padding:12px 0}h2{font-size:16px;margin:0 0 8px}
.r{display:flex;align-items:center;gap:12px;margin:4px 0;flex-wrap:wrap}.r span{width:190px}audio{width:min(100%,420px)}
</style></head><body>
<h1>現状の試聴(開発の組・盲検ではありません)</h1>
<p>男声 → 女声。各例とも 5 本を同じ音量にそろえています。今の候補 = 生成器を使わない構音の描画(Rust・γ 0.5)。前の候補 = 10-28 に耳で錨に負けた v5 の経路。</p>
__ROWS__
</body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
