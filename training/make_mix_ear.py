"""--conds / --labels で比べる条件を替えられる(例: 伸縮の耳 warp_ab1 = TAB,M35,W130S・D37)。
つまみの耳(製品候補 = Rust の製品経路・柔らかい引き・参照 20s・D29): 参照 R を聴いて、mix = 0 / 0.2 / 0.35 の 3 本を『聞き取りやすい順』『参照の人らしい順』に並べる。
旧: 変換の耳(製品候補 = 基準経路・F2_2・参照 20s): 参照(目標の実音声)を聴いた上で、3 本を順位づける。基準のある問い。
各試行 = 参照 R(目標の別の実音声 8s)+ 3 本(A/B/C をシャッフル): 目標どおりの変換・別の目標の変換・元の男声。
問い 1 =『参照の人に近い順』(1 = 最も近い)・問い 2 =『自然さ(不自然さの少ない順)』。回答の鍵は _key_聴取後に開く.json(サーバは配信しない)。
    uv run python make_conv_ear.py --wav_dir <conv_ja --save_dir> --dir conv_ab1 --n 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from render_d1_ab import norm_trial

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
SR = 48000
SEC = 8


def rd(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(1)[: SEC * SR].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--dir", default="conv_ab1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cond", default="TAB")
    ap.add_argument("--save", default="/save_mix1")
    ap.add_argument("--conds", default="TAB,M20,M35")
    ap.add_argument("--labels", default="mix0,mix0.2,mix0.35")
    ap.add_argument("--prefix", default="mix")
    ap.add_argument("--head", default="つまみ")
    ap.add_argument("--what", default="『目標らしさ ↔ 聞き取りやすさ』のつまみだけ")
    ap.add_argument("--h1", default="つまみの耳(男声 → 目標の女声)")
    ap.add_argument("--q1", default="参照の人らしい順")
    ap.add_argument("--q2", default="聞き取りやすい順")
    ap.add_argument("--ref", choices=["cen", "src"], default="cen")
    ap.add_argument("--rdesc", default="")
    ap.add_argument("--body", default="")
    a = ap.parse_args()
    conds = a.conds.split(",")
    labels = dict(zip(conds, a.labels.split(",")))
    d = Path(a.wav_dir)
    files = {}
    for f in d.glob("p*_t*_*.wav"):
        m = re.match(r"p(\d+)_t(\d+)_(.+)\.wav", f.name)
        files[(int(m.group(1)), m.group(3))] = (int(m.group(2)), f)
    rng = random.Random(7)
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    key = {"wav_dir": str(d), "trials": {}}
    trials = []
    by_t: dict = {}
    for (pi_, c_), (t_, f_) in files.items():
        by_t.setdefault(t_, set()).add(pi_)
    pairs = sorted((min(v), t_) for t_, v in by_t.items())
    for k, (pi, t) in enumerate(rng.sample(pairs, min(a.n, len(pairs)))):
        clips = {nm: rd(files[(pi, nm)][1]) for nm in conds}
        ref = rd(d / f"cen_{t:03d}.wav") if a.ref == "cen" else rd(files[(pi, "SRC")][1])
        n = min(len(v) for v in list(clips.values()) + [ref])
        allc = {nm: v[:n] for nm, v in clips.items()}
        allc["ref"] = ref[:n]
        normed = norm_trial(allc, allc[conds[0]])
        tid = f"{a.prefix}_{k + 1:02d}"
        names = list(conds)
        random.Random(tid).shuffle(names)
        td = out / tid
        td.mkdir(exist_ok=True)
        sf.write(td / "R.wav", normed["ref"], SR)
        key["trials"][tid] = {"target": t, "pair": pi, "map": {}}
        for jj, nm in enumerate(names):
            sf.write(td / f"{'ABCD'[jj]}.wav", normed[nm], SR)
            key["trials"][tid]["map"]["ABCD"[jj]] = labels[nm]
        trials.append({"id": tid, "set": a.prefix, "head": a.head, "letters": list("ABCD"[: len(conds)])})
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    h = (EB / "rvoc_ab3/listen.html").read_text()
    h = re.sub(r"<title>.*?</title>", "<title>変換の耳</title>", h, flags=re.S)
    h = h.replace("<h1>出力部(ボコーダ)の写し合成</h1>", "<h1>" + a.h1 + "</h1>")
    rdesc = a.rdesc or ("参照 R(目標の人の実際の声)" if a.ref == "cen" else "参照 R(変換する前の元の男声)")
    body = a.body or ("/".join("ABCD"[: len(conds)]) + " は同じ男声の発話を、同じ目標へ" + a.what + "変えて変換したものです(順番はシャッフル)。")
    intro = ("問い: まず<b>" + rdesc + "</b>を聴いてください。" + body +
             "①<b>" + a.q1 + "</b> ②<b>" + a.q2 + "</b>に並べてください(1 = 最も)。気づき(こもり・ざらつき・声質)はメモへ。音量は試行内で揃えてあります。<br>"
             "操作: ボタンで<b>同じ再生位置のまま</b>切替(ループ再生)。キーボード 数字 1=R," + ",".join(f"{i + 2}={L}" for i, L in enumerate("ABCD"[: len(conds)])) + "・スペース=停止。記入は自動で一時保存され、下の「保存」でサーバへ書き込みます。")
    h = re.sub(r'<div class="mut">問い:.*?</div>', '<div class="mut">' + intro + "</div>", h, count=1, flags=re.S)
    h = re.sub(r"const TRIALS=\[.*?\];const ERR=\[\];const KEY='[^']*';", "const TRIALS=" + json.dumps(trials, ensure_ascii=False) + ";const ERR=[];const KEY='ab_answers_" + a.dir + "_v1';", h, count=1, flags=re.S)
    h = h.replace("t.letters.map(L=>[L,t.id+'/'+L+'.wav'])", "[['R',t.id+'/R.wav','参照 R']].concat(t.letters.map(L=>[L,t.id+'/'+L+'.wav']))")
    h = h.replace("'chorus'", "'sim'").replace("'overall'", "'nat'")
    h = h.replace("'コーラス感が少ない順位'", "'" + a.q1.replace("順", "順位") + "'").replace("'総合で好ましい順位'", "'" + a.q2.replace("順", "順位") + "'")
    h = h.replace("コーラスのみ記入", "近さのみ記入").replace("コーラス順位 記入済み", "近さの順位 記入済み")
    h = h.replace("/save_rvoc3", a.save)
    assert "rvoc_ab3" not in h.replace("../err_bands", "")
    (out / "listen.html").write_text(h)
    (out / "_key_page.json").write_text(json.dumps({t: {"map": {L: "" for L in v["map"]}} for t, v in key["trials"].items()}, ensure_ascii=False))
    print("ear page ->", out, "trials", len(trials), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
