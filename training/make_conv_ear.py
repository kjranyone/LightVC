"""変換の耳(製品候補 = 基準経路・F2_2・参照 20s): 参照(目標の実音声)を聴いた上で、3 本を順位づける。基準のある問い。
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
    ap.add_argument("--save", default="/save_conv1")
    a = ap.parse_args()
    d = Path(a.wav_dir)
    files = {}
    for f in d.glob("p*_t*_*.wav"):
        m = re.match(r"p(\d+)_t(\d+)_(.+)\.wav", f.name)
        files[(int(m.group(1)), m.group(3))] = (int(m.group(2)), f)
    pis = sorted({k[0] for k in files})
    tis = sorted({v[0] for v in files.values()})
    S = len(tis)
    rng = random.Random(7)
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    pick_t = rng.sample(tis, a.n)
    key = {"cond": a.cond, "wav_dir": str(d), "trials": {}}
    trials = []
    for k, t in enumerate(pick_t):
        pset = [pi for pi in pis if files[(pi, a.cond)][0] == t]
        pi = rng.choice(pset)
        t2 = rng.choice([x for x in tis if x != t])
        pj = rng.choice([p for p in pis if files[(p, a.cond)][0] == t2])
        clips = {"same": rd(files[(pi, a.cond)][1]), "other": rd(files[(pj, a.cond)][1]), "source": rd(files[(pi, "SRC")][1])}
        ref = rd(d / f"cen_{t:02d}.wav")
        n = min(len(v) for v in list(clips.values()) + [ref])
        allc = {nm: v[:n] for nm, v in clips.items()}
        allc["ref"] = ref[:n]
        normed = norm_trial(allc, allc["same"])
        tid = f"conv_{k + 1:02d}"
        names = ["same", "other", "source"]
        random.Random(tid).shuffle(names)
        td = out / tid
        td.mkdir(exist_ok=True)
        sf.write(td / "R.wav", normed["ref"], SR)
        key["trials"][tid] = {"target": t, "other_target": t2, "pair": pi, "other_pair": pj, "map": {}}
        for j, nm in enumerate(names):
            sf.write(td / f"{'ABC'[j]}.wav", normed[nm], SR)
            key["trials"][tid]["map"]["ABC"[j]] = nm
        trials.append({"id": tid, "set": "conv", "head": "変換", "letters": ["A", "B", "C"]})
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    h = (EB / "rvoc_ab3/listen.html").read_text()
    h = re.sub(r"<title>.*?</title>", "<title>変換の耳</title>", h, flags=re.S)
    h = h.replace("<h1>出力部(ボコーダ)の写し合成</h1>", "<h1>変換の耳(男声 → 目標の女声)</h1>")
    intro = ("問い: まず<b>参照 R(目標の人の実際の声)</b>を聴いてください。次に A/B/C を、①<b>参照の人に近い順</b> ②<b>自然さ(不自然さが少ない順)</b>に並べてください(1 = 最も近い/最も自然)。"
             "3 本は『目標どおりに変換した声』『別の人を目標に変換した声』『元の男声』をシャッフルしたものです(正解は聴取後に開く鍵)。話している内容は参照と違います。気づき(声質・不自然さ)はメモへ。音量は試行内で揃えてあります。<br>"
             "操作: ボタンで<b>同じ再生位置のまま</b>切替(ループ再生)。キーボード 数字 1=R,2=A,3=B,4=C・スペース=停止。記入は自動で一時保存され、下の「保存」でサーバへ書き込みます。")
    h = re.sub(r'<div class="mut">問い:.*?</div>', '<div class="mut">' + intro + "</div>", h, count=1, flags=re.S)
    h = re.sub(r"const TRIALS=\[.*?\];const ERR=\[\];const KEY='[^']*';", "const TRIALS=" + json.dumps(trials, ensure_ascii=False) + ";const ERR=[];const KEY='ab_answers_conv_ab1_v1';", h, count=1, flags=re.S)
    h = h.replace("t.letters.map(L=>[L,t.id+'/'+L+'.wav'])", "[['R',t.id+'/R.wav','参照 R']].concat(t.letters.map(L=>[L,t.id+'/'+L+'.wav']))")
    h = h.replace("'chorus'", "'sim'").replace("'overall'", "'nat'")
    h = h.replace("'コーラス感が少ない順位'", "'参照の人に近い順位'").replace("'総合で好ましい順位'", "'自然さの順位(不自然さが少ない順)'")
    h = h.replace("コーラスのみ記入", "近さのみ記入").replace("コーラス順位 記入済み", "近さの順位 記入済み")
    h = h.replace("/save_rvoc3", a.save)
    assert "rvoc_ab3" not in h.replace("../err_bands", "")
    (out / "listen.html").write_text(h)
    (out / "_key_page.json").write_text(json.dumps({t: {"map": {L: "" for L in v["map"]}} for t, v in key["trials"].items()}, ensure_ascii=False))
    print("ear page ->", out, "trials", len(trials), "targets", S, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
