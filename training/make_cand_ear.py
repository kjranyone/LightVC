"""候補の耳(オーナー判断 2026-10-17・解析の関門 (ii) hf の余裕だけ未達を承知の例外): 開発用の組 dev・dev2・dev3 から無作為に n 試行。
各試行 = 参照 R(目標の別の録音・声域をそろえた中心 cen_reg 8 秒)+ A/B/C(シャッフル): 候補 RRQSDC19LE・kNN-VC・R なしの製品 G03S10ZH(同じ源・同じ目標)。
問い 1 =『参照の人に近い順』・問い 2 =『自然さ(不自然さの少ない順)』・気づきはメモ。音量は試行内で候補に揃え共通の減衰だけ(render_d1_ab.norm_trial)。鍵 = _key_聴取後に開く.json(サーバは配信しない)。
    uv run python make_cand_ear.py --root <dev の親> --dir cand_ab1 --n 8
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
SETS = ("dev", "dev2", "dev3")
FORMS = {"cand": ("d139", "RRQSDC19LE"), "knn": ("d135_real", "KNN"), "base": ("d114", "G03S10ZH")}


def rd(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(1)[: SEC * SR].astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dir", default="cand_ab1")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--save", default="/save_cand1")
    a = ap.parse_args()
    RT = Path(a.root)
    pool = []
    for D in SETS:
        ok = {m["ti"] for m in json.loads((RT / D / "cen_reg/meta.json").read_text()) if m["ok"]}
        for f in sorted((RT / D / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            if ti in ok and all((RT / D / d / f"{f.stem[:-4]}_{c}.wav").exists() for d, c in FORMS.values()):
                pool.append((D, pi, ti, f.stem[:-4]))
    rng = random.Random(1017)
    tg = sorted({(D, ti) for D, _, ti, _ in pool})
    pick = rng.sample(tg, a.n)
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    key = {"forms": {k: v[1] for k, v in FORMS.items()}, "root": str(RT), "trials": {}}
    trials = []
    for j, (D, ti) in enumerate(pick):
        D_, pi, _, k = rng.choice([p for p in pool if p[0] == D and p[2] == ti])
        clips = {nm: rd(RT / D / d / f"{k}_{c}.wav") for nm, (d, c) in FORMS.items()}
        ref = rd(RT / D / "cen_reg" / f"cen_{ti:03d}.wav")
        n = min(len(v) for v in list(clips.values()) + [ref])
        allc = {nm: v[:n] for nm, v in clips.items()}
        allc["ref"] = ref[:n]
        normed = norm_trial(allc, allc["cand"])
        tid = f"cand_{j + 1:02d}"
        names = list(FORMS)
        random.Random(tid).shuffle(names)
        td = out / tid
        td.mkdir(exist_ok=True)
        sf.write(td / "R.wav", normed["ref"], SR)
        key["trials"][tid] = {"set": D, "target": ti, "pair": pi, "key": k, "map": {}}
        for i, nm in enumerate(names):
            sf.write(td / f"{'ABC'[i]}.wav", normed[nm], SR)
            key["trials"][tid]["map"]["ABC"[i]] = nm
        trials.append({"id": tid, "set": "cand", "head": "候補", "letters": ["A", "B", "C"]})
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    h = (EB / "conv_ab1/listen.html").read_text()
    h = re.sub(r"<title>.*?</title>", "<title>候補の耳</title>", h, flags=re.S)
    h = re.sub(r"<h1>.*?</h1>", "<h1>候補の耳(男声 → 目標の女声・2026-10-17)</h1>", h, count=1, flags=re.S)
    intro = ("問い: まず<b>参照 R(目標の人の別の録音・同じ声域)</b>を聴いてください。次に A/B/C を、①<b>参照の人に近い順</b> ②<b>自然さ(不自然さが少ない順)</b>に並べてください(1 = 最も近い/最も自然)。"
             "3 本は同じ男声の入力を同じ目標へ変換した 3 つの方式をシャッフルしたものです(正解は聴取後に開く鍵)。話している内容は参照と違います。気づき(声質・ガビ・こもり・f0・明るさ)はメモへ。音量は試行内で揃えてあります。<br>"
             "操作: ボタンで<b>同じ再生位置のまま</b>切替(ループ再生)。キーボード 数字 1=R,2=A,3=B,4=C・スペース=停止。記入は自動で一時保存され、下の「保存」でサーバへ書き込みます。")
    h = re.sub(r'<div class="mut">問い:.*?</div>', '<div class="mut">' + intro + "</div>", h, count=1, flags=re.S)
    h = re.sub(r"const TRIALS=\[.*?\];const ERR=\[\];const KEY='[^']*';", "const TRIALS=" + json.dumps(trials, ensure_ascii=False) + ";const ERR=[];const KEY='ab_answers_cand_ab1_v1';", h, count=1, flags=re.S)
    h = h.replace("/save_conv1", a.save)
    assert "conv_ab1" not in h and "save_conv1" not in h, [m.start() for m in re.finditer("conv_ab1|save_conv1", h)][:3]
    (out / "listen.html").write_text(h)
    print("ear page ->", out, "trials", len(trials), [(t["set"], t["target"]) for t in key["trials"].values()], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
