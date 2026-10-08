"""DDSP-VC の耳の試料(Rust 製品経路の CLI で生成)と試聴ページ 2 種。事前登録 results/ddsp_vc/prereg.yaml の fork(PASS 時)。

  results/earbattery/ddsp_chorus/  コーラス有無(盲検 8 本: 変換 6・錨=目標の実音声 1・陽性対照=学習なし DSP 変換 1)
  results/earbattery/ddsp_sim/     目標らしさ(8 組: 参照と変換の組 6・対照=別の目標の変換と組ませた 2)。組ごとに「似ている/やや/似ていない」

    CUDA_VISIBLE_DEVICES= uv run python make_ddsp_ear.py --export ../results/ddsp_vc/export
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

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as AD
from train_ddsp_vc import index, load48

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
TMP = ROOT / "results/ddsp_vc/ear_src"


def med_f0(x: np.ndarray) -> float:
    f, _ = AD.causal_yin(x.astype(np.float64), voi_max=0.25)
    return float(np.median(f[f > 0])) if (f > 0).sum() > 20 else 0.0


def rust(export: Path, ref: Path, enroll: Path, inp: Path, out: Path) -> str:
    r = subprocess.run(["cargo", "run", "--release", "-q", "-p", "lightvc-core", "--example", "ddsp_convert", "--",
                        "--model", str(export), "--ref", str(ref), "--src-enroll", str(enroll), "--in", str(inp), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    return r.stderr.strip().splitlines()[-1]


def level(x: np.ndarray, rms: float = 0.05) -> np.ndarray:
    return x * (rms / max(float(np.sqrt((x ** 2).mean())), 1e-6))


SIM_PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>目標らしさ 盲検</title><style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 90px}h1{font-size:20px;margin:0 0 6px}.mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 12px;cursor:pointer;margin-right:6px}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}label{margin-right:14px;white-space:nowrap}
textarea{width:100%;min-height:44px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;justify-content:center;align-items:center;flex-wrap:wrap}
</style></head><body><main><h1>目標らしさ(盲検・8 組)</h1>
<div class="mut">各組で「参照」(目標の本人の声)と「変換」を聴き比べ、<b>変換が参照と同じ人の声に聞こえるか</b>を付けてください。
声の高さの違いではなく声の質(その人らしさ)で判断してください。音質の問題は「メモ」へ。組の中に、わざと別の人の変換を混ぜた対照があります。</div>
<div id="items"></div><div class="card"><textarea id="note" placeholder="メモ(任意)"></textarea></div></main>
<footer><span id="prog" class="mut"></span><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const IT=__ITEMS__;const KEY='ddsp_sim_v1';let ans={ratings:{},note:''};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}document.getElementById('prog').textContent='記入 '+Object.keys(ans.ratings).length+' / '+IT.length}
let cur=null;function play(src,b){if(cur){cur.a.pause();cur.b.classList.remove('on')}const a=new Audio(src);a.play();b.classList.add('on');cur={a,b};a.onended=()=>b.classList.remove('on')}
const box=document.getElementById('items');for(const it of IT){const d=document.createElement('div');d.className='card';d.innerHTML='<b>'+it.id+'</b> ';
 for(const [lab,src] of [['参照',it.ref],['変換',it.conv]]){const b=document.createElement('button');b.textContent='▶ '+lab;b.onclick=()=>play(src,b);d.appendChild(b)}
 const r=document.createElement('div');for(const v of ['似ている','やや','似ていない']){const l=document.createElement('label');const i=document.createElement('input');i.type='radio';i.name=it.id;i.checked=ans.ratings[it.id]===v;i.onchange=()=>{ans.ratings[it.id]=v;persist()};l.append(i,' '+v);r.appendChild(l)}
 d.appendChild(r);box.appendChild(d)}
const nt=document.getElementById('note');nt.value=ans.note||'';nt.oninput=()=>{ans.note=nt.value;persist()};
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');const body=JSON.stringify({...ans,client_saved_at:new Date().toISOString()});
 try{const r=await fetch('/save_ddsp_sim',{method:'POST',headers:{'Content-Type':'application/json'},body});const j=await r.json();m.textContent=j.ok?'保存しました → '+j.path:'保存失敗'}
 catch(e){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download='ddsp_sim_answers.json';a.click();m.textContent='サーバに届かないためダウンロードしました'}};
persist();
</script></body></html>
"""


AB_PAGE = SIM_PAGE.replace("<title>目標らしさ 盲検</title>", "<title>A/B 品質 盲検</title>") \
    .replace("<h1>目標らしさ(盲検・8 組)</h1>", "<h1>どちらが良いか(盲検 A/B)</h1>") \
    .replace("各組で「参照」(目標の本人の声)と「変換」を聴き比べ、<b>変換が参照と同じ人の声に聞こえるか</b>を付けてください。\n声の高さの違いではなく声の質(その人らしさ)で判断してください。音質の問題は「メモ」へ。組の中に、わざと別の人の変換を混ぜた対照があります。",
             "同じ男声を同じ目標へ変換した 2 通り(1・2)です。<b>総合的に良い方</b>(自然さ・雑音の少なさ・明瞭さ・目標らしさ)を選んでください。参照は目標の本人の声です。") \
    .replace("for(const [lab,src] of [['参照',it.ref],['変換',it.conv]])", "for(const [lab,src] of [['参照',it.ref],['1',it.one],['2',it.two]])") \
    .replace("for(const v of ['似ている','やや','似ていない'])", "for(const v of ['1が良い','同等','2が良い'])") \
    .replace("const KEY='ddsp_sim_v1'", "const KEY='ddsp_ab_v1'").replace("'/save_ddsp_sim'", "'/save_ddsp_ab'").replace("ddsp_sim_answers.json", "ddsp_ab_answers.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=str(ROOT / "results/ddsp_vc/export"))
    ap.add_argument("--export_b", default=None, help="A/B 比較の相手(例: 段 G 前の ckpt の書き出し)")
    ap.add_argument("--eb", default=str(EB))
    ap.add_argument("--tmp", default=str(TMP))
    a = ap.parse_args()
    export = Path(a.export).resolve()
    eb_out, tmp = Path(a.eb), Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    spk, tr, ev = index()
    targets = []
    for k in [k for k in ev if k.startswith("real_female")]:
        its = spk[k]
        ref = load48(its[0][1])[:3 * 48000]
        f = med_f0(ref)
        if 190 <= f <= 330 and len(ref) >= 2 * 48000:
            targets.append((k.split("/")[-1][:8], ref, f))
        if len(targets) == 3:
            break
    info = (VC.parent / "speaker-info.txt").read_text().splitlines()[1:]
    males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    sources = []
    for s in males[2::7]:
        p = VC / s / f"{s}_040.wav"
        if not p.is_file():
            continue
        x = load48(p)[:6 * 48000]
        f = med_f0(x)
        if 90 <= f <= 160:
            enr = np.concatenate([load48(VC / s / f"{s}_{u:03d}.wav") for u in (41, 42, 43) if (VC / s / f"{s}_{u:03d}.wav").is_file()])
            sources.append((s, x, enr, f))
        if len(sources) == 2:
            break
    import glob
    jd = sorted(glob.glob(str(ROOT / "data/male_tts_corpus/*")))[5]
    jw = sorted(glob.glob(jd + "/*neutral*.wav"))
    sources.append(("ja_" + Path(jd).name, load48(Path(jw[0]))[:6 * 48000], np.concatenate([load48(Path(w)) for w in jw[1:4]]), 0.0))
    conv = {}
    logs = {}
    for sn, x, enr, _ in sources:
        soundfile.write(tmp / f"{sn}.wav", x, 48000, subtype="FLOAT")
        soundfile.write(tmp / f"{sn}_enroll.wav", enr, 48000, subtype="FLOAT")
        for tn, ref, _ in targets:
            soundfile.write(tmp / f"REF_{tn}.wav", ref, 48000, subtype="FLOAT")
            out = tmp / f"{sn}__{tn}.wav"
            logs[f"{sn}__{tn}"] = rust(export, tmp / f"REF_{tn}.wav", tmp / f"{sn}_enroll.wav", tmp / f"{sn}.wav", out)
            conv[(sn, tn)] = soundfile.read(out)[0].astype(np.float32)
            print(sn, tn, logs[f"{sn}__{tn}"], flush=True)
    key = json.loads((EB / "dspvc_p0/_key_聴取後に開く.json").read_text())
    pc_x = next(x for x, n in key["map"].items() if n.endswith("|pitch_vtl"))
    pos_ctrl = soundfile.read(EB / "dspvc_p0" / f"{pc_x}.wav")[0].astype(np.float32)
    rng = random.Random("ddsp_ear_20260927")
    pairs = [(s[0], t[0]) for s in sources for t in targets[:2]]
    clips = {f"conv|{s}|{t}": conv[(s, t)] for s, t in pairs}
    clips["anchor|REF|" + targets[0][0]] = targets[0][1]
    clips["pos_ctrl|dspvc_p0_pitch_vtl"] = pos_ctrl
    d = eb_out / "ddsp_chorus"
    d.mkdir(parents=True, exist_ok=True)
    names = list(clips)
    rng.shuffle(names)
    lv = {k: level(v) for k, v in clips.items()}
    pk = max(float(np.abs(v).max()) for v in lv.values())
    g = 0.95 / pk if pk > 0.95 else 1.0
    kc = {"map": {}, "rust_logs": logs, "prereg": "results/ddsp_vc/prereg.yaml"}
    for i, nm in enumerate(names):
        soundfile.write(d / f"X{i + 1}.wav", (lv[nm] * g).astype(np.float32), 48000)
        kc["map"][f"X{i + 1}"] = nm
    (d / "_key_聴取後に開く.json").write_text(json.dumps(kc, indent=1, ensure_ascii=False))
    ds = eb_out / "ddsp_sim"
    ds.mkdir(parents=True, exist_ok=True)
    items = [("match", s[0], t[0], t[0]) for s in sources for t in targets[:2]]
    items += [("control", sources[0][0], targets[1][0], targets[0][0]), ("control", sources[1][0], targets[2][0], targets[1][0])]
    rng.shuffle(items)
    ks = {"items": {}, "prereg": "results/ddsp_vc/prereg.yaml"}
    page = []
    for i, (kind, s, tconv, tref) in enumerate(items):
        iid = f"Q{i + 1}"
        cv = conv[(s, tconv)]
        rf = [t[1] for t in targets if t[0] == tref][0]
        soundfile.write(ds / f"{iid}_conv.wav", (level(cv) * 0.9 / max(1.0, float(np.abs(level(cv)).max()) / 0.95)).astype(np.float32), 48000)
        soundfile.write(ds / f"{iid}_ref.wav", (level(rf) * 0.9 / max(1.0, float(np.abs(level(rf)).max()) / 0.95)).astype(np.float32), 48000)
        ks["items"][iid] = {"kind": kind, "src": s, "conv_target": tconv, "ref_target": tref}
        page.append({"id": iid, "ref": f"{iid}_ref.wav", "conv": f"{iid}_conv.wav"})
    (ds / "_key_聴取後に開く.json").write_text(json.dumps(ks, indent=1, ensure_ascii=False))
    (ds / "listen.html").write_text(SIM_PAGE.replace("__ITEMS__", json.dumps(page)))
    if a.export_b:
        dab = eb_out / "ddsp_ab"
        dab.mkdir(parents=True, exist_ok=True)
        kab = {"items": {}, "A": str(export), "B": a.export_b}
        page_ab = []
        for i, (s_, t_) in enumerate([(s[0], t[0]) for s in sources for t in targets[:2]]):
            outb = tmp / f"B_{s_}__{t_}.wav"
            rust(Path(a.export_b).resolve(), tmp / f"REF_{t_}.wav", tmp / f"{s_}_enroll.wav", tmp / f"{s_}.wav", outb)
            ya, yb = conv[(s_, t_)], soundfile.read(outb)[0].astype(np.float32)
            first_a = rng.random() < 0.5
            iid = f"P{i + 1}"
            for nm, sig in (("1", ya if first_a else yb), ("2", yb if first_a else ya)):
                soundfile.write(dab / f"{iid}_{nm}.wav", (level(sig) * 0.9 / max(1.0, float(np.abs(level(sig)).max()) / 0.95)).astype(np.float32), 48000)
            soundfile.write(dab / f"{iid}_ref.wav", (level([t[1] for t in targets if t[0] == t_][0]) * 0.9).astype(np.float32), 48000)
            kab["items"][iid] = {"src": s_, "tgt": t_, "1": "A" if first_a else "B", "2": "B" if first_a else "A"}
            page_ab.append({"id": iid, "ref": f"{iid}_ref.wav", "one": f"{iid}_1.wav", "two": f"{iid}_2.wav"})
        (dab / "_key_聴取後に開く.json").write_text(json.dumps(kab, indent=1, ensure_ascii=False))
        (dab / "listen.html").write_text(AB_PAGE.replace("__ITEMS__", json.dumps(page_ab)))
    if eb_out != EB:
        print("ear pages (test) ->", d, ds, flush=True)
        return 0
    subprocess.run([sys.executable, "make_probe_page.py", "--dir", "ddsp_chorus", "--save", "/save_ddsp_chorus", "--store", "ddsp_chorus_v1",
                    "--intro", "男声を目標の女声へ変換した音声などが混ざった 8 本です(対照を含む)。<b>各クリップにコーラス/フェーザー感(揺れる・うねる・複数人が重なる感じ)があるか</b>を「有・微・無」で付けてください。音質全般の気づきはメモへ。音量は揃えてあります。<br>ボタン(またはキー1〜8)で切替・ループ再生、スペースで停止。"],
                   cwd=Path(__file__).parent, check=True)
    print("ear pages ->", d, ds, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
