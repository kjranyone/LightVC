"""耳の較正(artic_vc §8e・rev1 レビュー 2 巡目の T1・T2)。基準のある問いのみ・盲検・試行内で音量を揃える。
T1(ABX・6 試行): X = 目標の実音声(中心の埋め込み用の別発話から 8s)・A/B = 本人の別発話の写し合成 と 学習外 LDA が最も紛らわしいとした別人の写し合成。
T2(6 試行): 同じ発話の 2 本(COPY 対 C0LP4 を 3・COPY 対 TAB を 3)で『同じ人に聞こえるか』『どちらが自然か』。
素材 = d19/self(COPY・C0LP4)・d16/self(COPY・TAB)と d4b の --dump の得点。鍵は _key_聴取後に開く.json(サーバは配信しない)。
    uv run python make_calib_ear.py --d19 <scratchpad>/d19/self --d16 <scratchpad>/d16/self --scores <scratchpad>/d19_scores.npz --dir calib_ear1
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


def rd(p: Path, sec: float = 8.0) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR
    return x.mean(1)[: int(sec * SR)].astype(np.float64)


PAGE = """<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>耳の較正</title><style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:820px;margin:0 auto;padding:20px 16px 110px}h1{font-size:20px;margin:0 0 6px}h2{font-size:16px;margin:24px 0 6px}
.mut{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 14px;cursor:pointer;margin:2px}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}.q{margin:6px 0}.q label{margin-right:14px;white-space:nowrap}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
</style></head><body><main>
<h1>耳の較正(約 8 分)</h1>
<div class="mut">音量は試行内で揃えてあります。各ボタンで再生(もう一度押すと停止)。答えは自動で一時保存され、最後に「保存」でサーバへ書き込みます。<br>
<b>T1</b>: X(ある人の実際の声)を聴いて、<b>A と B のどちらが X と同じ人か</b>を選んでください。A・B とも合成音です(話している内容は X と違います)。<br>
<b>T2</b>: 同じ発話の 2 本です。<b>同じ人に聞こえるか</b> と <b>どちらが自然か</b> を選んでください。</div>
<div id="trials"></div></main>
<footer><span id="prog" class="mut"></span><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const TRIALS=__TRIALS__;const KEY='calib_ear1_v1';
let ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
let cur=null;
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}prog()}
function playBtn(src,label){const b=document.createElement('button');b.textContent=label;const a=new Audio(src);a.preload='auto';
 b.onclick=()=>{if(cur&&cur.a===a&&!a.paused){a.pause();b.classList.remove('on');cur=null;return}if(cur){cur.a.pause();cur.b.classList.remove('on')}a.currentTime=0;a.play();b.classList.add('on');cur={a,b};a.onended=()=>b.classList.remove('on')};return b}
function radio(id,q,name,opts){const d=document.createElement('div');d.className='q';d.innerHTML='<span>'+q+': </span>';
 for(const [v,l] of opts){const lab=document.createElement('label');const r=document.createElement('input');r.type='radio';r.name=id+'_'+name;r.value=v;
  r.checked=((ans[id]||{})[name])===v;r.onchange=()=>{ans[id]=ans[id]||{};ans[id][name]=v;persist()};lab.append(r,' '+l);d.appendChild(lab)}return d}
const root=document.getElementById('trials');let grp='';
TRIALS.forEach((t,i)=>{if(t.set!==grp){grp=t.set;const h=document.createElement('h2');h.textContent=t.set==='T1'?'T1: X と同じ人はどちら?':'T2: 同じ発話の 2 本';root.appendChild(h)}
 const c=document.createElement('div');c.className='card';c.innerHTML='<div class="mut">'+(i+1)+'. '+t.id+'</div>';
 const pl=document.createElement('div');
 if(t.set==='T1'){pl.append(playBtn(t.id+'/X.wav','X(実際の声)'),playBtn(t.id+'/A.wav','A'),playBtn(t.id+'/B.wav','B'));c.append(pl,radio(t.id,'X と同じ人','pick',[['A','A'],['B','B'],['?','わからない']]))}
 else{pl.append(playBtn(t.id+'/A.wav','A'),playBtn(t.id+'/B.wav','B'));c.append(pl,radio(t.id,'同じ人に聞こえるか','same',[['yes','はい'],['no','いいえ'],['?','わからない']]),radio(t.id,'どちらが自然か','nat',[['A','A'],['B','B'],['=','同じ']]))}
 root.appendChild(c)});
function prog(){const n=TRIALS.filter(t=>{const v=ans[t.id]||{};return t.set==='T1'?v.pick:(v.same&&v.nat)}).length;document.getElementById('prog').textContent='記入済み '+n+' / '+TRIALS.length}
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');const body=JSON.stringify({answers:ans,client_saved_at:new Date().toISOString()});
 try{const r=await fetch('__SAVE__',{method:'POST',headers:{'Content-Type':'application/json'},body});const j=await r.json();m.textContent=j.ok?'保存しました':'保存失敗'}
 catch(e){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download='answers_ear.json';a.click();m.textContent='ファイルとしてダウンロードしました'}};
prog();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d19", required=True)
    ap.add_argument("--d16", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--dir", default="calib_ear1")
    ap.add_argument("--save", default="/save_calib1")
    a = ap.parse_args()
    d19, d16 = Path(a.d19), Path(a.d16)
    z = np.load(a.scores)
    names, pis, tis, conds, scores = z["name"], z["pi"], z["ti"], z["cond"], z["score"]
    rng = random.Random(11)
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    key: dict = {"trials": {}}
    trials = []
    copy_rows = [i for i in range(len(names)) if conds[i] == "COPY"]
    by_t: dict = {}
    for i in copy_rows:
        by_t.setdefault(int(tis[i]), []).append(i)
    tlist = sorted(t for t, v in by_t.items() if len(v) >= 2)
    for k, t in enumerate(rng.sample(tlist, 6)):
        ia = by_t[t][0]
        sc = scores[ia].copy(); sc[t] = -9
        other = int(sc.argmax())
        ib = by_t.get(other, [None])[0]
        if ib is None:
            continue
        X = rd(d19 / f"cen_{t:03d}.wav")
        clips = {"same": rd(d19 / str(names[ia])), "other": rd(d19 / str(names[ib]))}
        n = min(len(X), *(len(v) for v in clips.values()))
        allc = {"X": X[:n], **{kk: v[:n] for kk, v in clips.items()}}
        normed = norm_trial(allc, allc["same"])
        tid = f"t1_{k + 1:02d}"
        order = ["same", "other"]; random.Random(tid).shuffle(order)
        td = out / tid; td.mkdir(exist_ok=True)
        sf.write(td / "X.wav", normed["X"].astype(np.float32), SR)
        key["trials"][tid] = {"target": t, "confusable": other, "A": order[0], "B": order[1], "lda_score_same": float(scores[ia][t]), "lda_score_other_centroid": float(scores[ia][other])}
        for L, nm in zip("AB", order):
            sf.write(td / f"{L}.wav", normed[nm].astype(np.float32), SR)
        trials.append({"id": tid, "set": "T1"})
    t2 = []
    for i in rng.sample(copy_rows, 6):
        pi, ti = int(pis[i]), int(tis[i])
        t2.append((pi, ti))
    for k, (pi, ti) in enumerate(t2):
        if k < 3:
            deg, src = "C0LP4", d19
        else:
            deg, src = "TAB", d16
        fa = src / f"p{pi:04d}_t{ti:03d}_COPY.wav"
        fb = src / f"p{pi:04d}_t{ti:03d}_{deg}.wav"
        if not (fa.exists() and fb.exists()):
            continue
        clips = {"COPY": rd(fa), deg: rd(fb)}
        n = min(len(v) for v in clips.values())
        clips = {kk: v[:n] for kk, v in clips.items()}
        normed = norm_trial(clips, clips["COPY"])
        tid = f"t2_{k + 1:02d}"
        order = ["COPY", deg]; random.Random(tid).shuffle(order)
        td = out / tid; td.mkdir(exist_ok=True)
        for L, nm in zip("AB", order):
            sf.write(td / f"{L}.wav", normed[nm].astype(np.float32), SR)
        key["trials"][tid] = {"pair": [pi, ti], "A": order[0], "B": order[1]}
        trials.append({"id": tid, "set": "T2"})
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (out / "listen.html").write_text(PAGE.replace("__TRIALS__", json.dumps(trials)).replace("__SAVE__", a.save))
    print("page ->", out, "trials", len(trials), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
