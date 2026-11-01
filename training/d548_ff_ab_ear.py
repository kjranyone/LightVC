"""D548: 置き換えの候補 P52OP(gate_ff_v4・オーナー判断 10-23)と受け入れ済みの P43OP を同じ試行で盲検の A / B に。試行 = d514 と同じ(devff5 の操作者 14 人から 1 組ずつ・Random(514))。
各試行 = 入力・目標の参照・A・B(P43OP と P52OP の順は試行ごとに Random(548))。音量 = 4 本を同じ RMS(−23 dBFS・有音)・ピーク 0.95 超なら 4 本に共通の減衰。
問い = 自然さ・目標らしさ・聞き取りやすさ・ジャギー / フェーザー / かすれの少なさ(それぞれ A / B / 同じ)。出力 = results/earbattery/ff_ab_ear/。保存 = serve_ab.py の /save_ffab。
    uv run python d548_ff_ab_ear.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
S = Path("/home/kojirotanaka/lightvc_scratch_20261013/devff5")
OUT = ROOT / "results/earbattery/ff_ab_ear"
SR = 48000


def active_rms(x: np.ndarray) -> float:
    n = len(x) // 480
    e = (x[: n * 480].reshape(n, 480) ** 2).mean(1)
    thr = e.max() * 10 ** (-40 / 10)
    return float(np.sqrt(e[e > thr].mean() + 1e-12))


def best8(x: np.ndarray) -> np.ndarray:
    w = 8 * SR
    if len(x) <= w:
        return np.pad(x, (0, w - len(x)))
    hop = SR // 4
    best, bo = -1.0, 0
    for o in range(0, len(x) - w + 1, hop):
        r = float((x[o:o + w] ** 2).mean())
        if r > best:
            best, bo = r, o
    return x[bo:bo + w]


def load(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    assert sr == SR, (p, sr)
    return x.mean(1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    by: dict = {}
    for r in pairs:
        by.setdefault(r["spk"], []).append(r)
    rng = random.Random(514)
    chosen = [rng.choice(sorted(v, key=lambda r: r["pi"])) for _, v in sorted(by.items())]
    rng.shuffle(chosen)
    key, trials = {}, []
    off = random.Random(548).randrange(2)
    for i, r in enumerate(chosen, 1):
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        sysw = {"P43OP": load(S / "gate/CAND" / f"{k}_CAND.wav")[: 8 * SR], "P52OP": load(S / "gate4/P52" / f"{k}_P52.wav")[: 8 * SR]}
        order = ["P43OP", "P52OP"] if (i + off) % 2 else ["P52OP", "P43OP"]
        xs = {"in": load(S / "d56" / f"{k}_SRC.wav")[: 8 * SR], "tgt": best8(load(S / "d37" / f"cen_{r['ti']:03d}.wav")), "A": sysw[order[0]], "B": sysw[order[1]]}
        xs = {n: x * (10 ** (-23 / 20) / active_rms(x)) for n, x in xs.items()}
        pk = max(float(np.abs(x).max()) for x in xs.values())
        g = 0.95 / pk if pk > 0.95 else 1.0
        tid = f"t{i:02d}"
        for n, x in xs.items():
            sf.write(OUT / f"{tid}_{n}.wav", (x * g).astype(np.float32), SR, subtype="PCM_16")
        key[tid] = {"pair": k, "operator": r["spk"], "target": r["target"], "A": order[0], "B": order[1], "common_gain_db": round(20 * np.log10(g), 2)}
        trials.append(tid)
    (OUT / "_key_聴取後に開く.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
    (OUT / "listen.html").write_text(PAGE.replace("__TRIALS__", json.dumps(trials)), encoding="utf-8")
    print("trials", len(trials), "->", OUT, flush=True)
    return 0


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>候補の聞き比べ</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff;--ok:#1e7a3c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117;--ok:#7ddc9a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 120px}h1{font-size:20px;margin:0 0 6px}
.mut{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:8px}.st{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--line);color:var(--mut)}
.st.done{color:var(--ok);border-color:var(--ok)}.pl{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 6px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 14px;cursor:pointer}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.row{display:grid;grid-template-columns:230px 1fr;gap:8px;align-items:center;margin:8px 0}.opts{display:flex;gap:12px;flex-wrap:wrap}.opts label{white-space:nowrap}
textarea{width:100%;min-height:44px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
@media (max-width:560px){.row{grid-template-columns:1fr}}
</style></head><body><main>
<h1>女声 → 女声の候補の聞き比べ(2026-10-23・A / B は試行ごとに順番が入れ替わります)</h1>
<div class="mut">各試行で <b>入力</b>(操作者 = 本物の女声)・<b>目標</b>(変換先の人の別の録音)・<b>A</b>・<b>B</b>(2 つの変換)を聴き比べてください。4 本は同じ位置から切り替わります。音量は試行の中でそろえてあります。A と B のどちらがどの版かは伏せています。途中でやめても構いません。</div>
<div id="trials"></div>
<div class="card"><b>全体</b><div class="row"><div>A / B の差の大きさ</div><div class="opts" id="overall"></div></div>
<textarea id="ocm" placeholder="全体の所見(任意)"></textarea></div>
</main>
<footer><span id="prog" class="mut"></span><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const TRIALS=__TRIALS__;const KEY='ff_ab_ear_v1';
const Q=[["nat","より自然なのは",["A","B","同じ"]],["tgt","より目標の人に聞こえるのは",["A","B","同じ"]],["cont","より聞き取りやすいのは",["A","B","同じ"]],["art","ジャギー・フェーザー・かすれが少ないのは",["A","B","同じ"]]];
let ans={trials:{},overall:null,comment:''};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}prog()}
let cur=null;const P={};
function stopAll(){for(const p of Object.values(P))for(const a of Object.values(p))a.pause();document.querySelectorAll('.pl button.L').forEach(b=>b.classList.remove('on'))}
function play(id,n,btn){const t=cur&&cur.id===id?cur.a.currentTime:0;stopAll();const a=P[id][n];a.currentTime=Math.min(t,(a.duration||8)-0.05);a.loop=true;a.play();cur={id,a};btn.classList.add('on')}
function radio(box,name,opts,val,on){for(const o of opts){const l=document.createElement('label');const r=document.createElement('input');r.type='radio';r.name=name;r.value=o;r.checked=(val===o);r.onchange=()=>on(o);l.append(r,' '+o);box.appendChild(l)}}
function done(id){const v=ans.trials[id]||{};return Q.every(q=>v[q[0]])}
function render(){const root=document.getElementById('trials');TRIALS.forEach((id,i)=>{const c=document.createElement('div');c.className='card';
 const hd=document.createElement('div');hd.className='hd';hd.innerHTML='<b>試行 '+(i+1)+' / '+TRIALS.length+'</b>';const st=document.createElement('span');st.className='st';st.id='st_'+id;hd.appendChild(st);c.appendChild(hd);
 const pl=document.createElement('div');pl.className='pl';P[id]={};for(const [n,lab] of [['in','入力'],['tgt','目標'],['A','A'],['B','B']]){const a=new Audio(id+'_'+n+'.wav');a.preload='auto';P[id][n]=a;const b=document.createElement('button');b.className='L';b.textContent=lab;b.onclick=()=>play(id,n,b);pl.appendChild(b)}
 const s=document.createElement('button');s.textContent='■ 停止';s.onclick=()=>{stopAll();cur=null};pl.appendChild(s);c.appendChild(pl);
 for(const [k,lab,opts] of Q){const r=document.createElement('div');r.className='row';r.innerHTML='<div>'+lab+'</div>';const box=document.createElement('div');box.className='opts';
  radio(box,id+'_'+k,opts,(ans.trials[id]||{})[k],o=>{ans.trials[id]=ans.trials[id]||{};ans.trials[id][k]=o;persist()});r.appendChild(box);c.appendChild(r)}
 const t=document.createElement('textarea');t.placeholder='所見(任意)';t.value=(ans.trials[id]||{}).note||'';t.oninput=()=>{ans.trials[id]=ans.trials[id]||{};ans.trials[id].note=t.value;persist()};c.appendChild(t);root.appendChild(c)});
 radio(document.getElementById('overall'),'overall',['はっきり違う','少し違う','ほぼ同じ'],ans.overall,o=>{ans.overall=o;persist()});
 const oc=document.getElementById('ocm');oc.value=ans.comment||'';oc.oninput=()=>{ans.comment=oc.value;persist()};prog()}
function prog(){let n=0;for(const id of TRIALS){const d=done(id);n+=d;const st=document.getElementById('st_'+id);if(st){st.textContent=d?'回答済み':'未回答';st.classList.toggle('done',d)}}document.getElementById('prog').textContent='回答 '+n+' / '+TRIALS.length}
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');try{const r=await fetch('/save_ffab',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(ans)});m.textContent=r.ok?'保存しました '+new Date().toLocaleTimeString():'保存に失敗しました('+r.status+')'}catch(e){m.textContent='保存に失敗しました'}};
render();
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
