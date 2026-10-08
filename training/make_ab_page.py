"""盲検A/B試聴ページ results/earbattery/<dir>/listen.html を生成(鍵の対応表は埋め込まない)。

    uv run python make_ab_page.py                     # d1_ab(既定)
    uv run python make_ab_page.py --dir nrft_ab --order nrft --save /save_nrft --no-err \
        --title "decoder頑健化FT 盲検" --intro "各試行に元音声(加工なし)が1本混ざっています。"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
ERR = [("GT decode(錨・コーラスなし)", "gt_0005e65d3f11f99d_00002260.wav"),
       ("s11 出力そのもの", "s11_cfm_melin_0005e65d3f11f99d_00002260_full.wav"),
       ("s11 から16–50Hz誤差だけ除去", "s11_cfm_melin_0005e65d3f11f99d_00002260_full_minus_high.wav"),
       ("GT + s11の16–50Hz誤差だけ", "s11_cfm_melin_0005e65d3f11f99d_00002260_high.wav"),
       ("GT + s11の0–4Hz誤差だけ", "s11_cfm_melin_0005e65d3f11f99d_00002260_low.wav"),
       ("GT + s11の4–16Hz誤差だけ", "s11_cfm_melin_0005e65d3f11f99d_00002260_mid.wav")]

PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff;--warn:#b3261e;--ok:#1e7a3c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117;--warn:#ff8a80;--ok:#7ddc9a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 120px}
h1{font-size:20px;margin:0 0 6px}h2{font-size:16px;margin:28px 0 8px}
.mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.card.active{border-color:var(--acc)}
.hd{display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap}
.tid{font-weight:600;font-variant-numeric:tabular-nums;word-break:break-all}
.st{font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid var(--line);color:var(--mut)}
.st.done{color:var(--ok);border-color:var(--ok)}
.pl{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 6px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 14px;cursor:pointer}
button.L{min-width:52px;font-weight:700;font-size:16px}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
input[type=range]{width:100%}
.row{display:grid;grid-template-columns:150px 1fr;gap:8px;align-items:center;margin:6px 0}
.sel{display:flex;gap:10px;flex-wrap:wrap}.sel label{white-space:nowrap}
select{font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:2px 4px}
textarea{width:100%;min-height:44px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
.warn{color:var(--warn);font-size:13px;min-height:1em}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
@media (max-width:560px){.row{grid-template-columns:1fr}}
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="mut">問い: <b>コーラス/フェーザー感(高域が多重化・うなる感じ)が少ないのはどれか</b>。__INTRO__音量は試行内で揃えてあります。<br>
操作: 文字ボタンで<b>同じ再生位置のまま</b>切替(ループ再生)。キーボード 数字1,2,3…=A,B,C…・スペース=停止(最後に触った試行に効く)。順位は 1=最もコーラスが少ない/最も好ましい。記入は自動で一時保存され、下の「保存」でサーバへ書き込みます。</div>
<div id="trials"></div>
__ERRHTML__
</main>
<footer><span id="prog" class="mut"></span><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const TRIALS=__TRIALS__;const ERR=__ERR__;const KEY='__LSKEY__';
let ans={trials:{},err_note:''};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}
let cur=null,active=null;const players={};
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}prog()}
function fmt(t){if(!isFinite(t))return'0:00';const m=Math.floor(t/60),s=Math.floor(t%60);return m+':'+String(s).padStart(2,'0')}
function stopAll(){for(const p of Object.values(players))for(const a of Object.values(p.aud))a.pause();document.querySelectorAll('button.L').forEach(b=>b.classList.remove('on'))}
function play(id,L){const p=players[id];const t=cur&&cur.id===id?cur.a.currentTime:(p.pos||0);stopAll();const a=p.aud[L];a.currentTime=Math.min(t,(a.duration||t+1)-0.05);a.loop=true;a.play();cur={id,L,a};p.btn[L].classList.add('on');setActive(id)}
function setActive(id){active=id;document.querySelectorAll('.card').forEach(c=>c.classList.toggle('active',c.dataset.id===id))}
function mkPlayer(id,items,box){const p={aud:{},btn:{},pos:0};players[id]=p;const pl=document.createElement('div');pl.className='pl';
 for(const [L,src,label] of items){const a=new Audio(src);a.preload='auto';p.aud[L]=a;const b=document.createElement('button');b.className='L';b.textContent=label||L;b.onclick=()=>play(id,L);p.btn[L]=b;pl.appendChild(b);
  a.addEventListener('timeupdate',()=>{if(cur&&cur.a===a){p.pos=a.currentTime;rng.value=a.duration?a.currentTime/a.duration*1000:0;tm.textContent=fmt(a.currentTime)+' / '+fmt(a.duration)}})}
 const stop=document.createElement('button');stop.textContent='■ 停止';stop.onclick=()=>{stopAll();cur=null};const top=document.createElement('button');top.textContent='⏮ 先頭';top.onclick=()=>{p.pos=0;if(cur&&cur.id===id)cur.a.currentTime=0};
 pl.append(stop,top);box.appendChild(pl);const rng=document.createElement('input');rng.type='range';rng.min=0;rng.max=1000;rng.value=0;
 rng.oninput=()=>{const a=(cur&&cur.id===id)?cur.a:Object.values(p.aud)[0];if(a.duration){p.pos=rng.value/1000*a.duration;if(cur&&cur.id===id)a.currentTime=p.pos}};
 const tm=document.createElement('div');tm.className='mut';tm.textContent='0:00';box.append(rng,tm)}
function rankRow(t,kind,label,box){const r=document.createElement('div');r.className='row';r.innerHTML='<div>'+label+'</div>';const s=document.createElement('div');s.className='sel';
 const v=((ans.trials[t.id]||{})[kind])||{};for(const L of t.letters){const lab=document.createElement('label');lab.textContent=L+': ';const sel=document.createElement('select');
  sel.innerHTML='<option value="">–</option>'+t.letters.map((_,i)=>'<option>'+(i+1)+'</option>').join('');sel.value=v[L]||'';
  sel.onchange=()=>{ans.trials[t.id]=ans.trials[t.id]||{};ans.trials[t.id][kind]=ans.trials[t.id][kind]||{};ans.trials[t.id][kind][L]=sel.value?+sel.value:null;check(t);persist()};lab.appendChild(sel);s.appendChild(lab)}
 r.appendChild(s);box.appendChild(r)}
function valid(t,kind){const v=((ans.trials[t.id]||{})[kind])||{};const xs=t.letters.map(L=>v[L]).filter(x=>x);return xs.length===t.letters.length&&new Set(xs).size===xs.length}
function check(t){const w=document.getElementById('w_'+t.id),st=document.getElementById('st_'+t.id);const a=valid(t,'chorus'),b=valid(t,'overall');
 const v=(ans.trials[t.id]||{});const partial=k=>{const o=v[k]||{};const xs=Object.values(o).filter(x=>x);return xs.length>0&&!valid(t,k)};
 w.textContent=(partial('chorus')||partial('overall'))?'順位が重複しているか未記入があります':'';st.textContent=a&&b?'記入済み':(a?'コーラスのみ記入':'未記入');st.classList.toggle('done',a&&b)}
function prog(){const n=TRIALS.filter(t=>valid(t,'chorus')).length;document.getElementById('prog').textContent='コーラス順位 記入済み '+n+' / '+TRIALS.length}
const root=document.getElementById('trials');let grp='';
TRIALS.forEach((t,i)=>{if(t.set!==grp){grp=t.set;const h=document.createElement('h2');h.textContent=t.head;root.appendChild(h)}
 const c=document.createElement('div');c.className='card';c.dataset.id=t.id;c.onclick=()=>setActive(t.id);
 c.innerHTML='<div class="hd"><span class="tid">'+(i+1)+'. '+t.id+'</span><span class="st" id="st_'+t.id+'"></span></div>';
 mkPlayer(t.id,t.letters.map(L=>[L,t.id+'/'+L+'.wav']),c);rankRow(t,'chorus','コーラス感が少ない順位',c);rankRow(t,'overall','総合で好ましい順位',c);
 const w=document.createElement('div');w.className='warn';w.id='w_'+t.id;c.appendChild(w);
 const ta=document.createElement('textarea');ta.placeholder='メモ(任意)';ta.value=((ans.trials[t.id]||{}).note)||'';ta.oninput=()=>{ans.trials[t.id]=ans.trials[t.id]||{};ans.trials[t.id].note=ta.value;persist()};c.appendChild(ta);
 root.appendChild(c);check(t)});
const ec=document.getElementById('errcard');if(ec){ec.dataset.id='err';ec.onclick=()=>setActive('err');
mkPlayer('err',ERR.map((e,i)=>['E'+i,'../err_bands/'+e[1],e[0]]),ec);
const eta=document.createElement('textarea');eta.placeholder='聴いた印象(例: 除去版でコーラスが消えた/残った)';eta.value=ans.err_note||'';eta.oninput=()=>{ans.err_note=eta.value;persist()};ec.appendChild(eta);}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='SELECT'||!active)return;const p=players[active];const Ls=Object.keys(p.aud);
 if(e.key===' '){e.preventDefault();stopAll();cur=null}else{const k=+e.key;if(k>=1&&k<=Ls.length){e.preventDefault();play(active,Ls[k-1])}}});
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');const body=JSON.stringify({...ans,client_saved_at:new Date().toISOString()});
 try{const r=await fetch('__SAVE__',{method:'POST',headers:{'Content-Type':'application/json'},body});const j=await r.json();m.textContent=j.ok?'保存しました → '+j.path:'保存失敗'}
 catch(e){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download='answers_ear.json';a.click();m.textContent='サーバに届かないためファイルとしてダウンロードしました'}};
prog();
</script></body></html>
"""

HEADS = {"fullc": "fullc — 最優先", "full": "full", "ovf": "ovf", "fullm": "fullm"}
ERRHTML = """<h2>誤差帯域の聴き比べ(盲検ではない・ラベル付き)</h2>
<div class="mut">s11 の出力から 16–50Hz(フレーム周期)の誤差だけを除くとコーラスが消えるか。GTに同帯域の誤差だけを足すとコーラスが乗るか。</div>
<div class="card" id="errcard"></div>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="d1_ab")
    ap.add_argument("--order", default="fullc,full,ovf,fullm")
    ap.add_argument("--save", default="/save_answers")
    ap.add_argument("--no-err", action="store_true")
    ap.add_argument("--title", default="D1 盲検A/B(2026-09-23)")
    ap.add_argument("--intro", default="各試行に錨(codec往復=コーラスなし)が1本混ざっています。")
    ap.add_argument("--key", default="_key_聴取後に開く.json", help="試行と文字だけを読む鍵ファイル(dir からの相対)")
    a = ap.parse_args()
    ab = EB / a.dir
    key = json.loads((ab / a.key).read_text())
    trials = []
    for s in a.order.split(","):
        for t in sorted(k for k in key if k.split("_", 1)[0] == s):
            trials.append({"id": t, "set": s, "head": HEADS.get(s, s),
                           "letters": sorted(key[t]["map"])})
    html = (PAGE.replace("__TRIALS__", json.dumps(trials, ensure_ascii=False))
            .replace("__ERR__", json.dumps([] if a.no_err else ERR, ensure_ascii=False))
            .replace("__ERRHTML__", "" if a.no_err else ERRHTML)
            .replace("__TITLE__", a.title).replace("__INTRO__", a.intro)
            .replace("__SAVE__", a.save)
            .replace("__LSKEY__", "d1ab_answers_v1" if a.dir == "d1_ab" else f"ab_answers_{a.dir}_v1"))
    (ab / "listen.html").write_text(html)
    print(f"{len(trials)} trials -> {ab / 'listen.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
