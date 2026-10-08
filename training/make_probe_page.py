"""コーラス有無の盲検判定ページ results/earbattery/<dir>/listen.html を生成(鍵は埋め込まない)。

    uv run python make_probe_page.py                                              # chorus_probe
    uv run python make_probe_page.py --dir artic_s04 --save /save_artic_s04 --store artic_s04_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"

PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>コーラス有無 盲検</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff;--ok:#1e7a3c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117;--ok:#7ddc9a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 110px}h1{font-size:20px;margin:0 0 6px}
.mut{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.pl{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 12px;cursor:pointer}
button.L{min-width:52px;font-weight:700}button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
input[type=range]{width:100%}
table{width:100%;border-collapse:collapse}td{padding:6px 4px;border-top:1px solid var(--line)}td:first-child{font-weight:700;width:52px}
label{margin-right:14px;white-space:nowrap}
textarea{width:100%;min-height:44px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
</style></head><body><main>
<h1>コーラス有無の判定(盲検・8本)</h1>
<div class="mut">__INTRO__</div>
<div class="card" id="pc"></div>
<div class="card"><table id="tb"></table><textarea id="note" placeholder="メモ(任意・例: X3はコーラスではなくザラつき)"></textarea></div>
</main>
<footer><span id="prog" class="mut"></span><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const XS=__XS__;const KEY=__STORE__;let ans={ratings:{},note:''};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}document.getElementById('prog').textContent='記入 '+Object.keys(ans.ratings).length+' / '+XS.length}
const aud={},btn={};let cur=null,pos=0;const pc=document.getElementById('pc');const pl=document.createElement('div');pl.className='pl';
function fmt(t){if(!isFinite(t))return'0:00';return Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0')}
function stopAll(){for(const a of Object.values(aud))a.pause();for(const b of Object.values(btn))b.classList.remove('on')}
function play(x){const t=cur?cur.currentTime:pos;stopAll();const a=aud[x];a.currentTime=Math.min(t,(a.duration||t+1)-0.05);a.loop=true;a.play();cur=a;btn[x].classList.add('on')}
const rng=document.createElement('input');rng.type='range';rng.min=0;rng.max=1000;rng.value=0;const tm=document.createElement('div');tm.className='mut';
for(const x of XS){const a=new Audio(x+'.wav');a.preload='auto';aud[x]=a;a.addEventListener('timeupdate',()=>{if(cur===a){pos=a.currentTime;rng.value=a.duration?a.currentTime/a.duration*1000:0;tm.textContent=fmt(a.currentTime)+' / '+fmt(a.duration)}});
 const b=document.createElement('button');b.className='L';b.textContent=x;b.onclick=()=>play(x);btn[x]=b;pl.appendChild(b)}
const st=document.createElement('button');st.textContent='■ 停止';st.onclick=()=>{stopAll();cur=null};const tp=document.createElement('button');tp.textContent='⏮ 先頭';tp.onclick=()=>{pos=0;if(cur)cur.currentTime=0};
pl.append(st,tp);pc.append(pl,rng,tm);rng.oninput=()=>{const a=cur||aud[XS[0]];if(a.duration){pos=rng.value/1000*a.duration;if(cur)a.currentTime=pos}};
const tb=document.getElementById('tb');for(const x of XS){const tr=document.createElement('tr');tr.innerHTML='<td>'+x+'</td>';const td=document.createElement('td');
 for(const v of __CHOICES__){const l=document.createElement('label');const r=document.createElement('input');r.type='radio';r.name=x;r.value=v;r.checked=ans.ratings[x]===v;r.onchange=()=>{ans.ratings[x]=v;persist()};l.append(r,' '+v);td.appendChild(l)}
 tr.appendChild(td);tb.appendChild(tr)}
const nt=document.getElementById('note');nt.value=ans.note||'';nt.oninput=()=>{ans.note=nt.value;persist()};
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key===' '){e.preventDefault();stopAll();cur=null;return}const k=+e.key;if(k>=1&&k<=XS.length){e.preventDefault();play(XS[k-1])}});
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');const body=JSON.stringify({...ans,client_saved_at:new Date().toISOString()});
 try{const r=await fetch(__SAVE__,{method:'POST',headers:{'Content-Type':'application/json'},body});const j=await r.json();m.textContent=j.ok?'保存しました → '+j.path:'保存失敗'}
 catch(e){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download=__DL__;a.click();m.textContent='サーバに届かないためダウンロードしました'}};
persist();
</script></body></html>
"""


DEFAULT_INTRO = '同じ発話を8通りに加工したものです。<b>各クリップにコーラス/フェーザー感があるか</b>を「有・微・無」で付けてください。コーラスのない錨が1本、確実にコーラスのある対照が1本混ざっています。音量は揃えてあります。<br>\nボタン(またはキー1〜8)で同じ再生位置のまま切替・ループ再生、スペースで停止。'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="chorus_probe")
    ap.add_argument("--save", default="/save_probe")
    ap.add_argument("--store", default="chorus_probe_v1")
    ap.add_argument("--intro", default=DEFAULT_INTRO)
    ap.add_argument("--choices", default="有,微,無")
    a = ap.parse_args()
    cp = EB / a.dir
    key = json.loads((cp / "_key_聴取後に開く.json").read_text())
    xs = sorted(key["map"], key=lambda s: int(s[1:]))
    html = (PAGE.replace("__XS__", json.dumps(xs)).replace("__STORE__", json.dumps(a.store))
            .replace("__SAVE__", json.dumps(a.save)).replace("__DL__", json.dumps(f"{a.dir}_answers.json"))
            .replace("__INTRO__", a.intro).replace("__CHOICES__", json.dumps(a.choices.split(","), ensure_ascii=False)))
    (cp / "listen.html").write_text(html)
    print(len(xs), "clips ->", cp / "listen.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
