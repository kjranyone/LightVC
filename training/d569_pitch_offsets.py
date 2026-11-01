"""D569(オーナー依頼 10-24・推論だけ): 男声 → 女声の P57S0 で、出力の高さを目標の声域へ移さず「入力の高さ + 0 / +5 / +7 / +12 半音」にした 4 版を作り、耳のページにする。
10-25 オーナー: 目標の声域への自動のずらしは実運用に無い → 自動の版は載せない(初版 mf_pitch_ear は +0 / +5 / +12 + 自動)。
組 = mf_v3_ear の 12 試行と同じ(関門の組 mfv3)。Rust の書き出しは gate_mf_run.py と同じ引数 + --no-f0-shift(条件の 25 列 = ln(入力の f0 / 200))。
25 列以外は関門の書き出しと一致することを確かめ、有声のフレームの 25 列に k·ln2/12 を足す。描画は関門と同じ(水準ずらし・操作者の遅い揺らぎ 0・C2)。
音量 = 試行の中の全部を −23 dBFS(有音の部分)・ピーク 0.95 を超えるなら試行に共通の減衰。
出力 = results/earbattery/mf_pitch_ear2/{tNN_in, tNN_tgt, tNN_p00, tNN_p05, tNN_p07, tNN_p12}.wav・listen.html。
    uv run python d569_pitch_offsets.py
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d566_mf_v3_ear import active_rms, best8, load
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
S = P.H / "mfv3"
ED0 = P.H / "e2e" / "eval_mfv3_pre"
ED = P.H / "e2e" / "eval_mfv3_pre_nof0"
OUT = ROOT / "results/earbattery/mf_pitch_ear2"
CKPT = ROOT / "results/diag_ppgvoc57/product_002500_m0.pt"
PRE = 12.0
KS = (0, 5, 7, 12)


def main() -> int:
    key0 = json.loads((ROOT / "results/earbattery/mf_v3_ear/_key_聴取後に開く.json").read_text())
    pairs = {f"p{r['pi']:04d}_t{r['ti']:03d}": r for r in json.loads((S / "devset.json").read_text())["pairs"]}
    ED.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t, v in key0.items():
        r = pairs[v["pair"]]
        w = S / "d56/_work" / f"pre_{v['pair']}.wav"
        jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["--no-f0-shift", "-i", w], ED / f"{v['pair']}.cond", ED / f"{v['pair']}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(job, jobs))
    dev = "cuda"
    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 0.0
    P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    import train_c1 as TC
    n0 = int(PRE * N.SR)
    key, info = {}, {}
    for t, v in key0.items():
        k, r = v["pair"], pairs[v["pair"]]
        c, lg = P.load_dump(ED, k)
        c1, lg1 = P.load_dump(ED0, k)
        n = min(len(c), len(c1))
        oth = [j for j in range(31) if j != 25]
        d_oth, d_lg = float(np.abs(c[:n, oth] - c1[:n, oth]).max()), float(np.abs(lg[:n] - lg1[:n]).max())
        assert d_oth < 1e-4 and d_lg < 1e-4, (k, d_oth, d_lg)
        pre = c[:2400, 0]
        mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
        c = c.copy()
        c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
        vv = c[:, 26] > 0.5
        cvf = P.cvin_frames(torch.from_numpy(TC.prime(load48(S / "d56/_work" / f"pre_{k}.wav").astype(np.float32)))[None].to(dev), len(lg))
        en = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        xs = {"in": load(S / "d56" / f"{k}_SRC.wav")[: 8 * N.SR], "tgt": best8(load(S / "d37" / f"cen_{r['ti']:03d}.wav"))}
        f0in = float(200 * np.exp(np.median(c[n0 // N.HOP:][vv[n0 // N.HOP:], 25])))
        f0auto = float(200 * np.exp(np.median(c1[n0 // N.HOP:n][c1[n0 // N.HOP:n, 26] > 0.5, 25])))
        for kk in KS:
            ck = c.copy()
            ck[:, 25] = np.where(vv, c[:, 25] + kk * np.log(2) / 12, c[:, 25])
            y = P.render(g, ck, lg, en, dev, T_op=P.op_tab(S, k), cvf=cvf)[n0:n0 + 8 * N.SR]
            xs[f"p{kk:02d}"] = y.astype(np.float32)
        xs = {m: x * (10 ** (-23 / 20) / active_rms(x)) for m, x in xs.items()}
        pk = max(float(np.abs(x).max()) for x in xs.values())
        gn = 0.95 / pk if pk > 0.95 else 1.0
        for m, x in xs.items():
            sf.write(OUT / f"{t}_{m}.wav", (x * gn).astype(np.float32), N.SR, subtype="PCM_16")
        key[t] = {**v, "f0_in_median_hz": round(f0in, 1), "f0_auto_median_hz": round(f0auto, 1), "auto_shift_semitones": round(12 * np.log2(f0auto / f0in), 1),
                  "common_gain_db": round(20 * np.log10(gn), 2), "check_max_abs_diff_other_cols": d_oth}
        print(t, key[t], flush=True)
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
    (OUT / "listen.html").write_text(PAGE.replace("__TRIALS__", json.dumps(list(key))).replace("__INFO__", json.dumps({t: v["f0_in_median_hz"] for t, v in key.items()})), encoding="utf-8")
    print("done", len(key), "->", OUT, flush=True)
    return 0


PAGE = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>高さの比較</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117}}
:root[data-theme="dark"]{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 120px}h1{font-size:20px;margin:0 0 6px}
.mut{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.pl{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 6px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 12px;cursor:pointer}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
textarea{width:100%;min-height:44px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;justify-content:center;flex-wrap:wrap}
</style></head><body><main>
<h1>男声 → 女声: 出力の高さの比較</h1>
<div class="mut">同じ変換(P57S0)で、出力の高さだけを替えた版です。<b>+0 / +5 / +7 / +12</b> = 入力の男声の高さから何半音上げたか(抑揚の形はそのまま)。<b>入力</b>・<b>目標</b>(変換先の人の別の録音)は比較用。ボタンで同じ位置から切り替わります。音量は試行の中でそろえてあります。各版の「聞かせられる / 聞かせられない」は任意です。途中でやめても構いません。</div>
<div id="trials"></div>
<div class="card"><b>所見</b><textarea id="ocm" placeholder="全体の所見(任意)"></textarea></div>
</main>
<footer><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const TRIALS=__TRIALS__;const INFO=__INFO__;const KEY='mf_pitch_ear2_v1';
let ans={notes:{},ok:{},comment:''};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}ans.ok=ans.ok||{};
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
let cur=null;const P={};
function stopAll(){for(const p of Object.values(P))for(const a of Object.values(p))a.pause();document.querySelectorAll('.pl button.L').forEach(b=>b.classList.remove('on'))}
function play(id,n,btn){const t=cur&&cur.id===id?cur.a.currentTime:0;stopAll();const a=P[id][n];a.currentTime=Math.min(t,(a.duration||8)-0.05);a.loop=true;a.play();cur={id,a};btn.classList.add('on')}
const root=document.getElementById('trials');
TRIALS.forEach((id,i)=>{const c=document.createElement('div');c.className='card';const f=INFO[id];
 c.innerHTML='<b>試行 '+(i+1)+' / '+TRIALS.length+'</b> <span class="mut">入力の高さ 約 '+Math.round(f)+' Hz(+12 で約 '+Math.round(f*2)+' Hz)</span>';
 const pl=document.createElement('div');pl.className='pl';P[id]={};
 for(const [n,lab] of [['in','入力'],['p00','+0'],['p05','+5'],['p07','+7'],['p12','+12'],['tgt','目標']]){const a=new Audio(id+'_'+n+'.wav');a.preload='auto';P[id][n]=a;const b=document.createElement('button');b.className='L';b.textContent=lab;b.onclick=()=>play(id,n,b);pl.appendChild(b)}
 const s=document.createElement('button');s.textContent='■ 停止';s.onclick=()=>{stopAll();cur=null};pl.appendChild(s);c.appendChild(pl);
 const g=document.createElement('div');g.className='mut';g.style.display='flex';g.style.flexWrap='wrap';g.style.gap='4px 14px';
 for(const v of ['p00','p05','p07','p12']){const sp=document.createElement('span');sp.textContent=v.replace('p0','+').replace('p12','+12')+': ';
  for(const o of ['聞かせられる','聞かせられない']){const l=document.createElement('label');const r=document.createElement('input');r.type='radio';r.name=id+'_'+v;r.checked=((ans.ok[id]||{})[v]===o);r.onchange=()=>{ans.ok[id]=ans.ok[id]||{};ans.ok[id][v]=o;persist()};l.append(r,o+' ');sp.appendChild(l)}g.appendChild(sp)}
 c.appendChild(g);
 const t=document.createElement('textarea');t.placeholder='所見(任意)';t.value=ans.notes[id]||'';t.oninput=()=>{ans.notes[id]=t.value;persist()};c.appendChild(t);root.appendChild(c)});
const oc=document.getElementById('ocm');oc.value=ans.comment||'';oc.oninput=()=>{ans.comment=oc.value;persist()};
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');try{const r=await fetch('/save_mfpitch2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(ans)});m.textContent=r.ok?'保存しました '+new Date().toLocaleTimeString():'保存に失敗しました('+r.status+')'}catch(e){m.textContent='保存に失敗しました'}};
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
