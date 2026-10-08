"""ZS-VC の耳の盲検(ゼロショット男→女)。results/earbattery/zsvc_ab/listen.html(保存先 /save_zsvc)。

各試行 = 元の男声・目標の参照(3s)・変換。問い = 目標らしさ(似ている/やや/似ていない)と品質(合格/微妙/不合格)+メモ。
うち 2 試行は、変換に使った目標と参照を入れ替えた対照(目標らしさの判定が当てずっぽうでないかを見る)。
描画は PyTorch(音響モデル + nvoc)。Rust の音響モデルは未移植。

    CUDA_VISIBLE_DEVICES=0 uv run python make_zsvc_ear.py --ckpt ../results/zsvc3/last.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import eval_zsvc as EZ
import nvoc as N
import train_zsvc as TZ
import zsvc as Z
import artic_dsp as AD

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/earbattery/zsvc_ab"
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"

PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ゼロショット変換 盲検</title><style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 90px}h1{font-size:20px;margin:0 0 6px}.mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:6px 12px;cursor:pointer;margin:0 6px 6px 0}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}label{margin-right:14px;white-space:nowrap}.q{margin-top:4px}
textarea{width:100%;min-height:44px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
footer{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;justify-content:center;align-items:center;flex-wrap:wrap}
</style></head><body><main><h1>ゼロショット変換(男声 → 目標の女声)</h1>
<div class="mut">各試行に 3 本: <b>元の声</b>(入力の男声)・<b>参照</b>(目標の本人の声 3 秒。変換はこれだけを手がかりにしています)・<b>変換</b>。
<br>「目標らしさ」= 変換が参照の人の声に聞こえるか。「品質」= 声として聴けるか(ざらつき・こもり・コーラス・機械っぽさ)。気づきはメモへ。<br>
試行の中に、わざと別の人の参照と組ませた対照があります。</div>
<div id="items"></div><div class="card"><textarea id="note" placeholder="全体のメモ(任意)"></textarea></div></main>
<footer><span id="prog" class="mut"></span><button id="save" class="on">保存</button><span id="msg" class="mut"></span></footer>
<script>
const IT=__ITEMS__;const KEY='zsvc_ab_v2';let ans={sim:{},qual:{},memo:{},note:''};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}document.getElementById('prog').textContent='記入 '+Object.keys(ans.qual).length+' / '+IT.length}
let cur=null;function play(src,b){if(cur){cur.a.pause();cur.b.classList.remove('on')}const a=new Audio(src);a.play();b.classList.add('on');cur={a,b};a.onended=()=>b.classList.remove('on')}
function radios(id,group,opts){const r=document.createElement('div');r.className='q';r.append(group==='sim'?'目標らしさ: ':'品質: ');
 for(const v of opts){const l=document.createElement('label');const i=document.createElement('input');i.type='radio';i.name=group+id;i.checked=ans[group][id]===v;i.onchange=()=>{ans[group][id]=v;persist()};l.append(i,' '+v);r.appendChild(l)}return r}
const box=document.getElementById('items');for(const it of IT){const d=document.createElement('div');d.className='card';d.innerHTML='<b>'+it.id+'</b><br>';
 for(const [lab,src] of [['元の声',it.src],['参照',it.ref],['変換',it.conv]]){const b=document.createElement('button');b.textContent='▶ '+lab;b.onclick=()=>play(src,b);d.appendChild(b)}
 d.appendChild(radios(it.id,'sim',['似ている','やや','似ていない']));d.appendChild(radios(it.id,'qual',['合格','微妙','不合格']));
 const t=document.createElement('textarea');t.placeholder='メモ';t.value=ans.memo[it.id]||'';t.oninput=()=>{ans.memo[it.id]=t.value;persist()};d.appendChild(t);box.appendChild(d)}
const nt=document.getElementById('note');nt.value=ans.note||'';nt.oninput=()=>{ans.note=nt.value;persist()};
document.getElementById('save').onclick=async()=>{const m=document.getElementById('msg');const body=JSON.stringify({...ans,client_saved_at:new Date().toISOString()});
 try{const r=await fetch('/save_zsvc2',{method:'POST',headers:{'Content-Type':'application/json'},body});const j=await r.json();m.textContent=j.ok?'保存しました':'保存失敗'}
 catch(e){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([body],{type:'application/json'}));a.download='zsvc_answers.json';a.click();m.textContent='サーバに届かないためダウンロードしました'}};
persist();
</script></body></html>
"""


def level(x: np.ndarray, rms: float = 0.05) -> np.ndarray:
    y = x * (rms / max(float(np.sqrt((x ** 2).mean())), 1e-6))
    pk = float(np.abs(y).max())
    return (y * min(1.0, 0.95 / pk)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="snap(ema・voc_ema・rip)か last.pt")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    dev = "cuda"
    from train_ddsp_vc import index, load48
    spk, tr, ev = index()
    _, mh = TZ.male_index()
    es = EZ.build_evalset(spk, ev, mh)
    tg = []
    for k in [k for k in ev if k.startswith("real_female") and len(spk[k]) >= 2]:
        lm, sd = EZ._logf0(spk[k][0][0])
        if 250 <= np.exp(lm) <= 350:
            tg.append({"name": k.split("/")[-1][:8], "ref": load48(spk[k][0][1])[:3 * N.SR], "stat": (lm, sd)})
        if len(tg) == 3:
            break
    srcs = [{"name": f"ja{i}", **s} for i, s in enumerate(es["src"])]
    info = (VC.parent / "speaker-info.txt").read_text().splitlines()[1:]
    males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    for s in males[3::9]:
        p = VC / s / f"{s}_040.wav"
        if not p.is_file():
            continue
        x = load48(p)[:6 * N.SR]
        x = x[:len(x) // N.HOP * N.HOP]
        import f0_fix as FX
        f0 = FX.fix_f0(AD.causal_yin(x.astype(np.float64), voi_max=0.45)[0].astype(np.float32))[0]
        enr = np.concatenate([load48(VC / s / f"{s}_{u:03d}.wav") for u in (41, 42, 43) if (VC / s / f"{s}_{u:03d}.wav").is_file()])
        fe = FX.fix_f0(AD.causal_yin(enr.astype(np.float64), voi_max=0.45)[0].astype(np.float32))[0]
        srcs.append({"name": s, "x": x, "f0": f0, "stat": FX.logf0_stats(fe)})
        if len(srcs) == 6:
            break
    srcs = [srcs[0], srcs[1], srcs[4], srcs[5]] if len(srcs) >= 6 else srcs
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    m = Z.ZSVC(cv=bool(ck.get("cv", False)), env_keep=int(ck.get("env_keep", 0) or 0), rip=bool(ck.get("rip", False))).to(dev)
    m.load_state_dict(ck["ema"])
    m.eval()
    voc = N.NVoc().to(dev)
    voc.load_state_dict(ck["voc_ema"] if ck.get("voc_ema") is not None else torch.load(ROOT / "results/nvoc5r2/last.pt", map_location=dev, weights_only=False)["ema"])
    voc.eval()
    rng = random.Random("zsvc_ab_20260930")
    trials = [("match", s, tg[i % 3], tg[i % 3]) for i, s in enumerate(srcs)]
    trials += [("control", srcs[0], tg[1], tg[0]), ("control", srcs[-1], tg[2], tg[1])]
    rng.shuffle(trials)
    out.mkdir(parents=True, exist_ok=True)
    key, page = {"ckpt": a.ckpt, "trials": {}}, []
    for i, (kind, s, t_conv, t_ref) in enumerate(trials):
        lf = np.log(np.maximum(s["f0"], 1.0))
        f0m = np.where(s["f0"] > 0, np.exp(t_conv["stat"][0] + (lf - s["stat"][0])), 0.0).astype(np.float32)
        w, _ = EZ.render(m, voc, s["x"], f0m, t_conv["ref"], dev)
        tid = f"T{i + 1}"
        soundfile.write(out / f"{tid}_src.wav", level(s["x"]), N.SR)
        soundfile.write(out / f"{tid}_ref.wav", level(t_ref["ref"]), N.SR)
        soundfile.write(out / f"{tid}_conv.wav", level(w), N.SR)
        key["trials"][tid] = {"kind": kind, "src": s["name"], "conv_target": t_conv["name"], "ref_shown": t_ref["name"]}
        page.append({"id": tid, "src": f"{tid}_src.wav", "ref": f"{tid}_ref.wav", "conv": f"{tid}_conv.wav"})
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (out / "listen.html").write_text(PAGE.replace("__ITEMS__", json.dumps(page)))
    print("ear page ->", out / "listen.html", len(page), "trials", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
