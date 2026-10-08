"""話者の経路の耳(学習なし・オーナー承認 2026-10-01「両方行う」): 男声 → 女声で、目標の包絡の時間軌道を差し替えた音が目標に似るか・質は許せるか。

素材 = a2vc 設計レビュー 4 巡目の差し替えラダー(VCTK 同文 3〜6 を DTW 整列・STFT 上で log-mel DCT 低次 k 次の包絡を差し替え・励起は入力の実励起を RRPS で音域移動)。
各試行: 参照(目標話者の別の発話 ≥ 10s・伏せない)→ 伏せた候補 7 本に「目標に似ているか」「質(合格/不合格)」。
候補: H1_24(軌道 DCT24)・H1_12(DCT12)・H1Q10_24(10s 参照のフレームで量子化 = 現実的なゼロショットの上限)・H1LP4_24(変化を 4Hz 未満に限定)・
R0(音域移動だけ)・T0(目標本人の同じ文 = 陽性の対照)・NEG(別の女声の同じ文 = 陰性の対照)。音量は試行内で RMS をそろえ共通の減衰。長さは試行内で同じ(最短か 10s・長さで条件が分からないように)。

    uv run python make_spk_ear.py --src <scratchpad>/r4_spk --n 4
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
CANDS = ("H1_24", "H1_12", "H1Q10_24", "H1LP4_24", "R0", "T0", "NEG")


def load48(p: Path) -> np.ndarray:
    import librosa
    x, _ = librosa.load(str(p), sr=48000, mono=True)
    return x.astype(np.float64)


def reference(spk: str, min_sec: float = 10.0) -> np.ndarray:
    import librosa
    out, tot = [], 0
    for u in range(41, 120):
        p = VC / spk / f"{spk}_{u:03d}.wav"
        if not p.exists():
            continue
        y, _ = librosa.effects.trim(load48(p), top_db=35)
        out += [y, np.zeros(9600)]
        tot += len(y)
        if tot >= min_sec * 48000:
            break
    return np.concatenate(out)


def norm_trial(clips: dict) -> dict:
    ys = {k: v / (np.sqrt((v ** 2).mean()) + 1e-9) * 0.05 for k, v in clips.items()}
    pk = max(float(np.abs(v).max()) for v in ys.values())
    a = 0.95 / pk if pk > 0.95 else 1.0
    return {k: (v * a).astype(np.float32) for k, v in ys.items()}


PAGE = r"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>話者の経路の耳</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f6fdb;--accfg:#fff;--ok:#1e7a3c}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--fg:#ecebe6;--mut:#a3a29b;--line:#34332f;--acc:#6ea0ff;--accfg:#0d1117;--ok:#7ddc9a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Hiragino Sans","Noto Sans JP",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 120px}h1{font-size:20px;margin:0 0 6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:14px 0}
.ref{border-left:4px solid var(--acc);padding-left:10px;margin-bottom:10px}
.row{display:grid;grid-template-columns:36px 1fr;gap:8px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}
.lab{font-weight:600}audio{width:100%;max-width:100%}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:4px 12px;cursor:pointer}
button.on{background:var(--acc);color:var(--accfg);border-color:var(--acc)}
.q{color:var(--mut);font-size:13px;margin-right:4px}
textarea{width:100%;min-height:52px;font:inherit;border:1px solid var(--line);border-radius:8px;padding:6px;background:var(--card);color:var(--fg)}
#save{position:fixed;bottom:16px;right:16px;padding:10px 18px;background:var(--acc);color:var(--accfg);border:none;border-radius:10px}
#msg{position:fixed;bottom:22px;right:150px;color:var(--mut);font-size:13px}
</style></head><body><main>
<h1>話者の経路の耳(学習なし)</h1>
<p>各試行の最初に<b>目標の声(参照)</b>があります。続く 7 本(伏せてあります)について、<b>参照の人に似ているか</b>と<b>質(合格/不合格)</b>を付けてください。気づき(コーラス・ざらつき・機械的など)はメモへ。7 本の中には目標本人の録音と別人の録音も混ざっています。音量はそろえてあります。素材は英語(VCTK)です。</p>
<div id="trials"></div>
</main><button id="save">保存</button><span id="msg"></span>
<script>
const T=__TRIALS__;const KEY='spk_ear_v1';
let ans={};try{const s=localStorage.getItem(KEY);if(s)ans=JSON.parse(s)}catch(e){}
function persist(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function btn(label,on,f){const b=document.createElement('button');b.textContent=label;if(on)b.classList.add('on');b.onclick=f;return b}
function render(){const root=document.getElementById('trials');root.innerHTML='';
T.forEach((t,i)=>{const c=document.createElement('div');c.className='card';
 c.innerHTML=`<div class="ref"><div class="lab">試行 ${i+1}:目標の声(参照)</div><audio controls preload="none" src="${t.id}/ref.wav"></audio></div>`;
 ans[t.id]=ans[t.id]||{c:{},memo:''};
 t.letters.forEach(L=>{const r=document.createElement('div');r.className='row';
  const a=ans[t.id].c[L]=ans[t.id].c[L]||{};
  const d=document.createElement('div');d.innerHTML=`<audio controls preload="none" src="${t.id}/${L}.wav"></audio>`;
  const b1=document.createElement('div');b1.className='btns';b1.innerHTML='<span class="q">似ているか</span>';
  ['似ている','少し似ている','似ていない'].forEach(v=>b1.appendChild(btn(v,a.sim===v,()=>{a.sim=v;persist();render()})));
  const b2=document.createElement('div');b2.className='btns';b2.innerHTML='<span class="q">質</span>';
  ['合格','不合格'].forEach(v=>b2.appendChild(btn(v,a.q===v,()=>{a.q=v;persist();render()})));
  d.appendChild(b1);d.appendChild(b2);
  r.innerHTML=`<div class="lab">${L}</div>`;r.appendChild(d);c.appendChild(r)});
 const m=document.createElement('textarea');m.placeholder='メモ(任意)';m.value=ans[t.id].memo||'';m.oninput=()=>{ans[t.id].memo=m.value;persist()};c.appendChild(m);
 root.appendChild(c)})}
render();
document.getElementById('save').onclick=async()=>{const msg=document.getElementById('msg');msg.textContent='保存中…';
 try{const r=await fetch('/save_spk_ear',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({answers:ans,saved_at:new Date().toISOString()})});const j=await r.json();msg.textContent=j.ok?'保存しました':'保存失敗'}catch(e){msg.textContent='保存失敗: '+e}};
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="差し替えラダーの置き場(r4_spk)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--dir", default="spk_ear")
    a = ap.parse_args()
    src = Path(a.src)
    jobs = json.loads((src / "jobs.json").read_text())["jobs"]
    mf = [j for j in jobs if j[0] == "MF"]
    rng = random.Random(20261001)
    pick = rng.sample(mf, a.n)
    fem_t0 = sorted({j[2] for j in mf})
    out = EB / a.dir
    out.mkdir(parents=True, exist_ok=True)
    key: dict = {"source": str(src), "conditions": {
        "H1_24": "男声(音域移動)に目標の DCT24 包絡の時間軌道", "H1_12": "同 DCT12", "H1Q10_24": "同 DCT24 を 10s 参照のフレームで量子化",
        "H1LP4_24": "同 DCT24 の変化を 4Hz 未満に限定", "R0": "音域移動だけ", "T0": "目標本人の同じ文(陽性の対照)", "NEG": "別の女声の同じ文(陰性の対照)"},
        "trials": {}}
    trials = []
    for i, (_, m, f, ratio) in enumerate(pick):
        neg_f = rng.choice([x for x in fem_t0 if x != f])
        neg_job = next(j for j in mf if j[2] == neg_f)
        files = {c: src / ("wav" if c in ("H1_24", "H1_12", "R0", "T0") else "wav3" if c.startswith("H1Q") else "wav2") / f"MF__{m}__{f}__{c}.wav"
                 for c in CANDS if c != "NEG"}
        files["NEG"] = src / "wav" / f"MF__{neg_job[1]}__{neg_f}__T0.wav"
        clips = {c: soundfile.read(p)[0].astype(np.float64) for c, p in files.items()}
        n = min(min(len(v) for v in clips.values()), 10 * 48000)
        clips = {c: v[:n] for c, v in clips.items()}
        clips["ref"] = reference(f)
        normed = norm_trial(clips)
        letters = list("ABCDEFG")
        order = list(CANDS)
        random.Random(f"spk{i}").shuffle(order)
        tid = f"t{i + 1}"
        td = out / tid
        td.mkdir(exist_ok=True)
        soundfile.write(td / "ref.wav", normed["ref"], 48000)
        key["trials"][tid] = {"src": m, "tgt": f, "neg": neg_f, "f0_ratio": round(float(ratio), 3), "map": {}}
        for L, c in zip(letters, order):
            soundfile.write(td / f"{L}.wav", normed[c], 48000)
            key["trials"][tid]["map"][L] = c
        trials.append({"id": tid, "letters": letters})
        print(tid, m, "→", f, "neg", neg_f, flush=True)
    (out / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (out / "listen.html").write_text(PAGE.replace("__TRIALS__", json.dumps(trials)))
    print("ear page ->", out / "listen.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
