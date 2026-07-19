from __future__ import annotations

import sys
import json
import html
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SESSION = None
PAIRS = {}
OUT_JSONL = None
DONE = set()


def slot_a_is_cand_a(evaluator: str, pair_id: str) -> bool:
    h = hashlib.sha1(f"{evaluator}|{pair_id}".encode()).hexdigest()
    return (int(h[:8], 16) % 2) == 0


def resolve_slot(evaluator: str, pair_id: str, slot: str):
    pair = PAIRS[pair_id]
    if slot == "ref":
        return pair.get("reference")
    a_is_a = slot_a_is_cand_a(evaluator, pair_id)
    if slot == "a":
        return (pair["cand_a"] if a_is_a else pair["cand_b"])["path"]
    if slot == "b":
        return (pair["cand_b"] if a_is_a else pair["cand_a"])["path"]
    return None


def load_done() -> None:
    if OUT_JSONL.exists():
        for line in OUT_JSONL.read_text().splitlines():
            try:
                r = json.loads(line)
                DONE.add((r.get("evaluator_id"), r.get("pair_id")))
            except Exception:
                continue


def next_pair(evaluator: str) -> tuple:
    ids = list(PAIRS.keys())
    for i, pid in enumerate(ids):
        if (evaluator, pid) not in DONE:
            return pid, i, len(ids)
    return None, len(ids), len(ids)


def record(evaluator: str, body: dict) -> dict:
    pid = body["pair_id"]
    pair = PAIRS[pid]
    a_is_a = slot_a_is_cand_a(evaluator, pid)
    shown_order = "AB" if a_is_a else "BA"

    def true_choice(c: str) -> str:
        if c in ("tie", "reject_both"):
            return c
        if c == "A":
            return "cand_a" if a_is_a else "cand_b"
        if c == "B":
            return "cand_b" if a_is_a else "cand_a"
        return c

    tags_shown_a = body.get("bad_tags_a", [])
    tags_shown_b = body.get("bad_tags_b", [])
    tags_true_a = tags_shown_a if a_is_a else tags_shown_b
    tags_true_b = tags_shown_b if a_is_a else tags_shown_a

    now = datetime.now().isoformat(timespec="seconds")
    row = {
        "eval_id": f"{SESSION['session']}::{pid}::{evaluator}::{now}",
        "timestamp": now,
        "evaluator_id": evaluator,
        "session": SESSION["session"],
        "pair_id": pid,
        "preset": body.get("preset", ""),
        "persona": body.get("persona", ""),
        "scene": body.get("scene", ""),
        "relation": body.get("relation", ""),
        "candidate_a_path": pair["cand_a"]["path"],
        "candidate_b_path": pair["cand_b"]["path"],
        "candidate_a_label": pair["cand_a"].get("label", ""),
        "candidate_b_label": pair["cand_b"].get("label", ""),
        "candidate_a_checkpoint": pair["cand_a"].get("checkpoint", ""),
        "candidate_b_checkpoint": pair["cand_b"].get("checkpoint", ""),
        "candidate_a_controls": pair["cand_a"].get("controls", {}),
        "candidate_b_controls": pair["cand_b"].get("controls", {}),
        "reference_path": pair.get("reference"),
        "shown_order": shown_order,
        "choice_shown": body.get("choice", ""),
        "choice": true_choice(body.get("choice", "")),
        "bad_tags_a": tags_true_a,
        "bad_tags_b": tags_true_b,
        "memo": body.get("memo", ""),
    }
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    DONE.add((evaluator, pid))
    return row


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(obj, ensure_ascii=False).encode())

    def do_GET(self) -> None:
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/api/session":
            self._json({"session": SESSION["session"], "prompt": SESSION["task_prompt"],
                        "bad_tags": SESSION["bad_tags"], "selectors": SESSION["selectors"],
                        "total": len(PAIRS)})
        elif u.path == "/api/next":
            ev = q.get("evaluator", ["anon"])[0]
            pid, idx, total = next_pair(ev)
            if pid is None:
                self._json({"done": True, "index": idx, "total": total})
            else:
                pair = PAIRS[pid]
                self._json({"done": False, "pair_id": pid, "index": idx, "total": total,
                            "has_ref": bool(pair.get("reference"))})
        elif u.path == "/api/progress":
            ev = q.get("evaluator", ["anon"])[0]
            done = sum(1 for (e, _) in DONE if e == ev)
            self._json({"done": done, "total": len(PAIRS)})
        elif u.path == "/audio":
            ev = q.get("evaluator", ["anon"])[0]
            pid = q.get("pair", [""])[0]
            slot = q.get("slot", [""])[0]
            path = resolve_slot(ev, pid, slot) if pid in PAIRS else None
            if not path or not Path(path).exists():
                self._send(404, "text/plain", b"not found")
                return
            data = Path(path).read_bytes()
            self._send(200, "audio/wav", data)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if u.path == "/api/submit":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            ev = body.get("evaluator", "anon")
            if body.get("pair_id") in PAIRS:
                record(ev, body)
                self._json({"ok": True})
            else:
                self._json({"ok": False, "err": "unknown pair"}, 400)
        else:
            self._send(404, "text/plain", b"not found")


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LightVC A/B</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#14151a;color:#e8e8ee;font:15px/1.5 system-ui,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:18px}
h1{font-size:16px;margin:0 0 4px;color:#c9b8ff}
.prompt{color:#9fe0c8;margin-bottom:12px}
.bar{height:6px;background:#26272f;border-radius:3px;overflow:hidden;margin:8px 0 16px}
.bar>i{display:block;height:100%;background:#8a6cff;width:0}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.card{background:#1c1d24;border:1px solid #2c2e38;border-radius:12px;padding:14px}
.card h2{margin:0 0 8px;font-size:15px}
button{font:inherit;color:#e8e8ee;background:#2a2c36;border:1px solid #3a3d4a;border-radius:9px;padding:9px 12px;cursor:pointer}
button:hover{background:#343744}
.play{width:100%;font-size:18px;padding:16px;margin-bottom:8px}
.pA{border-color:#4a6cff}.pB{border-color:#ff8a4a}
.tags label{display:inline-block;margin:2px 4px 2px 0;padding:3px 7px;background:#23242c;border:1px solid #33353f;border-radius:20px;font-size:12px;cursor:pointer}
.tags input{display:none}
.tags input:checked+span{color:#14151a}
.tags label:has(input:checked){background:#ffcf6c;border-color:#ffcf6c;color:#14151a}
.choice{display:flex;gap:8px;margin:14px 0;flex-wrap:wrap}
.choice button{flex:1;padding:14px;font-size:15px}
.choice button.sel{background:#8a6cff;border-color:#8a6cff;color:#12121a;font-weight:600}
.ref{margin-bottom:12px}
.sel-row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
select,input[type=text]{background:#1c1d24;color:#e8e8ee;border:1px solid #33353f;border-radius:8px;padding:8px}
#memo{width:100%;box-sizing:border-box;margin-top:8px}
.submit{width:100%;padding:15px;font-size:16px;background:#2f8f5f;border-color:#2f8f5f;color:#eafff2;font-weight:600;margin-top:12px}
.submit:disabled{opacity:.4;cursor:not-allowed}
.hint{color:#8a8d99;font-size:12px;margin-top:6px}
.done{text-align:center;padding:60px 0;font-size:20px;color:#9fe0c8}
kbd{background:#2a2c36;border:1px solid #3a3d4a;border-radius:5px;padding:1px 6px;font-size:12px}
.topbar{display:flex;gap:10px;align-items:center;margin-bottom:6px}
</style></head><body><div class="wrap">
<div class="topbar"><h1 id="sname">LightVC A/B</h1>
<span class="hint">評価者ID <input type="text" id="ev" style="width:120px" placeholder="your_id"></span>
<span class="hint" id="prog"></span></div>
<div class="prompt" id="prompt"></div>
<div class="bar"><i id="barfill"></i></div>
<div id="app"></div>
</div>
<script>
let S=null, cur=null, choice=null, ev="anon";
const $=s=>document.querySelector(s);
const evInput=$("#ev");
evInput.value=localStorage.getItem("lvc_ev")||"";
evInput.onchange=()=>{ev=evInput.value||"anon";localStorage.setItem("lvc_ev",ev);load();};

async function init(){
  S=await (await fetch("/api/session")).json();
  $("#sname").textContent="A/B — "+S.session;
  $("#prompt").textContent=S.prompt;
  ev=evInput.value||"anon";
  load();
}
function tagbox(side){
  return '<div class="tags">'+S.bad_tags.map(t=>
    `<label><input type="checkbox" data-side="${side}" value="${t}"><span>${t}</span></label>`).join("")+'</div>';
}
function selectors(){
  return '<div class="sel-row">'+Object.entries(S.selectors).map(([k,opts])=>
    `<select id="sel_${k}">`+opts.map(o=>`<option value="${o}">${o||k}</option>`).join("")+`</select>`).join("")+'</div>';
}
async function load(){
  choice=null;
  const n=await (await fetch("/api/next?evaluator="+encodeURIComponent(ev))).json();
  const p=await (await fetch("/api/progress?evaluator="+encodeURIComponent(ev))).json();
  $("#prog").textContent=p.done+" / "+p.total;
  $("#barfill").style.width=(100*p.done/Math.max(p.total,1))+"%";
  if(n.done){$("#app").innerHTML='<div class="done">✔ 全'+n.total+'ペア完了。お疲れさまでした。</div>';return;}
  cur=n;
  const q=`evaluator=${encodeURIComponent(ev)}&pair=${encodeURIComponent(n.pair_id)}`;
  const ref=n.has_ref?`<div class="ref"><button onclick="play('ref')">▶ 原音 Reference <kbd>0</kbd></button></div>`:"";
  $("#app").innerHTML=ref+`
   <div class="cards">
    <div class="card pA"><h2>A</h2><button class="play pA" onclick="play('a')">▶ A を再生 <kbd>1</kbd></button>${tagbox('a')}</div>
    <div class="card pB"><h2>B</h2><button class="play pB" onclick="play('b')">▶ B を再生 <kbd>2</kbd></button>${tagbox('b')}</div>
   </div>
   <div class="choice">
     <button id="cA" onclick="pick('A')">A が良い <kbd>a</kbd></button>
     <button id="cB" onclick="pick('B')">B が良い <kbd>b</kbd></button>
     <button id="cT" onclick="pick('tie')">tie <kbd>t</kbd></button>
     <button id="cR" onclick="pick('reject_both')">両方だめ <kbd>r</kbd></button>
   </div>
   ${selectors()}
   <input type="text" id="memo" placeholder="memo（任意）">
   <button class="submit" id="submit" disabled onclick="submit()">保存して次へ <kbd>Enter</kbd></button>
   <div class="hint">A/B はブラインド（毎回どちらが何かは隠されています）。bad tag は両方に付けられます。</div>`;
  window._q=q;
  ["a","b","ref"].forEach(s=>{const el=document.createElement("audio");el.id="au_"+s;el.src="/audio?"+q+"&slot="+s;document.body.appendChild(el);});
}
function play(s){document.querySelectorAll("audio").forEach(a=>{a.pause();});const el=$("#au_"+s);if(el){el.currentTime=0;el.play();}}
function pick(c){choice=c;["A","B","tie","reject_both"].forEach(x=>{const b=$("#c"+(x==="tie"?"T":x==="reject_both"?"R":x));if(b)b.classList.toggle("sel",x===c);});$("#submit").disabled=false;}
async function submit(){
  if(!choice)return;
  const tags=s=>[...document.querySelectorAll(`input[data-side="${s}"]:checked`)].map(e=>e.value);
  const g=k=>{const e=$("#sel_"+k);return e?e.value:"";};
  await fetch("/api/submit",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({evaluator:ev,pair_id:cur.pair_id,choice,
      bad_tags_a:tags('a'),bad_tags_b:tags('b'),memo:$("#memo").value,
      persona:g('persona'),scene:g('scene'),relation:g('relation'),preset:g('preset')})});
  document.querySelectorAll("audio").forEach(a=>a.remove());
  load();
}
document.addEventListener("keydown",e=>{
  if(e.target.tagName==="INPUT"||e.target.tagName==="TEXTAREA")return;
  const k=e.key.toLowerCase();
  if(k==="1")play('a');else if(k==="2")play('b');else if(k==="0")play('ref');
  else if(k==="a")pick('A');else if(k==="b")pick('B');else if(k==="t")pick('tie');else if(k==="r")pick('reject_both');
  else if(e.key==="Enter"&&!$("#submit").disabled)submit();
});
init();
</script></body></html>"""


def main() -> None:
    global SESSION, PAIRS, OUT_JSONL
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session JSON from build_ab_session.py")
    ap.add_argument("--out", default=None, help="results JSONL (default results/ab_results/<session>.jsonl)")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    SESSION = json.loads(Path(args.session).read_text())
    PAIRS = {p["pair_id"]: p for p in SESSION["pairs"]}
    OUT_JSONL = Path(args.out) if args.out else \
        Path("../results/ab_results") / f"{SESSION['session']}.jsonl"
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    load_done()

    print(f"session '{SESSION['session']}': {len(PAIRS)} pairs")
    print(f"results -> {OUT_JSONL}  (already judged: {len(DONE)})")
    print(f"open  http://{args.host}:{args.port}   (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
