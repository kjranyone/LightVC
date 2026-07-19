from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).parent))
import kansei_proxies as kp

BANDS = [("low", "0–1k"), ("mid", "1–4k"), ("presence", "4–5k"),
         ("sibilance", "5–9k"), ("brilliance", "9–16k"), ("air", "16k+")]
ROLE_PRIORITY = ["orig", "original", "reference", "source", "target", "oracle",
                 "ceiling", "base", "finetuned", "knnvc", "knn", "gen", "generated"]
GOOD_ROLES = {"orig", "original", "reference", "oracle", "target"}
BAD_ROLES = {"gen", "generated"}

ROOTS = []
MODEL = None
CACHE = {}
TITLE = "LightVC listen"


def analyze_file(path) -> dict:
    key = str(path)
    if key not in CACHE:
        y = kp.load_wav(path)
        m = kp.analyze(y, full=False)
        CACHE[key] = {
            "bands": [round(m[f"band_{k}"], 5) for k, _ in BANDS],
            "hf_ratio": round(m["hf_ratio"], 4),
            "cliff": round(m["eight_k_cliff"], 3),
            "centroid": round(m["centroid_hz"], 0),
        }
    return CACHE[key]


def build_dir(di: int, directory, glob: str) -> list:
    groups = {}
    for w in sorted(Path(directory).glob(glob)):
        stem = w.stem
        group, role = (stem.rsplit("_", 1) if "_" in stem else (stem, "clip"))
        groups.setdefault(group, []).append((role, w))
    out = []
    for group, items in groups.items():
        items.sort(key=lambda it: (ROLE_PRIORITY.index(it[0]) if it[0] in ROLE_PRIORITY else 99, it[0]))
        clips = []
        for role, w in items:
            met = analyze_file(w)
            flag = ""
            if role in BAD_ROLES and met["hf_ratio"] < 0.02:
                flag = "muffled?"
            clips.append({"role": role, "d": di, "rel": w.name,
                          "good": role in GOOD_ROLES, "bad": role in BAD_ROLES,
                          "flag": flag, **met})
        out.append({"group": group, "clips": clips})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if u.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif u.path == "/api/model":
            self._send(200, "application/json; charset=utf-8",
                       json.dumps({"title": TITLE, "sections": MODEL}, ensure_ascii=False).encode())
        elif u.path == "/audio":
            q = parse_qs(u.query)
            try:
                di = int(q.get("d", ["-1"])[0])
            except ValueError:
                di = -1
            rel = unquote(q.get("file", [""])[0])
            if 0 <= di < len(ROOTS):
                path = (ROOTS[di] / rel).resolve()
                if ROOTS[di] in path.parents and path.exists():
                    self._send(200, "audio/wav", path.read_bytes())
                    return
            self._send(404, "text/plain", b"not found")
        else:
            self._send(404, "text/plain", b"not found")


PAGE = r"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LightVC listen</title>
<style>
:root{--bg:#0e1116;--panel:#171a21;--panel2:#1d222c;--line:#2a3140;--tx:#e6e9ef;--mut:#8a93a3;
--sig:#4fd1c5;--warn:#f0a742;--good:#5ec98a;--hf:#4fd1c5;--lf:#3a4658;color-scheme:dark light}
@media (prefers-color-scheme:light){:root{--bg:#f4f6f9;--panel:#fff;--panel2:#eef1f6;--line:#d8dee8;
--tx:#141922;--mut:#5a6675;--sig:#0e9c8e;--warn:#c67a12;--good:#2f9e63;--hf:#0e9c8e;--lf:#c2ccd9}}
:root[data-theme=light]{--bg:#f4f6f9;--panel:#fff;--panel2:#eef1f6;--line:#d8dee8;--tx:#141922;
--mut:#5a6675;--sig:#0e9c8e;--warn:#c67a12;--good:#2f9e63;--hf:#0e9c8e;--lf:#c2ccd9}
:root[data-theme=dark]{--bg:#0e1116;--panel:#171a21;--panel2:#1d222c;--line:#2a3140;--tx:#e6e9ef;
--mut:#8a93a3;--sig:#4fd1c5;--warn:#f0a742;--good:#5ec98a;--hf:#4fd1c5;--lf:#3a4658}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:26px 20px 64px}
h1{font-size:22px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 8px;font:13px ui-monospace,monospace}
.dir{margin-top:30px;border-top:2px solid var(--sig);padding-top:12px}
.dir>h2{font-size:16px;margin:0 0 2px;color:var(--sig)}
.dir>.dsub{color:var(--mut);font:12px ui-monospace,monospace;margin:0 0 12px}
.gp{margin:14px 0}.gp>h3{font:600 12px ui-monospace,monospace;color:var(--mut);margin:0 0 7px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px;
display:flex;flex-direction:column;gap:9px}
.card.bad{border-color:color-mix(in srgb,var(--warn) 55%,var(--line))}
.card.good{border-color:color-mix(in srgb,var(--good) 45%,var(--line))}
.role{font-weight:600;font-size:14px;display:flex;gap:8px;align-items:center}
.flag{font:600 11px ui-monospace,monospace;background:var(--warn);color:#141922;padding:2px 7px;border-radius:6px}
audio{width:100%;height:34px}
.bars{display:flex;gap:5px;align-items:flex-end;height:50px}
.bcol{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px}
.btrack{width:100%;height:40px;background:var(--panel2);border-radius:3px;display:flex;align-items:flex-end;overflow:hidden}
.bfill{width:100%;background:var(--lf);border-radius:3px 3px 0 0}.bfill.hf{background:var(--hf)}
.bcol span{font:500 9px ui-monospace,monospace;color:var(--mut)}
.nums{display:flex;flex-wrap:wrap;gap:3px 12px;font:12px ui-monospace,monospace;color:var(--mut)}
.nums b{color:var(--tx);font-variant-numeric:tabular-nums}
button.theme{float:right;background:var(--panel);color:var(--mut);border:1px solid var(--line);
border-radius:8px;padding:5px 10px;cursor:pointer;font:12px ui-monospace,monospace}
.legend{margin-top:16px;font:12px ui-monospace,monospace;color:var(--mut)}
.legend b{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:5px}
</style></head><body><div class="wrap">
<button class="theme" onclick="tog()">◐ theme</button>
<h1 id="title">LightVC listen</h1><p class="sub" id="sub"></p><div id="app"></div>
<div class="legend"><span><b style="background:var(--hf)"></b>高域帯(5–16kHz+)</span>
&nbsp;&nbsp;<span>16kHzのクリップは8kで切れて見えます — 判断は音で</span></div>
</div><script>
function tog(){const r=document.documentElement;r.setAttribute("data-theme",
r.getAttribute("data-theme")==="light"?"dark":"light");}
function bars(bs){const mx=Math.max(...bs)||1;const L=["0–1k","1–4k","4–5k","5–9k","9–16k","16k+"];
return '<div class="bars">'+bs.map((v,i)=>`<div class="bcol"><div class="btrack"><div class="bfill ${i>=3?"hf":""}" style="height:${(Math.sqrt(v/mx)*100).toFixed(0)}%"></div></div><span>${L[i]}</span></div>`).join("")+'</div>';}
function card(c){const cls="card"+(c.bad?" bad":(c.good?" good":""));
const flag=c.flag?`<span class="flag">${c.flag}</span>`:"";
return `<div class="${cls}"><div class="role">${c.role}${flag}</div>
<audio controls preload="none" src="/audio?d=${c.d}&file=${encodeURIComponent(c.rel)}"></audio>
${bars(c.bands)}<div class="nums"><span>hf <b>${c.hf_ratio.toFixed(4)}</b></span>
<span>cliff <b>${c.cliff.toFixed(3)}</b></span><span>cent <b>${c.centroid.toFixed(0)}</b></span></div></div>`;}
async function load(){const m=await (await fetch("/api/model")).json();
document.getElementById("title").textContent=m.title;
document.getElementById("sub").textContent=m.sections.length+" セクション — 棒=6帯域スペクトル(シアン=高域)";
document.getElementById("app").innerHTML=m.sections.map(s=>
`<section class="dir"><h2>${s.label}</h2><div class="dsub">${s.dir}</div>`+
s.groups.map(g=>`<div class="gp"><h3>${g.group}</h3><div class="grid">${g.clips.map(card).join("")}</div></div>`).join("")+
`</section>`).join("");}
load();
</script></body></html>"""


def main() -> None:
    global ROOTS, MODEL, TITLE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+", required=True, help="one or more wav dirs")
    ap.add_argument("--labels", nargs="*", default=None, help="section labels (per dir)")
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--title", default="LightVC listen — 比較")
    ap.add_argument("--port", type=int, default=8772)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    TITLE = args.title
    MODEL = []
    for di, d in enumerate(args.dir):
        root = Path(d).resolve()
        if not root.exists():
            print(f"skip missing: {root}")
            continue
        ROOTS.append(root)
        label = (args.labels[di] if args.labels and di < len(args.labels) else root.name)
        print(f"analyzing [{len(ROOTS)-1}] {label}: {root}")
        MODEL.append({"label": label, "dir": str(root), "groups": build_dir(len(ROOTS) - 1, root, args.glob)})

    n = sum(len(g["clips"]) for s in MODEL for g in s["groups"])
    print(f"{len(MODEL)} sections / {n} clips")
    print(f"open  http://{args.host}:{args.port}   (Ctrl-C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
