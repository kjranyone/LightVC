"""盲検A/B試聴サーバ(127.0.0.1のみ・SSHポート転送で使う)。

results/earbattery を配信し、鍵(_key*)と代理予測(_proxy_prediction*)は403で遮断する。
POST /save_answers → d1_ab/answers_ear.json、POST /save_probe → chorus_probe/answers_ear.json、POST /save_nrft → nrft_ab/answers_ear.json(前版は .bak)。

    uv run python serve_ab.py --port 8765
    手元: ssh -L 8765:127.0.0.1:8765 <接続先>  →  http://localhost:8765/d1_ab/listen.html
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent.parent / "results/earbattery"
SAVES = {"/save_answers": ROOT / "d1_ab/answers_ear.json",
         "/save_probe": ROOT / "chorus_probe/answers_ear.json",
         "/save_nrft": ROOT / "nrft_ab/answers_ear.json",
         "/save_ceil": ROOT / "ceil_ab/answers_ear.json",
         "/save_dec2": ROOT / "dec2_ab/answers_ear.json",
         "/save_artic_s04": ROOT / "artic_s04/answers_ear.json",
         "/save_dspvc": ROOT / "dspvc_p0/answers_ear.json",
         "/save_ddsp_chorus": ROOT / "ddsp_chorus/answers_ear.json",
         "/save_ddsp_sim": ROOT / "ddsp_sim/answers_ear.json",
         "/save_ddsp_ab": ROOT / "ddsp_ab/answers_ear.json",
         "/save_nvoc": ROOT / "nvoc_ab/answers_ear.json",
         "/save_zsvc": ROOT / "zsvc_ab/answers_ear.json",
         "/save_zsvc2": ROOT / "zsvc_ab2/answers_ear.json",
         "/save_a2vc": ROOT / "a2vc_s1_ab/answers_ear.json",
         "/save_spk_ear": ROOT / "spk_ear/answers_ear.json",
         "/save_psola": ROOT / "psola_ear/answers_ear.json",
         "/save_rvoc": ROOT / "rvoc_ab/answers_ear.json",
         "/save_rvoc2": ROOT / "rvoc_ab2/answers_ear.json",
         "/save_rvoc3": ROOT / "rvoc_ab3/answers_ear.json",
         "/save_conv1": ROOT / "conv_ab1/answers_ear.json"}
DENY = ("_key", "_proxy_prediction")


class Handler(SimpleHTTPRequestHandler):
    def _denied(self) -> bool:
        p = unquote(self.path).lower()
        return any(d in p for d in DENY)

    def do_GET(self):
        if self._denied():
            self.send_error(403, "blind: key/prediction files are not served")
            return
        super().do_GET()

    def do_HEAD(self):
        if self._denied():
            self.send_error(403)
            return
        super().do_HEAD()

    def list_directory(self, path):
        self.send_error(403, "directory listing disabled")
        return None

    def do_POST(self):
        ans = SAVES.get(self.path)
        if ans is None:
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0 or n > 2_000_000:
            self.send_error(400)
            return
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "bad json")
            return
        data["server_saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if ans.exists():
            shutil.copy2(ans, ans.with_suffix(".json.bak"))
        ans.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        body = json.dumps({"ok": True, "path": str(ans.relative_to(ROOT.parent.parent))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=str(ROOT)))
    print(f"serving {ROOT} on http://127.0.0.1:{a.port}/d1_ab/listen.html", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
