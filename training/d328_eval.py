"""D328 の計測: dev(男声 TTS v4 Large)と devf(同じ TTS・同じ文の女声)を同じ系で描画し、v1⊥(60 本どうし)と 2 標本の直接判別(組ごとに文が一致)で男声の分を測る。
描画の名前: dev → p00NN_tTTT_CV.wav・devf → p01NN_tTTT_CV.wav(判別の話者の割り当てを分けるため)。
    uv run python d328_eval.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d328
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=100)
    ap.add_argument("--ppg", default="", help="推論の事後の形の上書き(hard / tau0.5 など・D329・診断)")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    od = O / "conv"
    od.mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = a.ppg or st.get("ppg", "soft")
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    P.CTX["adapt"] = bool(st.get("adapt", False))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    rows = []
    for s_, off in (("dev", 0), ("devf", 100)):
        ed = P.H / "e2e" / f"eval_{s_}"
        for r in json.loads((P.H / s_ / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, lg = P.load_dump(ed, k)
            sf.write(od / f"p{r['pi'] + off:04d}_t{r['ti']:03d}_CV.wav", P.render(g, c, lg, P.load_enroll(P.H / s_ / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(P.H / s_, k) if P.OPSLOW["on"] else None), N.SR)
            rows.append({"pi": r["pi"] + off, "kind": "F" if off else "M", "spk": r["spk"]})
    (O / "spk.json").write_text(json.dumps({"pairs": rows}, ensure_ascii=False))
    print("rendered", len(rows), flush=True)
    T = Path(__file__).parent
    groups = [f"MT={od}/p00*_CV.wav", f"FT={od}/p01*_CV.wav", f"MT_SRC={P.H}/dev/d56/p00*_SRC.wav", f"FT_SRC={P.H}/devf/d56/p00*_SRC.wav"]
    r1 = subprocess.run([sys.executable, str(T / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r1.stdout[-3000:], r1.stderr[-1500:] if r1.returncode else "", flush=True)
    pairs = [f"TTSM_CV={od}/p01*_CV.wav|{od}/p00*_CV.wav|{O / 'spk.json'}"]
    r2 = subprocess.run([sys.executable, str(T / "d326_pair.py"), "--perm", str(a.perm), "--pairs", *pairs, "--out", str(O / "pair.json")], capture_output=True, text=True)
    print(r2.stdout[-3000:], r2.stderr[-1500:] if r2.returncode else "", flush=True)
    return r1.returncode or r2.returncode


if __name__ == "__main__":
    raise SystemExit(main())
