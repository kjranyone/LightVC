"""D287: 交差の組の偽物らしさ(D284・D286)のどの不一致が効くかを、自己再構成 SR(保留 30 人・diag_ppgvoc1 の 5k)の入力を 1 か所だけ崩して切り分ける(学習なし)。
xtab = 表と周期性の定数を別の保留の話者(1 つずらし)の登録に替え、高さをその話者の声域へ(登録の mu の差)/ xtab_nof0 = 表だけ替え高さはそのまま /
f0up3 / f0dn3 = 自分の表のまま高さだけ ±3 半音。
    uv run python d287_srmismatch.py --ckpt ../results/diag_ppgvoc1/last_ema.pt --out ../results/conv_p0/d287
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="xtab,xtab_nof0,f0up3,f0dn3")
    ap.add_argument("--ffself", default="", help="この組(devff 等)の操作者を本人に戻す: 表 = 操作者の登録(op_PPPP)・高さの移動(目標の mu − 操作者の mu)を戻す")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    if a.ffself:
        ed, S = P.H / "e2e" / f"eval_{a.ffself}", P.H / a.ffself
        groups = []
        for v in ("asis", "self"):
            od = O / v
            od.mkdir(parents=True, exist_ok=True)
            for f in sorted(ed.glob("p0*.cond")):
                pi, ti = int(f.stem[1:5]), int(f.stem.split("_t")[1][:3])
                c, lg = P.load_dump(ed, f.stem)
                en_t, en_o = S / "d85_work" / f"en_{ti:03d}", S / "d85_work" / f"op_{pi:04d}"
                if v == "self":
                    mt, mo = (float(json.loads((d / "enroll.json").read_text())["mu"]) for d in (en_t, en_o))
                    vo = c[:, 26] > 0.5
                    c = c.copy()
                    c[vo, 25] -= mt - mo
                    spk = P.load_enroll(en_o)
                else:
                    spk = P.load_enroll(en_t)
                sf.write(od / f"{f.stem}_PV.wav", P.render(g, c, lg, spk, dev), N.SR)
            groups.append(f"{v}={od}/p*_PV.wav")
            print("rendered", v, flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
        print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
        return r.returncode
    held = P.e2e_held()
    sr_items, _ = P.eval_items(held)
    en = [P.H / "e2e" / "held_en" / f"en_{i:03d}" for i in range(len(held))]
    mu = [float(json.loads((d / "enroll.json").read_text())["mu"]) for d in en]
    groups = []
    for v in a.variants.split(","):
        od = O / v
        od.mkdir(parents=True, exist_ok=True)
        for i, (nm, c, lg, spk) in enumerate(sr_items):
            c = c.copy()
            j = (i + 1) % len(sr_items)
            vo = c[:, 26] > 0.5
            if v.startswith("xtab"):
                spk = sr_items[j][3]
                if v == "xtab":
                    c[vo, 25] += mu[j] - mu[i]
            elif v == "f0up3":
                c[vo, 25] += 3 / 12 * np.log(2)
            elif v == "f0dn3":
                c[vo, 25] -= 3 / 12 * np.log(2)
            sf.write(od / f"{nm}.wav", P.render(g, c, lg, spk, dev), N.SR)
        groups.append(f"{v}={od}/h*_SR.wav")
        print("rendered", v, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
