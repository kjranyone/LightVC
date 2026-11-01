"""D573(推論だけ): 製品 P57S0 を D571 の並行の組(d571par・93 組)に固定の +7 半音で描画し、同じ組の TAB0 / CONV / GU / REALK と比べる。規則は OVERNIGHT の D573。
手順は D570(d570_render.py)と同じ: Rust --no-f0-shift の書き出し(操作者の参照の末尾 12 s を前置き)・c0 の水準ずらし・操作者の遅い揺らぎ 0・有声の log f0 + k 半音。
目標の登録 = dev/d85_work/en_TTT・操作者の登録 = D571 の操作者の表と同じ男声 TTS(22 s 以上の連結)を artic-enroll。
出力 = <out>/k<k>/<pNNNN_tTTT>_OURS<k>.wav(ピーク 0.9)。
    uv run python d573_product.py --out <dev>/d571par --k 7
"""
from __future__ import annotations

import argparse
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
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
MALE = ROOT / "data/male_tts_v4l"
CKPT = ROOT / "results/diag_ppgvoc57/product_002500_m0.pt"
PRE = 12.0


def enroll_job(j: tuple) -> None:
    import subprocess
    w, o = j
    if (Path(o) / "enroll.json").exists():
        return
    r = subprocess.run([str(BIN), "artic-enroll", "-r", str(w), "-o", str(o)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=7)
    a = ap.parse_args()
    od = Path(a.out)
    en_dir = od.parent / "d85_work"
    work = od / "d573_work"
    work.mkdir(parents=True, exist_ok=True)
    ed = od / "d573_dump"
    ed.mkdir(parents=True, exist_ok=True)
    ko = od / f"k{a.k}"
    items = json.loads((od / "items.json").read_text())
    n0 = int(PRE * N.SR)
    ejobs = []
    for spk in sorted({it["spk"] for it in items}):
        w = work / f"op_{spk}.wav"
        if not w.exists():
            xs, tot = [], 0.0
            for f in sorted(MALE.glob(f"{spk}/t*.wav")):
                if tot >= 22.0:
                    break
                x = load48(f).astype(np.float32)
                xs.append(x)
                tot += len(x) / N.SR
            sf.write(w, np.concatenate(xs), N.SR)
        ejobs.append((w, work / f"op_{spk}"))
    with Pool(4) as pl:
        list(pl.imap_unordered(enroll_job, ejobs))
    djobs = []
    for it in items:
        k = f"p{it['item']:04d}_t{it['ti']:03d}"
        w = work / f"{k}_in.wav"
        if not w.exists():
            pre = load48(work / f"op_{it['spk']}.wav")[-n0:]
            sf.write(w, np.concatenate([pre, load48(od / f"{k}_SRC.wav")]).astype(np.float32), N.SR)
        djobs.append(([BIN, "artic", "--enroll", en_dir / f"en_{it['ti']:03d}", "--operator-enroll", work / f"op_{it['spk']}", "--med-from-operator"] + COM + ["--no-f0-shift", "-i", w], ed / f"{k}.cond", ed / f"{k}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(job, djobs))
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
    nm = f"OURS{a.k}"
    done = 0
    for it in items:
        k = f"p{it['item']:04d}_t{it['ti']:03d}"
        o = ko / f"{k}_{nm}.wav"
        if o.exists():
            continue
        c, lg = P.load_dump(ed, k)
        pre = c[:2400, 0]
        mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
        c = c.copy()
        c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
        c[:, 25] = np.where(c[:, 26] > 0.5, c[:, 25] + a.k * np.log(2) / 12, c[:, 25])
        xin = load48(work / f"{k}_in.wav").astype(np.float32)
        cvf = P.cvin_frames(torch.from_numpy(TC.prime(xin))[None].to(dev), len(lg))
        en = P.load_enroll(en_dir / f"en_{it['ti']:03d}")
        y = P.render(g, c, lg, en, dev, T_op=P.load_enroll(work / f"op_{it['spk']}")[0], cvf=cvf)
        nx = len(xin) - n0
        y = y[n0:n0 + nx]
        sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        done += 1
    print("rendered", done, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
