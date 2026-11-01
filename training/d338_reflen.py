"""D338: 生成の床(SR)のうち 20 s の参照に固有の分。保留の実女声のうち参照(クリップ A を除く female-dataset)を 60 s 以上とれる話者で、
d275_heldenroll.py と同じ作り方(名前順につなぐ・Rust artic-enroll)で 20 s / 60 s の表を作り、ppgvoc18 の SR を同じ話者・同じ数で比べる(v1⊥)。
    uv run python d338_reflen.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d338
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN
from d6_unit_cov import load48

LENS = (20.0, 60.0)


def one(job: tuple) -> str:
    i, spk, ka, L, O = job
    en = O / f"L{int(L)}" / f"en_{i:03d}"
    if (en / "enroll.json").exists():
        return "skip"
    stem_a = ka[len(spk) + 1:]
    xs, tot = [], 0.0
    for w in sorted((P.ROOT / "female-dataset" / spk).glob("*.wav")):
        if w.stem == stem_a:
            continue
        x = load48(w)
        xs.append(x)
        tot += len(x) / 48000
        if tot >= L:
            break
    if tot < L:
        return f"short {spk} {tot:.1f}"
    en.parent.mkdir(parents=True, exist_ok=True)
    ref = en.parent / f"ref_{i:03d}.wav"
    sf.write(ref, np.concatenate(xs), 48000)
    r = subprocess.run([str(BIN), "artic-enroll", "-r", str(ref), "-o", str(en)], cwd=P.ROOT, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    return "ok" if r.returncode == 0 else f"err {spk} {r.stderr[-300:]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    W = P.H / "e2e" / "held_reflen"
    held = P.e2e_held()
    avail = [sum(sf.info(w).duration for w in (P.ROOT / "female-dataset" / s).glob("*.wav") if w.stem != ka[len(s) + 1:]) for s, ka, _ in held]
    keep = [i for i, t in enumerate(avail) if t >= max(LENS)]
    print("speakers", len(keep), "/", len(held), flush=True)
    with Pool(6) as pl:
        res = list(pl.imap(one, [(i, held[i][0], held[i][1], L, W) for i in keep for L in LENS]))
    print("enroll", {r.split()[0]: sum(x.split()[0] == r.split()[0] for x in res) for r in res}, [r for r in res if not r.startswith(("ok", "skip"))], flush=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"], P.CTX["ch"], P.CTX["adapt"] = bool(st.get("ctx", False)), int(st.get("ctx_ch", 192)), bool(st.get("adapt", False))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    groups = []
    for L in LENS:
        od = O / f"L{int(L)}"
        od.mkdir(parents=True, exist_ok=True)
        for i in keep:
            c, lg = P.load_dump(P.H / "e2e" / "fem", held[i][1])
            sf.write(od / f"h{i:04d}_t{i:03d}_SR.wav", P.render(g, c, lg, P.load_enroll(W / f"L{int(L)}" / f"en_{i:03d}"), dev), N.SR)
        groups.append(f"L{int(L)}={od}/h*_SR.wav")
    cp = O / "CP"
    cp.mkdir(exist_ok=True)
    for i in keep:
        f = P.ROOT / f"results/diag_e2e1/eval/h{i:04d}_t{i:03d}_CP.wav"
        (cp / f.name).unlink(missing_ok=True)
        (cp / f.name).symlink_to(f)
    groups.append(f"CP={cp}/h*_CP.wav")
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
