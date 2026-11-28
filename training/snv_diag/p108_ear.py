"""段 P108(学習なし・耳の材料): 製品の回路の出力を、描画器に渡す高さと有声だけ替えて描く(prod = 今の追跡・g2 = 段 P107 の直し 2・v4 = 試作の追跡 snv_f0v4)。半音 +12。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p108_ear.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for k, v in {"SNV_V4_MODE": "multi", "SNV_V4_SPLIT": "400", "SNV_V4_UCOST": "0.5", "SNV_V4_UEXT": "0.35", "SNV_V4_UON": "0.2", "SNV_V4_EXCL": "1", "SNV_V4_BETA": "0.05"}.items():
    os.environ.setdefault(k, v)

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import p107_guard as GD
import snv_f0_guard as FG
import snv_f0v4 as V
import snv_out_guard as OG
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    k = 12
    out = ROOT / "results/snv_p108/ear"
    out.mkdir(parents=True, exist_ok=True)
    picks = json.load(open(ROOT / "results/snv_p107/edges_k12.json"))["items"]
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    gen = EV.load_gen()
    S = EV.Ens(E.SYS, gen)
    tg = [PB.utt(rd(p)) for p in sorted((PB.IN / "tg").glob("*.f32"))]
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    src = {f"real{i + 1}": (rd(PB.IN / sp / "in.f32"), [rd(p) for p in sorted((PB.IN / sp / "op").glob("*.f32"))]) for i, sp in enumerate(PB.SPS)}
    for v in ("em594", "em595"):
        src[f"fam_{v}"] = (lw(ROOT / f"data/snv_e1/{v}/t05.wav"), [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)])
    for v in ("em595", "em596"):
        src[f"scream_{v}"] = (lw(ROOT / f"data/snv_e1nv/{v}/scream_0.wav"), [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)])
    res: dict = {}
    for name, (x, ops) in src.items():
        u, enr = PB.utt(x), [PB.utt(o) for o in ops]
        lv = OG.speech_level([e["wav"] for e in enr])
        p0 = S.parts[0]
        pu = p0.prep(u)
        _, yo = S.convert(u, enr, st, k, with_y=True)
        f0 = pu["f0"].astype(np.float32)
        n = len(f0)
        fv = V.f0_track(torch.from_numpy(u["wav"][: P.n_frames(len(u["wav"])) * P.H]))[0].numpy()
        fv = np.concatenate([fv, np.zeros(max(0, n - len(fv)), np.float32)])[:n]
        med = FG.median_f0([p0.prep(e)["f0"] for e in enr])
        g2 = FG.unhold(FG.clean(f0, med), pu["ap"][:n]) if "ap" in pu else FG.clean(f0, med)
        r = {}
        for nm, f in (("prod", f0), ("g2", g2), ("v4", fv)):
            w = OG.apply(u["wav"], np.asarray(p0.render(GD.cond(pu, f), yo[:, :29], k, None), np.float32), lv, p_out)
            sf.write(out / f"{name}_{nm}.wav", w, EV.SR, subtype="PCM_16")
            for i, pk in enumerate(picks.get(name, {}).get("picks", []), 1):
                a, b = max(0, int(pk["t"] * 200) - 120), int(pk["t"] * 200) + 120
                sf.write(out / f"{name}_{i}_{nm}.wav", w[a * 240 : b * 240], EV.SR, subtype="PCM_16")
            r[nm] = {"voiced": round(float((f > 0).mean()), 3), "jumps": GD.E_j(f)}
        sf.write(out / f"{name}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    (ROOT / "results/snv_p108/ear.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
