"""段 P119(学習なし・耳の材料): 同じ変換を、描画器だけ替えて描く(A = 製品・C / S / Y = 製品の条件で学び足した重み)。
    cd training && SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_FEAT=data/snv_p_feat_p100 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p119_ear.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import snv_out_guard as OG
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
ARMS = {"A": "", "C": "results/snv_p114_m0/snap/ema_10k.pt", "S": "results/snv_p97_s240/snap/ema_30k.pt", "Y": "results/snv_p97_y240/snap/ema_30k.pt"}
TGT = "ef100"


def main() -> None:
    out = ROOT / "results/snv_p119/ear"
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    refs = [PB.utt(lw(ROOT / f"data/snv_e1/{TGT}/t0{j}.wav")) for j in (0, 1, 2)]
    ft = float(np.median(np.concatenate([r["f0"][r["f0"] > 0] for r in refs])))
    p_out = OG.speech_level([u["wav"] for u in refs])
    src = {}
    for v in ("ef032", "em500"):
        src[v] = (lw(ROOT / f"data/snv_e1/{v}/t05.wav"), [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)])
    for i, sp in enumerate(PB.SPS[:2]):
        src[f"real{i + 1}"] = (rd(PB.IN / sp / "in.f32")[: EV.SR * 12], [rd(p) for p in sorted((PB.IN / sp / "op").glob("*.f32"))])
    src["scream"] = (lw(ROOT / "data/snv_e1nv/em595/scream_0.wav"), [lw(ROOT / f"data/snv_e1/em595/t0{j}.wav") for j in (0, 1, 2)])
    ks: dict = {}
    for arm, ck in ARMS.items():
        os.environ.pop("SNV_RCK", None)
        if ck:
            os.environ["SNV_RCK"] = ck
        gen = EV.load_gen()
        S = EV.Ens(E.SYS, gen)
        st = S.structure("t" + TGT, refs)
        for name, (x, ops) in src.items():
            u, enr = PB.utt(x), [PB.utt(o) for o in ops]
            fs = float(np.median(np.concatenate([e["f0"][e["f0"] > 0] for e in enr])))
            k = float(np.clip(round(12 * np.log2(ft / fs)), -12, 12))
            ks[name] = k
            w = OG.apply(u["wav"], np.asarray(S.convert(u, enr, st, k), np.float32), OG.speech_level([e["wav"] for e in enr]), p_out)
            sf.write(out / f"{name}_{arm}.wav", np.clip(w, -0.99, 0.99), EV.SR, subtype="PCM_16")
            if arm == "A":
                sf.write(out / f"{name}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
        print(arm, "env_smooth", gen.env_smooth, "delay", gen.delay, flush=True)
    sf.write(out / "target_ref.wav", refs[0]["wav"], EV.SR, subtype="PCM_16")
    json.dump({"k": ks, "arms": ARMS, "target": TGT}, open(ROOT / "results/snv_p119/ear.json", "w"), ensure_ascii=False, indent=1)
    print(ks)


if __name__ == "__main__":
    main()
