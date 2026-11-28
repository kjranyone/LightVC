"""段 P110(学習なし・耳の材料): (A) 悲鳴・喘ぎ・笑いの高さの写し方 = 固定の +12 / +0 / 高い声ほど上げ幅を縮める形。(B) かすれ = 回路の出す周期性の行(帯 0・1)を目標の声の代表値まで底上げする形(今の追跡 / 新しい追跡)。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p110_ear.py
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
import snv_f0v4 as V
import snv_out_guard as OG
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
LO, HI = 300.0, 900.0


def taper(f0: np.ndarray, k: float) -> np.ndarray:
    w = np.clip((np.log(np.maximum(f0, 1.0)) - np.log(LO)) / (np.log(HI) - np.log(LO)), 0.0, 1.0)
    return np.where(f0 > 0, f0 * 2.0 ** (k * (1.0 - w) / 12.0), 0.0).astype(np.float32)


def main() -> None:
    out = ROOT / "results/snv_p110/ear"
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    S = EV.Ens(E.SYS, EV.load_gen())
    tg = [PB.utt(rd(p)) for p in sorted((PB.IN / "tg").glob("*.f32"))]
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    pt = np.concatenate([t["y"][25:27][:, t["f0"] > 0] for t in tg], 1)
    floor = 0.8 * np.median(pt, 1)
    p0 = S.parts[0]
    res: dict = {"per_floor": floor.tolist(), "taper": [LO, HI], "A": {}, "B": {}}

    def draw(pu: dict, f: np.ndarray, yo: torch.Tensor, lv: float, u: dict) -> np.ndarray:
        return OG.apply(u["wav"], np.asarray(p0.render(GD.cond(pu, f), yo[:, :29], 0, None), np.float32), lv, p_out)
    for v, c in (("em595", "scream"), ("em596", "scream"), ("em595", "moan"), ("em596", "laugh"), ("em597", "cry")):
        u = PB.utt(lw(ROOT / f"data/snv_e1nv/{v}/{c}_0.wav"))
        enr = [PB.utt(lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
        lv = OG.speech_level([e["wav"] for e in enr])
        pu = p0.prep(u)
        _, yo = S.convert(u, enr, st, 12, with_y=True)
        f0 = pu["f0"].astype(np.float32)
        name = f"{v}_{c}"
        for nm, f in (("k12", f0 * 2.0), ("k0", f0), ("taper", taper(f0, 12.0))):
            sf.write(out / f"A_{name}_{nm}.wav", draw(pu, f.astype(np.float32), yo, lv, u), EV.SR, subtype="PCM_16")
        sf.write(out / f"A_{name}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
        vv = f0 > 0
        res["A"][name] = {"src_f0_med": int(np.median(f0[vv])), "k12_med": int(np.median(f0[vv]) * 2), "taper_med": int(np.median(taper(f0, 12.0)[vv]))}
        print("A", name, res["A"][name], flush=True)
    for i, sp in enumerate(PB.SPS):
        x = rd(PB.IN / sp / "in.f32")
        u, enr = PB.utt(x), [PB.utt(rd(p)) for p in sorted((PB.IN / sp / "op").glob("*.f32"))]
        lv = OG.speech_level([e["wav"] for e in enr])
        pu = p0.prep(u)
        _, yo = S.convert(u, enr, st, 12, with_y=True)
        f0 = pu["f0"].astype(np.float32)
        n = len(f0)
        fv = V.f0_track(torch.from_numpy(u["wav"][: P.n_frames(len(u["wav"])) * P.H]))[0].numpy()
        fv = np.concatenate([fv, np.zeros(max(0, n - len(fv)), np.float32)])[:n]
        yb = yo.clone()
        fl = torch.from_numpy(floor.astype(np.float32)).to(yo.device)[None, :, None]
        yb[:, 25:27] = torch.maximum(yo[:, 25:27], fl)
        name = f"real{i + 1}"
        for nm, f, y in (("prod", f0, yo), ("per", f0, yb), ("v4", fv, yo), ("v4per", fv, yb)):
            sf.write(out / f"B_{name}_{nm}.wav", draw(pu, taper(f, 12.0) if False else (f * 2.0).astype(np.float32), y, lv, u), EV.SR, subtype="PCM_16")
        po = yo[0, 25:27].detach().cpu().numpy()[:, f0 > 0]
        res["B"][name] = {"per01_out_med": np.round(np.median(po, 1), 2).tolist(), "below_floor": np.round((po < floor[:, None]).mean(1), 2).tolist()}
        print("B", name, res["B"][name], flush=True)
    sf.write(out / "target_ref.wav", tg[0]["wav"][: EV.SR * 12], EV.SR, subtype="PCM_16")
    (ROOT / "results/snv_p110/ear.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
