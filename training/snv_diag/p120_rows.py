"""段 P120(学習なし): K3G の出力の条件と正解の条件を行ごとに差し替えて描く(源 = 擬似の操作者の版・目標 = 元の声・正解 = 元の発話)。
    cd training && SNV_WIN_CAP_MS=7.5 SNV_W_YIN=600 SNV_TAU_UV=500 SNV_FEAT=data/snv_p_feat_p100 SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p120_rows.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import p119_copy as C9
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
REND = {"Y": "results/snv_p97_y240/snap/ema_30k.pt", "A": ""}
VOICES = ("ef100", "ef032", "ef300")


def main() -> None:
    out = ROOT / "results/snv_p120/ear"
    out.mkdir(parents=True, exist_ok=True)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    res: dict = {}
    stats: dict = {}
    for rn, ck in REND.items():
        os.environ.pop("SNV_RCK", None)
        if ck:
            os.environ["SNV_RCK"] = ck
        gen = EV.load_gen()
        S = EV.Ens(E.SYS, gen)
        for v in VOICES:
            refs = [PB.utt(lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
            st = S.structure("t" + v, refs)
            tgt = PB.utt(lw(ROOT / f"data/snv_e1/{v}/t05.wav"))
            u = PB.utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t05.wav"))
            enr = [PB.utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
            med = lambda us: float(np.median(np.concatenate([q["f0"][q["f0"] > 0] for q in us])))
            k = float(12 * np.log2(med(refs) / med(enr)))
            _, yo = S.convert(u, enr, st, k, with_y=True)
            n = min(yo.shape[-1], tgt["y"].shape[1])
            K = yo[:, :29, :n].float()
            T = torch.from_numpy(tgt["y"][:29, :n])[None].to(EV.DEV)
            f0 = tgt["f0"][:n].astype(np.float32)
            vv = torch.from_numpy((f0 > 0).astype(np.float32))[None].to(EV.DEV)
            lf = torch.from_numpy(np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32))[None].to(EV.DEV)
            mix = {"T": T, "K": K, "Kc0": torch.cat([K[:, :1], T[:, 1:]], 1), "Kenv": torch.cat([T[:, :1], K[:, 1:25], T[:, 25:]], 1), "Kper": torch.cat([T[:, :25], K[:, 25:]], 1)}
            if rn == "Y":
                m = vv[0] > 0
                d2 = lambda z: float((z[..., 2:] - 2 * z[..., 1:-1] + z[..., :-2]).abs().mean())
                stats[v] = {"c0 L1": round(float((K[0, 0] - T[0, 0]).abs().mean()), 4), "env1-24 L1(有声)": round(float((K[0, 1:25, m] - T[0, 1:25, m]).abs().mean()), 4), "per L1(有声)": round(float((K[0, 25:, m].clamp(0, 1.5) - T[0, 25:, m].clamp(0, 1.5)).abs().mean()), 4),
                            "per 平均(K / T・有声)": [round(float(K[0, 25:, m].clamp(0, 1.5).mean()), 3), round(float(T[0, 25:, m].clamp(0, 1.5).mean()), 3)],
                            "env1-24 の時間の 2 階差(K / T)": [round(d2(K[0, 1:25]), 4), round(d2(T[0, 1:25]), 4)], "c0 の 2 階差(K / T)": [round(d2(K[0, :1]), 4), round(d2(T[0, :1]), 4)],
                            "env1-24 の時間の標準偏差(K / T・有声)": [round(float(K[0, 1:25, m].std(-1).mean()), 4), round(float(T[0, 1:25, m].std(-1).mean()), 4)]}
                sf.write(out / f"{v}_orig.wav", tgt["wav"], EV.SR, subtype="PCM_16")
                sf.write(out / f"{v}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
            for name, y in mix.items():
                cond = torch.cat([EV.smooth_env(gen, y[:, :25]), lf[:, None], vv[:, None], y[:, 25:29].clamp(0, 1.5) * vv[:, None]], 1)
                w = np.asarray(EV.draw(gen, cond, torch.from_numpy(f0)[None].to(EV.DEV), None), np.float32)
                sf.write(out / f"{v}_{rn}_{name}.wav", np.clip(w, -0.99, 0.99), EV.SR, subtype="PCM_16")
                res.setdefault(rn, {}).setdefault(name, []).append(round(C9.lm(tgt["wav"], w), 3))
    for v, s in stats.items():
        print(v, json.dumps(s, ensure_ascii=False))
    for rn in res:
        for name, vals in res[rn].items():
            print(rn, name.ljust(5), vals, round(float(np.mean(vals)), 3))
    json.dump({"logmel": res, "stats": stats}, open(ROOT / "results/snv_p120/rows.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
