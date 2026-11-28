"""段 P106(results/snv_p106/prereg.yaml・学習なし): 半音 +12 の出力が「高さを上げただけの声」にどれだけ近いかを、源の種類と錨の有無で比べる。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p106_shift.py [k=12]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import snv_out_guard as OG
import snv_p1_eval as EV
from snv_a0_measure import to16

ROOT = Path(__file__).resolve().parent.parent.parent
G = "+mask:snv_p45_gate_t80"
SYS = {"prod": "ens:" + "|".join(f"{t}{G}+fb:anch" for t in ("snv_p24_c", "snv_p30_c", "snv_p91e_c")), "noanch": "ens:" + "|".join(f"{t}{G}" for t in ("snv_p24_c", "snv_p30_c", "snv_p91e_c"))}


def main() -> None:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    k = int(opt.get("k", 12))
    out = ROOT / "results/snv_p106"
    (out / "ear").mkdir(parents=True, exist_ok=True)
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": EV.DEV})
    def emb(w: np.ndarray) -> np.ndarray:
        e = ec.encode_batch(torch.from_numpy(to16(w.astype(np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV))[0, 0].detach().cpu().numpy()
        return e / np.linalg.norm(e)
    um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(EV.DEV).eval()
    mos = lambda w: float(um(torch.from_numpy(to16(np.asarray(w, np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV), 16000).item())
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    gen = EV.load_gen()
    tg = [PB.utt(rd(p)) for p in sorted((PB.IN / "tg").glob("*.f32"))]
    et = np.mean([emb(u["wav"]) for u in tg], 0)
    et /= np.linalg.norm(et)
    p_out = OG.speech_level([u["wav"] for u in tg])
    src = {f"real{i + 1}": (rd(PB.IN / sp / "in.f32"), [rd(p) for p in sorted((PB.IN / sp / "op").glob("*.f32"))]) for i, sp in enumerate(PB.SPS)}
    for v in ("em594", "em595", "ef000", "ef080"):
        src[f"fam_{v}"] = (lw(ROOT / f"data/snv_e1/{v}/t05.wav"), [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)])
    res: dict = {"k": k, "items": {}}
    S = {n: EV.Ens(t, gen) for n, t in SYS.items()}
    st = {n: s.structure("t", tg) for n, s in S.items()}
    for name, (x, ops) in src.items():
        u, enr = PB.utt(x), [PB.utt(o) for o in ops]
        lv = OG.speech_level([e["wav"] for e in enr])
        p0 = S["prod"].parts[0]
        pu = p0.prep(u)
        ws = {"shift": p0.render(pu, torch.from_numpy(pu["y"][:29])[None].to(EV.DEV), k, None)}
        ys = {}
        for n in SYS:
            ws[n], ys[n] = S[n].convert(u, enr, st[n], k, with_y=True)
        import pyworld
        xx = u["wav"].astype(np.float64)
        fo, tt = pyworld.harvest(xx, EV.SR, frame_period=5.0, f0_floor=60.0, f0_ceil=600.0)
        fo = pyworld.stonemask(xx, fo, tt, EV.SR)
        Tn = len(pu["f0"])
        fo = np.concatenate([fo, np.zeros(max(0, Tn - len(fo)))])[:Tn].astype(np.float32)
        fo = np.where(pu["f0"] > 0, np.where(fo > 0, fo, pu["f0"]), 0.0).astype(np.float32)
        pf = dict(pu) | {"f0": fo, "v": (fo > 0).astype(np.float32), "lf": np.where(fo > 0, np.log(np.maximum(fo, 1.0) / 200.0), 0.0).astype(np.float32)}
        ws["prod_f0o"] = p0.render(pf, ys["prod"][:, :29], k, None)
        r = {}
        e_s = None
        for n, w in ws.items():
            w = OG.apply(u["wav"], np.asarray(w, np.float32), lv, p_out)
            sf.write(out / "ear" / f"{name}_k{k}_{n}.wav", w, EV.SR, subtype="PCM_16")
            e = emb(w)
            e_s = e if n == "shift" else e_s
            r[n] = {"cos_t": round(float(e @ et), 3), "cos_s": round(float(e @ e_s), 3), "utmos": round(mos(w), 3)}
        v = pu["f0"] > 0
        ysrc = pu["y"][1:25][:, v]
        ytg = np.concatenate([t["y"][1:25][:, t["f0"] > 0] for t in tg], 1).mean(1, keepdims=True)
        for n in SYS:
            yo = ys[n][0, 1:25].detach().cpu().numpy()[:, v]
            r[n]["d_src"] = round(float(np.sqrt(((yo - ysrc) ** 2).sum(0)).mean()), 3)
            r[n]["d_tgmean"] = round(float(np.sqrt(((yo - ytg) ** 2).sum(0)).mean()), 3)
            r[n]["corr_dyn"] = round(float(np.corrcoef((yo - yo.mean(1, keepdims=True)).ravel(), (ysrc - ysrc.mean(1, keepdims=True)).ravel())[0, 1]), 3)
        r["src_cos_t"] = round(float(emb(u["wav"]) @ et), 3)
        r["src_utmos"] = round(mos(u["wav"]), 3)
        both = (pu["f0"] > 0) & (fo > 0)
        r["f0_dev_cent_med"] = round(float(np.median(1200 * np.abs(np.log2(pu["f0"][both] / fo[both])))), 1)
        r["f0_gross"] = round(float((np.abs(np.log2(pu["f0"][both] / fo[both])) > 0.2).mean()), 3)
        sf.write(out / "ear" / f"{name}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
        res["items"][name] = r
        print(name, json.dumps(r), flush=True)
    sf.write(out / "ear" / "target_ref.wav", tg[0]["wav"][: EV.SR * 15], EV.SR, subtype="PCM_16")
    p0 = S["prod"].parts[0]
    pt = p0.prep(tg[0])
    wt = np.asarray(p0.render(pt, torch.from_numpy(pt["y"][:29])[None].to(EV.DEV), 0, None), np.float32)
    sf.write(out / "ear" / "target_copy.wav", wt[: EV.SR * 15], EV.SR, subtype="PCM_16")
    res["target"] = {"utmos_ref": round(mos(tg[0]["wav"]), 3), "utmos_copy": round(mos(wt), 3), "cos_t_copy": round(float(emb(wt) @ et), 3)}
    print("target", json.dumps(res["target"]), flush=True)
    (out / f"shift_k{k}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
