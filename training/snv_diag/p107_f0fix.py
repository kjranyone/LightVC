"""段 P107 step b(学習なし・診断): 源の高さの跳びの形(オクターブか・何フレーム続くか)を数え、先読みなしの直し(オクターブの折り返し + 短い保持)を掛けた高さで製品の出力を描き直す。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p107_f0fix.py [k=12]
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
import p107_edges as E
import snv_out_guard as OG
import snv_p1_eval as EV
from snv_a0_measure import to16

ROOT = Path(__file__).resolve().parent.parent.parent
TH, HOLD = 0.4, 4


def fix(f0: np.ndarray) -> np.ndarray:
    o, prev, held = f0.copy(), 0.0, 0
    for t in range(len(f0)):
        f = f0[t]
        if f <= 0:
            if held == 0:
                prev = 0.0
            continue
        if prev > 0 and abs(np.log2(f / prev)) > TH:
            c = min((f / 2, f * 2, f / 3, f * 3), key=lambda q: abs(np.log2(q / prev)))
            if abs(np.log2(c / prev)) <= TH:
                f = c
            elif held < HOLD:
                f, held = prev, held + 1
                o[t] = f
                continue
        o[t], prev, held = f, f, 0
    return o.astype(np.float32)


def jumps(f0: np.ndarray) -> dict:
    v = f0 > 0
    d = np.diff(np.log2(np.maximum(f0, 1.0)))
    j = v[1:] & v[:-1] & (np.abs(d) > TH)
    a = np.abs(d[j])
    return {"per_s": round(float(j.sum() / max(1, v.sum()) * E.FPS), 2), "octave_share": round(float(((a > 0.85) & (a < 1.15)).mean()), 2) if j.any() else 0.0, "n": int(j.sum())}


def main() -> None:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    k = int(opt.get("k", 12))
    out = ROOT / "results/snv_p107"
    (out / "ear").mkdir(parents=True, exist_ok=True)
    um = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(EV.DEV).eval()
    mos = lambda w: float(um(torch.from_numpy(to16(np.asarray(w, np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV), 16000).item())
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
    res: dict = {"k": k, "items": {}}
    for name, (x, ops) in src.items():
        u, enr = PB.utt(x), [PB.utt(o) for o in ops]
        lv = OG.speech_level([e["wav"] for e in enr])
        p0 = S.parts[0]
        pu = p0.prep(u)
        w, yo = S.convert(u, enr, st, k, with_y=True)
        f1 = fix(pu["f0"])
        pf = dict(pu) | {"f0": f1, "v": (f1 > 0).astype(np.float32), "lf": np.where(f1 > 0, np.log(np.maximum(f1, 1.0) / 200.0), 0.0).astype(np.float32)}
        w1 = p0.render(pf, yo[:, :29], k, None)
        r = {"jump_before": jumps(pu["f0"]), "jump_after": jumps(f1), "changed_frames": round(float((np.abs(f1 - pu["f0"]) > 1e-3).mean()), 3)}
        for n, q in (("prod", w), ("f0fix", w1)):
            q = OG.apply(u["wav"], np.asarray(q, np.float32), lv, p_out)
            sf.write(out / "ear" / f"{name}_k{k}_{n}.wav", q, EV.SR, subtype="PCM_16")
            r[f"utmos_{n}"] = round(mos(q), 3)
        res["items"][name] = r
        print(name, json.dumps(r), flush=True)
    (out / f"f0fix_k{k}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
