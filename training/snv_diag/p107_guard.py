"""段 P107 step c(学習なし・診断): 描画器に渡す高さと有声だけを絞る 2 つの形で、製品の回路の出力を描き直す(回路と符号の入力は変えない)。
g1 = 既に在る高さの番(snv_f0_guard.clean・登録の高さの中央値に対する上への外れを落とす / 直前の値で保つ)。
g2 = g1 + 追跡が「保持」(確からしさ 1.0)を 2 フレームを超えて続けた所を無声にする。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p107_guard.py [k=12]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import snv_f0_guard as FG
import snv_out_guard as OG
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent


def unhold(f0: np.ndarray, ap: np.ndarray, n: int = 2) -> np.ndarray:
    o, run = f0.copy(), 0
    for t in range(len(f0)):
        run = run + 1 if (f0[t] > 0 and ap[t] >= 0.999) else 0
        if run > n:
            o[t] = 0.0
    return o


def cond(pu: dict, f: np.ndarray) -> dict:
    return dict(pu) | {"f0": f, "v": (f > 0).astype(np.float32), "lf": np.where(f > 0, np.log(np.maximum(f, 1.0) / 200.0), 0.0).astype(np.float32)}


def main() -> None:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    k = int(opt.get("k", 12))
    out = ROOT / "results/snv_p107"
    (out / "guard").mkdir(parents=True, exist_ok=True)
    picks = json.load(open(out / "edges_k12.json"))["items"]
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
        _, yo = S.convert(u, enr, st, k, with_y=True)
        f0 = pu["f0"].astype(np.float32)
        med = FG.median_f0([p0.prep(e)["f0"] for e in enr])
        xx = u["wav"][: P.n_frames(len(u["wav"])) * P.H]
        raw = P.analyze(xx)
        n = min(len(f0), len(raw["f0"]))
        same = float((np.abs(raw["f0"][:n] - f0[:n]) < 0.5).mean())
        ap = np.ones(len(f0), np.float32)
        ap[:n] = np.where(np.abs(raw["f0"][:n] - f0[:n]) < 0.5, raw["ap"][:n], 0.0)
        g1 = FG.clean(f0, med)
        g2 = unhold(g1, ap)
        fs = {"prod": f0, "g1": g1, "g2": g2}
        r = {"med": round(med, 1), "ap_usable": round(same, 3)}
        for nm, f in fs.items():
            w = OG.apply(u["wav"], np.asarray(p0.render(cond(pu, f), yo[:, :29], k, None), np.float32), lv, p_out)
            sf.write(out / "guard" / f"{name}_k{k}_{nm}.wav", w, EV.SR, subtype="PCM_16")
            for i, pk in enumerate(picks[name]["picks"], 1):
                a, b = max(0, int(pk["t"] * 200) - 120), int(pk["t"] * 200) + 120
                sf.write(out / "guard" / f"{name}_k{k}_{i}_{nm}.wav", w[a * 240 : b * 240], EV.SR, subtype="PCM_16")
            r[nm] = {"voiced": round(float((f > 0).mean()), 3), "hi_frames": int((f > 2.2 * med).sum()), "jumps": E_j(f), "changed": round(float((np.abs(f - f0) > 0.5).mean()), 3)}
        res["items"][name] = r
        print(name, json.dumps(r), flush=True)
    (out / f"guard_k{k}.json").write_text(json.dumps(res, indent=1))


def E_j(f: np.ndarray) -> int:
    v = f > 0
    d = np.diff(np.log2(np.maximum(f, 1.0)))
    return int((v[1:] & v[:-1] & (np.abs(d) > 0.4)).sum())


if __name__ == "__main__":
    main()
