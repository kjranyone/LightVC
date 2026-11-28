"""段 P107(results/snv_p107/prereg.yaml・学習なし): 製品の出力の「端々の破綻」の在り処を、源の高さの跳び・有声の短い切れ・源に無い音量の跳びで数え、句の端と中で比べる。上位の地点を短い音に切り出す。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p107_edges.py [k=12]
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
import snv_out_guard as OG
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
G = "+mask:snv_p45_gate_t80+fb:anch"
SYS = "ens:" + "|".join(f"{t}{G}" for t in ("snv_p24_c", "snv_p30_c", "snv_p91e_c"))
FPS = 200


def runs(v: np.ndarray) -> list[tuple[int, int, bool]]:
    out, s = [], 0
    for t in range(1, len(v) + 1):
        if t == len(v) or v[t] != v[s]:
            out.append((s, t, bool(v[s])))
            s = t
    return out


def main() -> None:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    k = int(opt.get("k", 12))
    out = ROOT / "results/snv_p107"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    gen = EV.load_gen()
    S = EV.Ens(SYS, gen)
    tg = [PB.utt(rd(p)) for p in sorted((PB.IN / "tg").glob("*.f32"))]
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    src = {f"real{i + 1}": (rd(PB.IN / sp / "in.f32"), [rd(p) for p in sorted((PB.IN / sp / "op").glob("*.f32"))]) for i, sp in enumerate(PB.SPS)}
    for v in ("em594", "em595"):
        src[f"fam_{v}"] = (lw(ROOT / f"data/snv_e1/{v}/t05.wav"), [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)])
    res: dict = {"k": k, "items": {}}
    for name, (x, ops) in src.items():
        u, enr = PB.utt(x), [PB.utt(o) for o in ops]
        pu = S.prep(u)
        w, yo = S.convert(u, enr, st, k, with_y=True)
        w = OG.apply(u["wav"], np.asarray(w, np.float32), OG.speech_level([e["wav"] for e in enr]), p_out)
        f0, T = pu["f0"], len(pu["f0"])
        v = f0 > 0
        c_s, c_o = pu["y"][0] * 10, yo[0, 0].detach().cpu().numpy() * 10
        ev = {"f0jump": np.zeros(T, bool), "vshort": np.zeros(T, bool), "c0jump": np.zeros(T, bool)}
        lf = np.log2(np.maximum(f0, 1.0))
        ev["f0jump"][1:] = v[1:] & v[:-1] & (np.abs(np.diff(lf)) > 0.4)
        rs = runs(v)
        for i, (a, b, on) in enumerate(rs):
            if on and b - a <= 3:
                ev["vshort"][a:b] = True
            if (not on) and b - a <= 2 and 0 < i < len(rs) - 1:
                ev["vshort"][a:b] = True
        dd = np.abs(np.diff(c_o) - np.diff(c_s))
        loud = c_s > np.percentile(c_s, 30)
        ev["c0jump"][1:] = (dd > 6.0) & (loud[1:] | loud[:-1])
        edge = np.zeros(T, bool)
        ph = runs(np.convolve(v.astype(float), np.ones(21) / 21, "same") > 0.05)
        for a, b, on in ph:
            if on and b - a >= 30:
                edge[max(0, a - 10) : a + 30] = True
                edge[b - 40 : min(T, b + 10)] = True
        act = np.zeros(T, bool)
        for a, b, on in ph:
            if on:
                act[max(0, a - 10) : min(T, b + 10)] = True
        mid = act & ~edge
        r = {"sec": round(T / FPS, 1), "edge_sec": round(edge.sum() / FPS, 1), "mid_sec": round(mid.sum() / FPS, 1)}
        for n, e in ev.items():
            r[n] = {"edge_per_s": round(float(e[edge].sum() / max(1, edge.sum()) * FPS), 2), "mid_per_s": round(float(e[mid].sum() / max(1, mid.sum()) * FPS), 2)}
        score = np.convolve(3.0 * ev["f0jump"] + 1.0 * ev["vshort"] + 2.0 * ev["c0jump"], np.ones(60), "same")
        picks = []
        sc = score.copy()
        for _ in range(4):
            t = int(sc.argmax())
            if sc[t] <= 0:
                break
            a, b = max(0, t - 120), min(T, t + 120)
            kinds = [n for n, e in ev.items() if e[max(0, t - 30) : t + 30].any()]
            picks.append({"t": round(t / FPS, 2), "score": round(float(sc[t]), 1), "kinds": kinds, "edge": bool(edge[max(0, t - 30) : t + 30].any())})
            i = len(picks)
            sf.write(out / "clips" / f"{name}_k{k}_{i}_out.wav", w[a * 240 : b * 240], EV.SR, subtype="PCM_16")
            sf.write(out / "clips" / f"{name}_k{k}_{i}_src.wav", u["wav"][a * 240 : b * 240], EV.SR, subtype="PCM_16")
            sc[max(0, t - 200) : t + 200] = 0
        r["picks"] = picks
        res["items"][name] = r
        print(name, json.dumps(r, ensure_ascii=False), flush=True)
    (out / f"edges_k{k}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
