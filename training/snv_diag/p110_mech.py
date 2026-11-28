"""段 P110 step b(学習なし): 回路が音素の無い発声で何を使っているかを差し替えで見る(1 系 snv_p30_c・合成の声)。
(1) 目標の構造の値を 0 にする / 大域の要約 z を源の声のものに替える → 出力の包絡の平均がどれだけ動くか。(2) 符号(演奏 a)の発話の平均から声を当てられる率(通常の発話 対 音素の無い発声)。
    cd training && SNV_DEV=cuda uv run --no-sync python snv_diag/p110_mech.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ["em594", "em595", "em596", "em597", "em598", "em599"]
NV = ["moan", "scream", "pant", "cry", "laugh", "sigh", "mixed"]


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "snv_p30_c"
    S = EV.System(tag, EV.load_gen())
    lw = lambda p: PB.utt(librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32))
    sp = lambda v, j: lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav")
    nv = lambda v, c, j=0: lw(ROOT / f"data/snv_e1nv/{v}/{c}_{j}.wav")
    tt = lambda a: torch.from_numpy(a)[None].to(EV.DEV)
    TG = "ef000"
    St = S.structure("t", [sp(TG, j) for j in (0, 1, 2)])
    mean = lambda y, v: y[1:25][:, v].mean(1)
    res: dict = {"tag": tag, "swap": {}, "leak": {}}
    acc: dict = {}
    for c in NV + ["speech"]:
        for s in SRC:
            enr = [sp(s, j) for j in (0, 1, 2)]
            Ss = S.structure("s" + s, enr)
            u = sp(s, 5) if c == "speech" else nv(s, c)
            t0 = sp(TG, 5) if c == "speech" else nv(TG, c, 0)
            t1 = sp(TG, 6) if c == "speech" else nv(TG, c, 1)
            v = u["f0"] > 0
            a = S.play(tt(u["y"]), tt(u["v"]), S.op_stat(enr))
            lf = tt(u["lf"])
            out = lambda Sx: S.C(a, lf, tt(u["v"]), Sx)[0].detach().cpu().numpy()
            m0, m1, ms = mean(t0["y"], t0["f0"] > 0), mean(t1["y"], t1["f0"] > 0), mean(u["y"], v)
            d = lambda y: float(np.linalg.norm(mean(y, v) - m0))
            var = {"normal": out(St), "v0": out(St | {"v": torch.zeros_like(St["v"])}), "z_src": out(St | {"z": Ss["z"]}), "kv_src": out(Ss | {"z": St["z"]}), "self": out(Ss)}
            for n, y in var.items():
                acc.setdefault((c != "speech", n), []).append(d(y))
            acc.setdefault((c != "speech", "floor"), []).append(float(np.linalg.norm(ms - m0)))
            acc.setdefault((c != "speech", "ceil"), []).append(float(np.linalg.norm(m1 - m0)))
            acc.setdefault((c != "speech", "normal_vs_v0"), []).append(float(np.linalg.norm(mean(var["normal"], v) - mean(var["v0"], v))))
    for isnv in (False, True):
        g = {n: round(float(np.mean(acc[(isnv, n)])), 3) for n in ("floor", "normal", "v0", "z_src", "kv_src", "self", "ceil", "normal_vs_v0")}
        g["prog"] = {n: round((g["floor"] - g[n]) / (g["floor"] - g["ceil"]), 2) for n in ("normal", "v0", "z_src", "kv_src", "self")}
        res["swap"]["nv" if isnv else "speech"] = g
        print("NV" if isnv else "通常", json.dumps(g), flush=True)
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"][:18]
    code = {}
    for vname in vs:
        enr = [sp(vname, j) for j in (0, 1, 2)]
        st = S.op_stat(enr)
        for key, u in [(f"sp{j}", sp(vname, j)) for j in (3, 4, 5, 6)] + [(c, nv(vname, c)) for c in NV]:
            a = S.play(tt(u["y"]), tt(u["v"]), st)[0].detach().cpu().numpy()
            code[(vname, key)] = a[:, u["f0"] > 0].mean(1)
    def ident(keys: list) -> float:
        ok = n = 0
        for k in keys:
            cen = {v: np.mean([code[(v, q)] for q in keys if q != k], 0) for v in vs}
            for v in vs:
                d = {w: np.linalg.norm(code[(v, k)] - cen[w]) for w in vs}
                ok += min(d, key=d.get) == v
                n += 1
        return round(ok / n, 3)
    res["leak"] = {"voices": len(vs), "chance": round(1 / len(vs), 3), "speech": ident([f"sp{j}" for j in (3, 4, 5, 6)]), "nv": ident(NV)}
    print("符号の平均から声を当てる率", json.dumps(res["leak"]), flush=True)
    (ROOT / f"results/snv_p110/mech_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
