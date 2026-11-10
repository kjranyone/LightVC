"""段 I10 の後の切り分け(学習なし): 本物の声の変換の内容の落ち(JVNV → 女声 k0 約 0.27・→ Irodori の男声 約 0.24・自己は 0.03)は、推論の符号器 E の出力に残る声ごとの一定のずれか。
推論で E の出力の内容の 59 次元を登録の定数で目標へ寄せる: shift = 有声・無声ごとに 操作者の登録の E の平均 → 目標の参照の E の平均 / std = 平均と標準偏差を合わせる。演技の 5 次元はそのまま。
目標の平均 = 目標の参照に E(その参照自身の E の構造)= C の構造の鍵と同じ計算。発話全体の統計は使わない(登録と参照の定数だけ)。
条件: jv_F_k0 / jv_M_k0 = JVNV の M1・M2(i10_jvnv_probe と同じ)→ A0F の目標 2 声 / 検証の Irodori の男声 2 声・ir_mF_k0 = 検証の Irodori の男声 8 声 → A0F・
real_F_k0 = 試験の区間のうち伴奏の無い 9 区間(外郎売 0〜5・namikawa 3〜5・登録 = 本物の登録)→ A0F。各条件 × {base, shift, std}。
    cd training && uv run --no-sync python snv_diag/i10_shift_probe.py [snv_i8_m13 snv_i9_aug snv_i10_combo] [out=results/snv_i10/shift_probe.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
DEV = EV.DEV
D_CT = 59


@torch.no_grad()
def e_stats(s: EV.System, us: list[dict], stat: dict) -> dict:
    a = torch.cat([s.Es(s.P11.xin(torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)), stat)[0][0, :D_CT] for u in us], 1)
    v = torch.from_numpy(np.concatenate([u["v"] for u in us])).to(DEV) > 0.5
    return {k: (a[:, m].mean(1), a[:, m].std(1) + 1e-4) for k, m in (("v", v), ("u", ~v)) if m.any()}


@torch.no_grad()
def convert(s: EV.System, src: dict, stat_utts: list[dict], S: dict, mode: str, mo: dict, mt: dict) -> np.ndarray:
    y = torch.from_numpy(src["y"])[None].to(DEV)
    v = torch.from_numpy(src["v"])[None].to(DEV)
    lf = torch.from_numpy(src["lf"])[None].to(DEV) * v
    a = s.play(y, v, s.op_stat(stat_utts)).clone()
    if mode != "base":
        for k, m in (("v", v[0] > 0.5), ("u", v[0] <= 0.5)):
            if k in mo and k in mt and m.any():
                x = a[0, :D_CT][:, m]
                x = (x - mo[k][0][:, None]) / mo[k][1][:, None] * mt[k][1][:, None] + mt[k][0][:, None] if mode == "std" else x + (mt[k][0] - mo[k][0])[:, None]
                a[0, :D_CT, m] = x
    yo = s.C(a, lf, v, S)
    cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
    f0 = torch.from_numpy(src["f0"])[None].to(DEV)
    return s.gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_i8_m13", "snv_i9_aug", "snv_i10_combo"]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y, lf, v = (torch.from_numpy(u[k])[None].to(DEV) for k in ("y", "lf", "v"))
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    ftg = meta["targets"][:2]
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    males = [v for v in split["val"] if v.startswith("em")]
    mtg = males[:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in ftg}
    trefs |= {v: [EV.utt(f"e1/{v}/{p.stem}") for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:3]] for v in mtg}
    trials: list = []
    for spk in ("M1", "M2"):
        en = [T.load_utt(T.FEAT / f"jvnv/{spk}_{e}_regular_02.npz") for e in EMOS]
        for e in EMOS:
            u = T.load_utt(T.FEAT / f"jvnv/{spk}_{e}_regular_01.npz")
            trials.append(("jv_F_k0", f"{spk}_{e}", u, en, ftg))
            trials.append(("jv_M_k0", f"{spk}_{e}", u, en, mtg))
    for v in males[2:10]:
        fs = sorted((T.FEAT / "e1" / v).glob("t*.npz"))
        trials.append(("ir_mF_k0", f"{v}/{fs[-1].stem}", T.load_utt(fs[-1]), [T.load_utt(p) for p in fs[:3]], ftg))
    enr = {sp: [EV.utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    for n in [f"uirou_{i}" for i in range(6)] + [f"namikawa_{i}" for i in (3, 4, 5)]:
        trials.append(("real_F_k0", n, EV.utt(f"real/{n}"), enr[n.rsplit("_", 1)[0]], ftg))
    systems = [EV.System(t, gen) for t in tags]
    tstat = {s.tag: {v: e_stats(s, rs, s.Es.structure(s.P11.xin(*EV.struct_inputs(rs)[:2]), EV.struct_inputs(rs)[2])) for v, rs in trefs.items()} for s in systems}
    rows: list = []
    ocache: dict = {}
    for cond, who, u, stat_utts, tg in trials:
        ref = tr(copy(u))
        for s in systems:
            ko = (s.tag, id(stat_utts))
            if ko not in ocache:
                ocache[ko] = e_stats(s, stat_utts, s.op_stat(stat_utts))
            for tgt in tg:
                S = s.structure(f"t|{tgt}", trefs[tgt])
                for mode in ("base", "shift", "std"):
                    out = tr(convert(s, u, stat_utts, S, mode, ocache[ko], tstat[s.tag][tgt]))
                    r = {"cond": cond, "mode": mode, "sys": s.tag, "src": who, "tgt": tgt, "cer": cer(ref, out), "ref": ref[:50], "out": out[:50]}
                    rows.append(r)
                    print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "conditions": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["conditions"][cond] = {}
        for t in tags:
            for mode in ("base", "shift", "std"):
                R = [r for r in rows if r["cond"] == cond and r["sys"] == t and r["mode"] == mode]
                res["conditions"][cond][f"{t}|{mode}"] = {"cer": round(float(np.mean([min(r["cer"], 1.0) for r in R])), 4), "cer_median": round(float(np.median([r["cer"] for r in R])), 4), "trials": len(R)}
    res["rows"] = rows
    print(json.dumps(res["conditions"], ensure_ascii=False, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i10/shift_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
